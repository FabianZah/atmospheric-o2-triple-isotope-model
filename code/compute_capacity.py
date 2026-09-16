"""Bounded FIFO admission for calculations and exclusive workbook exports."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import re

from starlette.responses import JSONResponse


@dataclass(eq=False)
class _Ticket:
    token: str | None
    exclusive: bool
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    started: bool = False


class ComputeConcurrencyMiddleware:
    """Keep a bounded queue; serve read-only routes independently of compute slots."""

    def __init__(self, app, maximum: int, queue_timeout_seconds: float,
                 max_waiting: int = 4) -> None:
        if maximum < 1 or queue_timeout_seconds <= 0 or max_waiting < 0:
            raise ValueError("invalid calculation capacity settings")
        self.app = app
        self.maximum = maximum
        self.queue_timeout_seconds = queue_timeout_seconds
        self.max_waiting = max_waiting
        self.waiting: list[_Ticket] = []
        self.active = 0
        self.exclusive = False
        self.tracked: dict[str, _Ticket] = {}

    def _drain(self) -> None:
        # These state transitions contain no awaits and run on the ASGI event loop.
        while self.waiting and self.active < self.maximum and not self.exclusive:
            ticket = self.waiting[0]
            if ticket.exclusive and self.active:
                break
            self.waiting.pop(0)
            self.active += 1
            self.exclusive = ticket.exclusive
            ticket.started = True
            ticket.ready.set()

    def _finish(self, ticket: _Ticket) -> None:
        if ticket.started:
            self.active -= 1
            if ticket.exclusive:
                self.exclusive = False
        else:
            self.waiting.remove(ticket)
        if ticket.token:
            self.tracked.pop(ticket.token, None)
        self._drain()

    @staticmethod
    async def _busy(scope, receive, send, detail):
        await JSONResponse(status_code=503, content={"detail": detail},
                           headers={"Retry-After": "2"})(scope, receive, send)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        root = scope.get("root_path", "")
        if root and path.startswith(root + "/"):
            path = path[len(root):]
        if scope.get("method") == "GET" and path.startswith("/api/v1/compute-status/"):
            token = path.removeprefix("/api/v1/compute-status/")
            ticket = self.tracked.get(token)
            status = "unknown" if ticket is None else "running" if ticket.started else "waiting"
            await JSONResponse({"state": status}, headers={"Cache-Control": "no-store"})(scope, receive, send)
            return
        if scope.get("method") != "POST" or not path.startswith("/api/v1/"):
            await self.app(scope, receive, send)
            return

        # Buffer the size-limited body so disconnected queued requests can be dropped.
        messages = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            messages.append(message)
            if not message.get("more_body", False):
                break
        headers = dict(scope.get("headers", []))
        token = headers.get(b"x-oxytib-request-id", b"").decode("ascii", errors="replace")
        if token and not re.fullmatch(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}", token):
            await JSONResponse({"detail": "invalid calculation request ID"}, status_code=400)(scope, receive, send)
            return
        if token in self.tracked:
            await JSONResponse({"detail": "calculation request ID is already active"}, status_code=409)(scope, receive, send)
            return
        exclusive = path.startswith("/api/v1/export/")
        immediate = not self.waiting and not self.exclusive and self.active < self.maximum
        immediate = immediate and (not exclusive or self.active == 0)
        if not immediate and len(self.waiting) >= self.max_waiting:
            await self._busy(scope, receive, send, "Calculation queue is full; please try again shortly.")
            return
        ticket = _Ticket(token or None, exclusive)
        self.waiting.append(ticket)
        if token:
            self.tracked[token] = ticket
        self._drain()

        async def replay_body():
            return messages.pop(0) if messages else await receive()

        async def disconnected():
            while (await receive())["type"] != "http.disconnect":
                pass

        try:
            if not ticket.started:
                ready = asyncio.create_task(ticket.ready.wait())
                closed = asyncio.create_task(disconnected())
                try:
                    done, _ = await asyncio.wait(
                        (ready, closed), timeout=self.queue_timeout_seconds,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if closed in done:
                        return
                    if ready not in done:
                        await self._busy(scope, receive, send,
                                         "The calculation queue wait limit was reached; please try again shortly.")
                        return
                finally:
                    ready.cancel()
                    closed.cancel()
                    await asyncio.gather(ready, closed, return_exceptions=True)
            await self.app(scope, replay_body, send)
        finally:
            self._finish(ticket)
