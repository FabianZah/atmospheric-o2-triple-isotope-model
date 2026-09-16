"""Bounded fair admission and session usage budgets for model calculations."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
import hashlib
import hmac
from http.cookies import CookieError, SimpleCookie
import math
import re
import secrets
import time

from starlette.responses import JSONResponse


@dataclass(eq=False)
class _Ticket:
    token: str | None
    exclusive: bool
    client: str = ""
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    started: bool = False
    started_at: float = 0.0
    retry_after: int = 0


@dataclass
class _Client:
    history: deque = field(default_factory=deque)
    last_turn: int = 0


class ComputeConcurrencyMiddleware:
    """Use spare slots freely, then prioritize clients with fewer active jobs."""

    def __init__(self, app, maximum: int, queue_timeout_seconds: float,
                 max_waiting: int = 4, max_waiting_per_client: int = 2,
                 client_budget_seconds: float = 360.0,
                 client_window_seconds: float = 600.0,
                 max_clients: int = 1024, max_history: int = 256,
                 clock=time.monotonic) -> None:
        if maximum < 1 or queue_timeout_seconds <= 0 or max_waiting < 0:
            raise ValueError("invalid calculation capacity settings")
        if (max_waiting_per_client < 1 or max_clients < 1 or max_history < 1
                or not math.isfinite(client_budget_seconds) or client_budget_seconds <= 0
                or not math.isfinite(client_window_seconds) or client_window_seconds <= 0):
            raise ValueError("invalid client usage settings")
        self.app = app
        self.maximum = maximum
        self.queue_timeout_seconds = queue_timeout_seconds
        self.max_waiting = max_waiting
        self.waiting: list[_Ticket] = []
        self.active = 0
        self.exclusive = False
        self.tracked: dict[str, _Ticket] = {}
        self.running: list[_Ticket] = []
        self.clients: dict[str, _Client] = {}
        self.max_waiting_per_client = max_waiting_per_client
        self.client_budget_seconds = client_budget_seconds
        self.client_window_seconds = client_window_seconds
        self.max_clients = max_clients
        self.max_history = max_history
        self.clock = clock
        self.turn = 0
        self.reserved: _Ticket | None = None
        self.cookie_key = secrets.token_bytes(32)

    def _identity(self, scope):
        headers = dict(scope.get("headers", []))
        cookie = SimpleCookie()
        try:
            cookie.load(headers.get(b"cookie", b"").decode("latin1"))
            value = cookie["oxytib_client"].value
            identifier, signature = value.split(".")
            if (re.fullmatch(r"[0-9a-f]{32}", identifier)
                    and re.fullmatch(r"[0-9a-f]{64}", signature) and hmac.compare_digest(
                    signature, hmac.new(self.cookie_key, identifier.encode(), hashlib.sha256).hexdigest())):
                return "session:" + identifier, None
        except (CookieError, KeyError, ValueError):
            pass
        identifier = secrets.token_hex(16)
        signature = hmac.new(self.cookie_key, identifier.encode(), hashlib.sha256).hexdigest()
        # API clients without a browser cookie share the peer-IP budget. Do not
        # trust an arbitrary client header or parse forwarded IPs here.
        peer = str((scope.get("client") or ("unknown",))[0])
        key = hmac.new(self.cookie_key, peer.encode(), hashlib.sha256).hexdigest()
        return "peer:" + key, identifier + "." + signature

    def _prune(self, now):
        occupied = {t.client for t in self.running + self.waiting}
        for key, client in list(self.clients.items()):
            while client.history and client.history[0][1] <= now - self.client_window_seconds:
                client.history.popleft()
            if not client.history and key not in occupied:
                del self.clients[key]

    def _retry_after(self, key, now):
        client = self.clients[key]
        while client.history and client.history[0][1] <= now - self.client_window_seconds:
            client.history.popleft()
        intervals = list(client.history) + [
            (t.started_at, now, self.maximum if t.exclusive else 1)
            for t in self.running if t.client == key
        ]
        def usage(at):
            cutoff = at - self.client_window_seconds
            return sum(max(0.0, end - max(start, cutoff)) * weight for start, end, weight in intervals)
        if len(client.history) >= self.max_history:
            return max(1, math.ceil(client.history[0][1] + self.client_window_seconds - now))
        if usage(now) < self.client_budget_seconds:
            return 0
        # A lower-bound retry time: already-running work continues to accrue use.
        low, high = now, now + self.client_window_seconds
        for _ in range(32):
            middle = (low + high) / 2
            if usage(middle) >= self.client_budget_seconds:
                low = middle
            else:
                high = middle
        return max(1, math.ceil(high - now))

    def _drain(self) -> None:
        # These state transitions contain no awaits and run on the ASGI event loop.
        while self.waiting and self.active < self.maximum and not self.exclusive:
            ticket = self.reserved or min(self.waiting, key=lambda t: (
                sum(r.client == t.client for r in self.running),
                self.clients[t.client].last_turn,
            ))
            retry = self._retry_after(ticket.client, self.clock())
            if retry:
                self.waiting.remove(ticket)
                self.reserved = None
                ticket.retry_after = retry
                ticket.ready.set()
                continue
            if ticket.exclusive and self.active:
                self.reserved = ticket  # Drain for this export; new arrivals cannot starve it.
                break
            self.waiting.remove(ticket)
            self.reserved = None
            self.active += 1
            self.exclusive = ticket.exclusive
            ticket.started = True
            ticket.started_at = self.clock()
            self.running.append(ticket)
            self.turn += 1
            self.clients[ticket.client].last_turn = self.turn
            ticket.ready.set()

    def _finish(self, ticket: _Ticket) -> None:
        if ticket.started:
            self.running.remove(ticket)
            self.clients[ticket.client].history.append((
                ticket.started_at, self.clock(), self.maximum if ticket.exclusive else 1,
            ))
            self.active -= 1
            if ticket.exclusive:
                self.exclusive = False
        elif ticket in self.waiting:
            self.waiting.remove(ticket)
        if self.reserved is ticket:
            self.reserved = None
        if ticket.token:
            self.tracked.pop(ticket.token, None)
        self._drain()

    @staticmethod
    async def _busy(scope, receive, send, detail):
        await JSONResponse(status_code=503, content={"detail": detail},
                           headers={"Retry-After": "2"})(scope, receive, send)

    @staticmethod
    async def _cooldown(scope, receive, send, retry):
        await JSONResponse(status_code=429, content={"detail":
            f"Calculation allowance temporarily used. Please try again in about {retry} seconds. Running calculations will finish normally."},
            headers={"Retry-After": str(retry), "Cache-Control": "no-store"})(scope, receive, send)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        root = scope.get("root_path", "")
        if root and path.startswith(root + "/"):
            path = path[len(root):]
        if scope.get("method") == "GET" and path == "/api/v1/model":
            _, new_cookie = self._identity(scope)
            async def send_model(message):
                if new_cookie and message["type"] == "http.response.start":
                    cookie = SimpleCookie()
                    cookie["oxytib_client"] = new_cookie
                    cookie["oxytib_client"]["path"] = root or "/"
                    cookie["oxytib_client"]["httponly"] = True
                    cookie["oxytib_client"]["samesite"] = "Lax"
                    cookie["oxytib_client"]["max-age"] = 86400
                    cookie["oxytib_client"]["secure"] = scope.get("scheme") == "https"
                    message = dict(message, headers=[
                        (k, v) for k, v in message.get("headers", []) if k.lower() != b"cache-control"
                    ] + [(b"set-cookie", cookie.output(header="").strip().encode("latin1")),
                         (b"cache-control", b"no-store")])
                await send(message)
            await self.app(scope, receive, send_model)
            return
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
        client, _ = self._identity(scope)
        self._prune(self.clock())
        if client not in self.clients:
            if len(self.clients) >= self.max_clients:
                await self._busy(scope, receive, send, "Calculation admission is temporarily busy; please try again shortly.")
                return
            self.clients[client] = _Client()
        retry = self._retry_after(client, self.clock())
        if retry:
            await self._cooldown(scope, receive, send, retry)
            return
        exclusive = path.startswith("/api/v1/export/")
        immediate = not self.waiting and not self.exclusive and self.active < self.maximum
        immediate = immediate and (not exclusive or self.active == 0)
        if not immediate and sum(t.client == client for t in self.waiting) >= self.max_waiting_per_client:
            await JSONResponse(status_code=429, content={"detail":
                "Your waiting queue is full. Please wait for a queued calculation to finish."},
                headers={"Retry-After": "2"})(scope, receive, send)
            return
        if not immediate and len(self.waiting) >= self.max_waiting:
            await self._busy(scope, receive, send, "Calculation queue is full; please try again shortly.")
            return
        ticket = _Ticket(token or None, exclusive, client)
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
            if not ticket.ready.is_set():
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
            if ticket.retry_after:
                await self._cooldown(scope, receive, send, ticket.retry_after)
            else:
                await self.app(scope, replay_body, send)
        finally:
            self._finish(ticket)
