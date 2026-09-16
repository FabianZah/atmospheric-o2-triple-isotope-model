"""Queue fairness, resource admission and disconnected-request contracts."""

import asyncio
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
import pytest

from compute_capacity import ComputeConcurrencyMiddleware


async def until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.001)
    await asyncio.wait_for(poll(), timeout=2)


def test_two_slots_fifo_and_exclusive_exports():
    async def exercise():
        app = FastAPI()
        entered = []
        gates = {name: asyncio.Event() for name in "abcdef"}

        @app.post("/api/v1/{operation:path}")
        async def calculation(request: Request):
            name = (await request.json())["name"]
            entered.append(name)
            await gates[name].wait()
            return {"name": name}

        limited = ComputeConcurrencyMiddleware(app, maximum=2, queue_timeout_seconds=2, max_waiting=3)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=limited), base_url="http://test") as http:
            tokens = {name: str(uuid4()) for name in gates}
            tasks = []

            async def start(name, export=False):
                task = asyncio.create_task(http.post(
                    "/api/v1/export/result.xlsx" if export else "/api/v1/forward",
                    json={"name": name}, headers={"X-OXYTIB-Request-ID": tokens[name]},
                ))
                tasks.append(task)
                await until(lambda: tokens[name] in limited.tracked)
                return task

            async def status(name):
                response = await http.get("/api/v1/compute-status/" + tokens[name])
                assert response.headers["cache-control"] == "no-store"
                return response.json()["state"]

            try:
                await start("a")
                await start("b")
                await until(lambda: len(entered) == 2)
                assert limited.active == 2
                assert await status("a") == "running"
                await start("c", export=True)
                await start("d")
                assert await status("c") == "waiting"
                gates["a"].set()
                await tasks[0]
                assert entered == ["a", "b"]  # Export waits for both slots; d cannot jump it.
                gates["b"].set()
                await tasks[1]
                await until(lambda: "c" in entered)
                assert entered == ["a", "b", "c"]
                assert limited.active == 1 and limited.exclusive
                assert await status("d") == "waiting"
                await start("e", export=True)
                await start("f")
                rejected = await http.post("/api/v1/forward", json={"name": "overflow"})
                assert rejected.status_code == 503 and "queue is full" in rejected.json()["detail"]
                gates["c"].set()
                await tasks[2]
                await until(lambda: "d" in entered)
                assert "e" not in entered
                gates["d"].set()
                await tasks[3]
                await until(lambda: "e" in entered)
                assert "f" not in entered
                gates["e"].set()
                await tasks[4]
                await until(lambda: "f" in entered)
                gates["f"].set()
                responses = await asyncio.gather(*tasks)
                assert all(r.status_code == 200 for r in responses)
                assert entered == list("abcdef")
                assert limited.active == 0 and not limited.waiting and not limited.tracked
                assert await status("f") == "unknown"
            finally:
                for gate in gates.values():
                    gate.set()
                await asyncio.gather(*tasks, return_exceptions=True)
    asyncio.run(exercise())


@pytest.mark.parametrize("disconnect", [True, False])
def test_waiting_disconnect_or_timeout_removes_ticket_and_preserves_active_slot(disconnect):
    async def exercise():
        gate = asyncio.Event()
        called = []

        async def application(scope, receive, send):
            called.append(scope["path"])
            await gate.wait()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        limited = ComputeConcurrencyMiddleware(application, maximum=1, queue_timeout_seconds=0.05)
        body = {"type": "http.request", "body": b"{}", "more_body": False}
        incoming1, incoming2 = asyncio.Queue(), asyncio.Queue()
        incoming1.put_nowait(body)
        incoming2.put_nowait(body)
        sent = []
        async def send(message):
            sent.append(message)
        scope = {"type": "http", "method": "POST", "path": "/api/v1/forward", "headers": []}
        first = asyncio.create_task(limited(scope, incoming1.get, send))
        await until(lambda: limited.active == 1)
        second = asyncio.create_task(limited(scope, incoming2.get, send))
        try:
            await until(lambda: len(limited.waiting) == 1)
            if disconnect:
                incoming2.put_nowait({"type": "http.disconnect"})
            await asyncio.wait_for(second, 2)
            assert limited.active == 1 and not limited.waiting
            assert len(called) == 1
            if disconnect:
                assert not sent
            else:
                assert sent[0]["status"] == 503
            gate.set()
            await first
            assert limited.active == 0
        finally:
            gate.set()
            await asyncio.gather(first, second, return_exceptions=True)
    asyncio.run(exercise())


def test_request_ids_and_prefix_status_are_scoped_and_bounded():
    async def exercise():
        gate = asyncio.Event()
        app = FastAPI()
        @app.post("/api/v1/forward")
        async def calculate():
            await gate.wait()
            return {"ok": True}
        limited = ComputeConcurrencyMiddleware(app, maximum=1, queue_timeout_seconds=2)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=limited), base_url="http://test") as http:
            token = str(uuid4())
            first = asyncio.create_task(http.post("/api/v1/forward", json={}, headers={"X-OXYTIB-Request-ID": token}))
            try:
                await until(lambda: token in limited.tracked)
                duplicate = await http.post("/api/v1/forward", json={}, headers={"X-OXYTIB-Request-ID": token})
                assert duplicate.status_code == 409
                invalid = await http.post("/api/v1/forward", json={}, headers={"X-OXYTIB-Request-ID": "bad"})
                assert invalid.status_code == 400
                other = await http.get("/api/v1/compute-status/" + str(uuid4()))
                assert other.json() == {"state": "unknown"}
                prefixed = FastAPI()
                prefixed.mount("/oxytib", limited)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=prefixed), base_url="http://test") as mounted:
                    assert (await mounted.get("/oxytib/api/v1/compute-status/" + token)).json() == {"state": "running"}
            finally:
                gate.set()
                await first
    asyncio.run(exercise())
