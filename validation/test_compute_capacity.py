"""Queue fairness, resource admission and disconnected-request contracts."""

import asyncio
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
import pytest

from compute_capacity import ComputeConcurrencyMiddleware


def controlled_app(gates, entered):
    app = FastAPI()

    @app.get("/api/v1/model")
    async def model():
        return {"model": "test"}

    @app.post("/api/v1/{operation:path}")
    async def calculate(request: Request):
        name = (await request.json())["name"]
        entered.append(name)
        await gates[name].wait()
        return {"name": name}
    return app


def test_shared_ip_sessions_get_two_slot_bursts_and_fair_turns():
    async def exercise():
        gates = {name: asyncio.Event() for name in ("a1", "a2", "a3", "a4", "b1", "b2")}
        entered, tasks = [], []
        limited = ComputeConcurrencyMiddleware(controlled_app(gates, entered), 2, 5)
        transport = httpx.ASGITransport(app=limited, client=("192.0.2.1", 1234))
        async with httpx.AsyncClient(transport=transport, base_url="https://test") as a, \
                httpx.AsyncClient(transport=transport, base_url="https://test") as b:
            model = await a.get("/api/v1/model")
            await b.get("/api/v1/model")
            assert a.cookies["oxytib_client"] != b.cookies["oxytib_client"]
            assert "HttpOnly" in model.headers["set-cookie"] and "Secure" in model.headers["set-cookie"]
            assert model.headers["cache-control"] == "no-store"
            cookie = a.cookies["oxytib_client"]
            await a.get("/api/v1/model")
            assert a.cookies["oxytib_client"] == cookie  # Reload does not reset identity.

            async def start(http, name):
                token = str(uuid4())
                task = asyncio.create_task(http.post("/api/v1/forward", json={"name": name},
                                          headers={"X-OXYTIB-Request-ID": token}))
                tasks.append(task)
                await until(lambda: token in limited.tracked)
                return task
            try:
                a1 = await start(a, "a1")
                a2 = await start(a, "a2")
                await until(lambda: entered == ["a1", "a2"])
                await start(a, "a3")
                await start(a, "a4")
                rejected = await a.post("/api/v1/forward", json={"name": "extra"})
                assert rejected.status_code == 429 and "waiting queue" in rejected.json()["detail"]
                b1 = await start(b, "b1")
                await start(b, "b2")
                gates["a1"].set()
                await a1
                await until(lambda: "b1" in entered)
                assert "a3" not in entered  # New colleague precedes older repeat requests.
                gates["a2"].set()
                await a2
                await until(lambda: "a3" in entered)
                gates["b1"].set()
                await b1
                await until(lambda: "b2" in entered)
                assert "a4" not in entered
            finally:
                for gate in gates.values():
                    gate.set()
                await asyncio.gather(*tasks)
            assert limited.active == 0 and not limited.waiting
    asyncio.run(exercise())


def test_rolling_budget_counts_both_slots_rechecks_queue_and_recovers():
    async def exercise():
        now = [0.0]
        gates = {name: asyncio.Event() for name in "abcd"}
        entered = []
        limited = ComputeConcurrencyMiddleware(controlled_app(gates, entered), 2, 5,
            client_budget_seconds=4, client_window_seconds=10, clock=lambda: now[0])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=limited), base_url="https://test") as http:
            await http.get("/api/v1/model")
            a = asyncio.create_task(http.post("/api/v1/forward", json={"name": "a"}))
            b = asyncio.create_task(http.post("/api/v1/forward", json={"name": "b"}))
            await until(lambda: len(entered) == 2)
            c = asyncio.create_task(http.post("/api/v1/forward", json={"name": "c"}))
            await until(lambda: len(limited.waiting) == 1)
            try:
                now[0] = 3.0  # Six occupied slot-seconds; no active task is interrupted.
                assert not a.done() and not b.done()
                gates["a"].set()
                await a
                rejected = await c
                assert rejected.status_code == 429 and int(rejected.headers["retry-after"]) > 0
                assert "c" not in entered and limited.active == 1
                gates["b"].set()
                await b
                again = await http.post("/api/v1/forward", json={"name": "d"})
                assert again.status_code == 429
                # Window [2,12] retains only one second from each completed job.
                now[0] = 12.0
                gates["d"].set()
                assert (await http.post("/api/v1/forward", json={"name": "d"})).status_code == 200
                assert not limited.tracked and not limited.waiting and limited.active == 0
            finally:
                for gate in gates.values():
                    gate.set()
                await asyncio.gather(a, b, c, return_exceptions=True)
    asyncio.run(exercise())


def test_client_records_and_history_are_bounded_and_expire():
    async def exercise():
        now = [0.0]
        gates = {"a": asyncio.Event()}
        gates["a"].set()
        limited = ComputeConcurrencyMiddleware(controlled_app(gates, []), 2, 2,
            max_clients=1, max_history=2, client_window_seconds=10, clock=lambda: now[0])
        transport = httpx.ASGITransport(app=limited)
        async with httpx.AsyncClient(transport=transport, base_url="https://test") as a, \
                httpx.AsyncClient(transport=transport, base_url="https://test") as b:
            await a.get("/api/v1/model")
            await b.get("/api/v1/model")
            for _ in range(2):
                assert (await a.post("/api/v1/forward", json={"name": "a"})).status_code == 200
            assert (await a.post("/api/v1/forward", json={"name": "a"})).status_code == 429
            assert (await b.post("/api/v1/forward", json={"name": "a"})).status_code == 503
            assert len(limited.clients) == 1
            now[0] = 11.0
            assert (await b.post("/api/v1/forward", json={"name": "a"})).status_code == 200
            assert len(limited.clients) == 1
    asyncio.run(exercise())


def test_invalid_cookies_cannot_select_someone_elses_budget():
    limited = ComputeConcurrencyMiddleware(None, 2, 1)
    scope = {"client": ("192.0.2.1", 1), "headers": [], "scheme": "https"}
    peer, cookie = limited._identity(scope)
    valid = dict(scope, headers=[(b"cookie", ("oxytib_client=" + cookie).encode())])
    assert limited._identity(valid)[0].startswith("session:")
    assert limited._identity(valid)[1] is None
    forged = dict(scope, headers=[(b"cookie", ("oxytib_client=" + cookie[:-1] + ("0" if cookie[-1] != "0" else "1")).encode())])
    assert limited._identity(forged)[0] == peer
    # Neither a made-up client ID nor arbitrary X-Forwarded-For changes fallback.
    spoof = dict(scope, headers=[(b"x-oxytib-client-id", b"other"), (b"x-forwarded-for", b"192.0.2.2")])
    assert limited._identity(spoof)[0] == peer
    invalid_text = dict(scope, headers=[(b"cookie", b'oxytib_client="' + b"a" * 32 + b'.\\351"')])
    assert limited._identity(invalid_text)[0] == peer


def test_exclusive_export_charges_all_reserved_slots():
    async def exercise():
        now = [0.0]
        gate = asyncio.Event()
        limited = ComputeConcurrencyMiddleware(controlled_app({"a": gate}, []), 2, 2,
            client_budget_seconds=4, client_window_seconds=10, clock=lambda: now[0])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=limited), base_url="https://test") as http:
            task = asyncio.create_task(http.post("/api/v1/export/result.xlsx", json={"name": "a"}))
            await until(lambda: limited.exclusive)
            now[0] = 3.0
            gate.set()
            await task
            assert (await http.post("/api/v1/forward", json={"name": "a"})).status_code == 429
    asyncio.run(exercise())


def test_cancelled_reserved_export_unblocks_other_clients():
    async def exercise():
        gates = {name: asyncio.Event() for name in "abc"}
        entered = []
        limited = ComputeConcurrencyMiddleware(controlled_app(gates, entered), 2, 5)
        transport = httpx.ASGITransport(app=limited)
        async with httpx.AsyncClient(transport=transport, base_url="https://test") as a, \
                httpx.AsyncClient(transport=transport, base_url="https://test") as b:
            await a.get("/api/v1/model")
            await b.get("/api/v1/model")
            running = asyncio.create_task(a.post("/api/v1/forward", json={"name": "a"}))
            await until(lambda: limited.active == 1)
            export = asyncio.create_task(b.post("/api/v1/export/result.xlsx", json={"name": "b"}))
            await until(lambda: limited.reserved is not None)
            waiting = asyncio.create_task(a.post("/api/v1/forward", json={"name": "c"}))
            try:
                await until(lambda: len(limited.waiting) == 2)
                assert entered == ["a"]
                export.cancel()
                await asyncio.gather(export, return_exceptions=True)
                await until(lambda: "c" in entered)
                assert limited.reserved is None and limited.active == 2
                assert "b" not in entered
            finally:
                for gate in gates.values():
                    gate.set()
                await asyncio.gather(running, export, waiting, return_exceptions=True)
            assert limited.active == 0 and not limited.waiting
    asyncio.run(exercise())


@pytest.mark.parametrize("setting,value", [
    ("client_budget_seconds", float("nan")), ("client_window_seconds", float("inf")),
    ("max_waiting_per_client", 0), ("max_clients", 0), ("max_history", 0),
])
def test_invalid_fair_use_configuration_is_rejected(setting, value):
    with pytest.raises(ValueError, match="client usage settings"):
        ComputeConcurrencyMiddleware(None, 2, 1, **{setting: value})


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

        limited = ComputeConcurrencyMiddleware(app, maximum=2, queue_timeout_seconds=2,
                                              max_waiting=3, max_waiting_per_client=4)
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
