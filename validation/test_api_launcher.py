"""API process ownership and graceful POSIX shutdown regressions."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("api_launcher", ROOT / "run_model.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_posix_api_replaces_launcher(monkeypatch):
    calls = []
    arguments = [sys.executable, str(ROOT / "code/web_api.py"), "--port", "8000"]

    class ReplacedProcess(Exception):
        pass

    def replace(executable, argv):
        calls.append((executable, argv))
        raise ReplacedProcess

    monkeypatch.setattr(launcher, "os", SimpleNamespace(
        name="posix", chdir=lambda path: calls.append(path), execv=replace,
    ))
    monkeypatch.setattr(launcher, "_run", lambda _: pytest.fail("spawned wrapper child"))
    with pytest.raises(ReplacedProcess):
        launcher._run_api(arguments)
    assert calls == [ROOT, (sys.executable, arguments)]


def test_windows_api_keeps_existing_launcher(monkeypatch):
    calls = []
    monkeypatch.setattr(launcher, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(launcher, "_run", lambda argv: calls.append(argv) or 7)
    arguments = [sys.executable, "server.py"]
    assert launcher._run_api(arguments) == 7
    assert calls == [arguments]


def test_api_command_uses_signal_safe_launcher(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_model.py", "api", "--host", "127.0.0.1", "--port", "8123"])
    calls = []
    monkeypatch.setattr(launcher, "_run_api", lambda argv: calls.append(argv) or 0)
    assert launcher.main() == 0
    assert calls == [[sys.executable, str(ROOT / "code/web_api.py"), "--host", "127.0.0.1", "--port", "8123"]]


@pytest.mark.parametrize("root_path", ["", "/oxytib"])
def test_real_http_prefix_stripping_assets_and_model(root_path, tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, OXYTIB_ROOT_PATH=root_path, FORWARDED_ALLOW_IPS="127.0.0.1")
    with (tmp_path / "http.log").open("w") as log:
        # Launch the server directly so Windows cleanup owns the server process.
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "code/web_api.py"), "--port", str(port)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                assert process.poll() is None
                try:
                    with urlopen(base + "/api/v1/health", timeout=1) as response:
                        assert json.load(response)["status"] == "ok"
                    break
                except OSError:
                    time.sleep(.1)
            else:
                pytest.fail("API startup timed out")
            # These paths are what a prefix-stripping proxy actually forwards.
            for path in ("/", "/assets/styles.css", "/assets/app.js", "/docs"):
                with urlopen(base + path, timeout=5) as response:
                    assert response.status == 200
            with urlopen(base + "/openapi.json", timeout=5) as response:
                schema = json.load(response)
                assert "/api/v1/forward" in schema["paths"]
                if root_path:
                    assert {"url": root_path} in schema["servers"]
            request = Request(base + "/api/v1/model", headers={
                "X-Forwarded-Proto": "https", "X-Forwarded-For": "198.51.100.25",
            })
            with urlopen(request, timeout=5) as response:
                cookie = response.headers["Set-Cookie"]
                assert "Secure" in cookie
                assert f"Path={root_path or '/'};" in cookie
            request = Request(base + "/api/v1/forward", data=b"{}",
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=15) as response:
                assert json.load(response)["calculation"] == "steady_forward"
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.mark.parametrize("trusted,peer,accepted", [
    ("192.0.2.10", "192.0.2.10", True),
    ("192.0.2.10", "192.0.2.11", False),
    ("172.30.241.0/29", "172.30.241.2", True),
    ("172.30.241.0/29", "172.30.242.2", False),
])
def test_uvicorn_only_accepts_headers_from_configured_proxies(monkeypatch, trusted, peer, accepted):
    import asyncio
    import uvicorn
    seen = {}
    async def endpoint(scope, receive, send):
        seen.update(client=scope["client"][0], scheme=scope["scheme"])
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", trusted)
    config = uvicorn.Config(endpoint, log_config=None)
    config.load()
    scope = {"type": "http", "client": (peer, 4321), "scheme": "http",
             "headers": [(b"x-forwarded-for", b"198.51.100.25"),
                         (b"x-forwarded-proto", b"https")]}
    asyncio.run(config.loaded_app(scope, None, None))
    assert seen == {"client": "198.51.100.25" if accepted else peer,
                    "scheme": "https" if accepted else "http"}


@pytest.mark.skipif(os.name != "posix", reason="SIGTERM/exec regression requires POSIX")
def test_real_api_sigterm_completes_server_shutdown(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log_path = tmp_path / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "run_model.py"), "api", "--port", str(port)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                assert process.poll() is None, log_path.read_text()
                try:
                    with urlopen(f"http://127.0.0.1:{port}/api/v1/health", timeout=1) as response:
                        assert json.load(response)["status"] == "ok"
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                pytest.fail("API startup timed out: " + log_path.read_text())
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=10) in (0, -signal.SIGTERM)
            output = log_path.read_text()
            assert f"Started server process [{process.pid}]" in output
            assert "Application shutdown complete." in output
            assert f"Finished server process [{process.pid}]" in output
        finally:
            # Remove only this test's isolated process group if a regression left children.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
