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
from urllib.request import urlopen

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
