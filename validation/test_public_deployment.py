"""Production deployment contract and remote-verifier tests."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml

import verify_public_deployment as verifier


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)


def test_production_compose_is_private_bounded_and_read_only() -> None:
    compose = yaml.safe_load(
        (ROOT / "deploy" / "compose.production.yaml").read_text(encoding="utf-8")
    )
    api = compose["services"]["model-api"]
    assert "ports" not in api
    assert api["expose"] == ["8000"]
    assert api["read_only"] is True
    assert api["tmpfs"] == ["/tmp:rw,noexec,nosuid,size=256m"]
    assert api["cap_drop"] == ["ALL"]
    assert api["security_opt"] == ["no-new-privileges:true"]
    assert api["pids_limit"] == "${OXYTIB_PIDS:-256}"
    assert api["mem_limit"] == "${OXYTIB_MEMORY:-2g}"
    assert api["cpus"] == "${OXYTIB_CPUS:-2.0}"
    assert api["networks"] == ["backend"]
    assert api["environment"]["OXYTIB_MAX_REQUEST_BYTES"] == (
        "${OXYTIB_MAX_REQUEST_BYTES:-1048576}"
    )
    assert api["environment"]["OXYTIB_MAX_COMPUTE_REQUESTS"] == (
        "${OXYTIB_MAX_COMPUTE_REQUESTS:-2}"
    )
    assert compose["networks"]["backend"]["internal"] is True

    proxy = compose["services"]["caddy"]
    assert proxy["depends_on"]["model-api"]["condition"] == "service_healthy"
    assert set(proxy["ports"]) == {"80:80", "443:443", "443:443/udp"}
    assert proxy["image"] == "${CADDY_IMAGE:-caddy:2.11.4-alpine}"
    assert "./caddy:/etc/caddy:ro" in proxy["volumes"]
    assert proxy["read_only"] is True
    assert proxy["pids_limit"] == 128
    assert proxy["mem_limit"] == "256m"
    assert proxy["cpus"] == "0.5"
    assert proxy["networks"] == ["frontend", "backend"]
    caddyfile = (ROOT / "deploy" / "caddy" / "Caddyfile").read_text(
        encoding="utf-8"
    )
    assert "max_size 1MB" in caddyfile
    assert "Content-Security-Policy" in caddyfile


def test_environment_template_and_ignore_policy_are_safe() -> None:
    template = (ROOT / "deploy" / ".env.production.example").read_text(
        encoding="utf-8"
    )
    assert "OXYTIB_DOMAIN=model.example.org" in template
    assert "OXYTIB_CORS_ORIGINS=" in template
    assert "CADDY_IMAGE=caddy:2.11.4-alpine" in template
    assert "OXYTIB_MAX_REQUEST_BYTES=1048576" in template
    assert "OXYTIB_MAX_COMPUTE_REQUESTS=2" in template
    traefik_template = (ROOT / "deploy" / ".env.traefik.example").read_text(
        encoding="utf-8"
    )
    assert "OXYTIB_ROOT_PATH=" in traefik_template
    assert "OXYTIB_ROOT_PATH=/oxytib" not in traefik_template
    assert "PASSWORD=" not in template
    assert "TOKEN=" not in template
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env.*" in ignore
    assert "!deploy/.env.production.example" in ignore
    assert "!deploy/.env.traefik.example" in ignore


def test_traefik_compose_is_staged_bounded_and_prefix_aware() -> None:
    compose = yaml.safe_load(
        (ROOT / "deploy" / "compose.traefik.yaml").read_text(encoding="utf-8")
    )
    api = compose["services"]["model-api"]
    assert api["ports"] == ["127.0.0.1:${OXYTIB_LOCAL_PORT:-18000}:8000"]
    assert api["read_only"] is True
    assert api["cap_drop"] == ["ALL"]
    assert api["security_opt"] == ["no-new-privileges:true"]
    assert api["pids_limit"] == "${OXYTIB_PIDS:-128}"
    assert api["mem_limit"] == "${OXYTIB_MEMORY:-1536m}"
    assert api["cpus"] == "${OXYTIB_CPUS:-0.75}"
    assert api["networks"] == ["traefik-global-proxy"]
    assert api["environment"]["OXYTIB_MAX_REQUEST_BYTES"] == (
        "${OXYTIB_MAX_REQUEST_BYTES:-1048576}"
    )
    assert api["environment"]["OXYTIB_MAX_COMPUTE_REQUESTS"] == (
        "${OXYTIB_MAX_COMPUTE_REQUESTS:-1}"
    )
    assert api["environment"]["OXYTIB_ROOT_PATH"] == "${OXYTIB_ROOT_PATH:-}"
    assert api["tmpfs"] == ["/tmp:rw,noexec,nosuid,size=256m"]
    assert compose["networks"]["traefik-global-proxy"] == {
        "external": True,
        "name": "traefik-global-proxy",
    }

    labels = dict(label.split("=", 1) for label in api["labels"])
    assert labels["traefik.enable"] == "${OXYTIB_TRAEFIK_ENABLE:-false}"
    assert "Path(`/oxytib`)" in labels["traefik.http.routers.oxytib.rule"]
    assert "PathPrefix(`/oxytib/`)" in labels["traefik.http.routers.oxytib.rule"]
    assert labels["traefik.http.routers.oxytib.middlewares"] == (
        "oxytib-slash,oxytib-rate,oxytib-inflight,oxytib-security,oxytib-strip"
    )
    assert labels["traefik.http.middlewares.oxytib-strip.stripprefix.prefixes"] == (
        "/oxytib"
    )
    assert labels["traefik.http.services.oxytib.loadbalancer.server.port"] == "8000"
    assert labels["traefik.http.middlewares.oxytib-rate.ratelimit.average"] == (
        "${OXYTIB_RATE_AVERAGE:-30}"
    )
    assert labels["traefik.http.middlewares.oxytib-rate.ratelimit.period"] == (
        "${OXYTIB_RATE_PERIOD:-1m}"
    )
    assert labels["traefik.http.middlewares.oxytib-rate.ratelimit.burst"] == (
        "${OXYTIB_RATE_BURST:-10}"
    )
    assert labels["traefik.http.middlewares.oxytib-inflight.inflightreq.amount"] == (
        "${OXYTIB_INFLIGHT_REQUESTS:-4}"
    )
    assert labels[
        "traefik.http.middlewares.oxytib-security.headers.contenttypenosniff"
    ] == "true"
    assert "frame-ancestors 'none'" in labels[
        "traefik.http.middlewares.oxytib-security.headers.contentsecuritypolicy"
    ]
    assert all("legacy" not in key for key in labels)

    template = (ROOT / "deploy" / ".env.traefik.example").read_text(
        encoding="utf-8"
    )
    assert "OXYTIB_TRAEFIK_ENABLE=false" in template
    assert "OXYTIB_HOST=model.example.org" in template
    assert "OXYTIB_MEMORY=1536m" in template
    assert "OXYTIB_MAX_REQUEST_BYTES=1048576" in template
    assert "OXYTIB_MAX_COMPUTE_REQUESTS=1" in template
    assert "OXYTIB_RATE_AVERAGE=30" in template
    assert "OXYTIB_INFLIGHT_REQUESTS=4" in template
    assert "PASSWORD=" not in template
    assert "TOKEN=" not in template


def test_tracked_public_text_has_no_legacy_product_name() -> None:
    legacy_name = "atmo" + "-mod"
    text_suffixes = {
        ".bib", ".cff", ".css", ".html", ".js", ".json", ".md", ".py",
        ".ris", ".toml", ".txt", ".yaml", ".yml",
    }
    matches = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        if any(part in {".git", ".pytest_cache", "__pycache__"} for part in path.parts):
            continue
        if legacy_name in path.read_text(encoding="utf-8", errors="ignore").lower():
            matches.append(str(path.relative_to(ROOT)))
    assert matches == []


def test_production_image_uses_pinned_python_and_api_dependencies() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.startswith("FROM python:3.12.14-slim-bookworm\n")
    assert "requirements-api-lock.txt" in dockerfile
    assert "requirements-api.txt" not in dockerfile
    lock = (ROOT / "code" / "requirements-api-lock.txt").read_text(
        encoding="utf-8"
    )
    required_pins = {
        "fastapi==0.129.0",
        "numpy==1.26.4",
        "pydantic==2.12.5",
        "scipy==1.13.1",
        "uvicorn==0.40.0",
    }
    assert required_pins <= set(lock.splitlines())
    assert ">=" not in lock
    assert "~=" not in lock


def test_container_ci_exercises_shared_server_limits_and_dense_export() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["container"]["steps"]
    start = next(step["run"] for step in steps if step.get("name") == "Start hardened application container")
    assert "--memory 768m --memory-swap 768m" in start
    assert "--cpus 0.75" in start
    assert "--env OXYTIB_MAX_COMPUTE_REQUESTS=1" in start
    assert "size=256m" in start
    load = next(step for step in steps if step.get("name") == "Verify dense posterior export within shared-server limits")
    assert "--check-dense-export" in load["run"]
    assert load["timeout-minutes"] == 12
    report = next(step for step in steps if step.get("name") == "Report container memory and exit state")
    assert report["if"] == "always()"
    assert "memory.peak" in report["run"]
    assert "memory.events" in report["run"]
    assert "{{json .State}}" in report["run"]


@pytest.mark.parametrize("failure", [None, "coarse", "identity", "worksheets", "health"])
def test_dense_export_verifier_checks_real_workload_and_workbook(monkeypatch, failure) -> None:
    calls = []

    def fake_json(base_url, path, payload=None, *, timeout=30.0):
        calls.append((base_url, path, payload, timeout))
        if path == "/api/v1/health":
            return {"status": "bad" if failure == "health" else "ok",
                    "publication_model_id": verifier.EXPECTED_MODEL_ID}
        return {
            "publication_model_id": "wrong" if failure == "identity" else verifier.EXPECTED_MODEL_ID,
            "result": {"field_shape": [17, 17] if failure == "coarse" else [833, 961],
                       "posterior_median": 4086.0},
        }

    def fake_bytes(base_url, path, payload=None, *, timeout=30.0):
        calls.append((base_url, path, payload, timeout))
        names = ["Summary", "Posterior"]
        if failure != "worksheets":
            names.append("Joint probability")
        sheets = "".join(f'<sheet name="{name}" sheetId="{index}"/>' for index, name in enumerate(names, 1))
        content = BytesIO()
        with ZipFile(content, "w") as archive:
            archive.writestr("xl/workbook.xml",
                             '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                             f"<sheets>{sheets}</sheets></workbook>")
            archive.writestr("xl/worksheets/sheet1.xml", "<worksheet/>")
        return content.getvalue()

    monkeypatch.setattr(verifier, "_request_json", fake_json)
    monkeypatch.setattr(verifier, "_request_bytes", fake_bytes)
    if failure is not None:
        with pytest.raises(RuntimeError):
            verifier.verify_dense_export("http://staging/oxytib")
    else:
        report = verifier.verify_dense_export("http://staging/oxytib")
        assert report["status"] == "pass"
        assert report["field_shape"] == [833, 961]
        assert report["xlsx_bytes"] > 0
        assert calls[0][3] == calls[1][3] == 600.0
        assert calls[1][2]["inference"] == calls[0][2]
        assert calls[1][2]["context"] == {"isotope_source": "Direct air O2"}


def test_verifier_request_preserves_prefix_payload_and_timeout(monkeypatch) -> None:
    captured = []

    def open_request(request, *, timeout):
        captured.append((request.full_url, request.method, request.data, timeout))
        return BytesIO(b'{"status": "ok"}')

    monkeypatch.setattr(verifier, "urlopen", open_request)
    assert verifier._request_json("http://localhost/oxytib", "/api/v1/health") == {"status": "ok"}
    verifier._request_bytes("http://localhost/oxytib", "/api/v1/export/coordinate.xlsx", {"test": 1}, timeout=600)
    assert captured[0] == ("http://localhost/oxytib/api/v1/health", "GET", None, 30.0)
    assert captured[1][:2] == ("http://localhost/oxytib/api/v1/export/coordinate.xlsx", "POST")
    assert json.loads(captured[1][2]) == {"test": 1}
    assert captured[1][3] == 600


def test_remote_verifier_checks_identity_and_modern_known_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = {
        "/api/v1/health": {
            "status": "ok",
            "api_version": "1.0",
            "publication_model_id": verifier.EXPECTED_MODEL_ID,
        },
        "/api/v1/model": {
            "publication_model_id": verifier.EXPECTED_MODEL_ID,
            "operational_domain": {"pCO2_ppm": {"minimum": 50.0, "maximum": 60000.0}},
        },
        "/api/v1/forward": {
            "publication_model_id": verifier.EXPECTED_MODEL_ID,
            "result": {
                "central_cap_delta17_prime_permil": (
                    verifier.EXPECTED_MODERN_DELTA17_PERMIL
                )
            },
        },
    }

    def fake_request(
        _base_url: str, path: str, _payload: dict | None = None
    ) -> dict:
        return responses[path]

    monkeypatch.setattr(verifier, "_request_json", fake_request)
    report = verifier.verify_deployment("https://model.example.org/")
    assert report["status"] == "pass"
    assert report["publication_model_id"] == verifier.EXPECTED_MODEL_ID
    assert report["modern_Delta_prime_17O_permil"] == pytest.approx(
        verifier.EXPECTED_MODERN_DELTA17_PERMIL
    )

    responses["/api/v1/health"]["publication_model_id"] = "wrong-model"
    with pytest.raises(RuntimeError, match="unexpected deployed model identity"):
        verifier.verify_deployment("https://model.example.org/")
