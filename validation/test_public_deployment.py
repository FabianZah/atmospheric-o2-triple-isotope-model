"""Production deployment contract and remote-verifier tests."""

from __future__ import annotations

from pathlib import Path

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
    assert api["cap_drop"] == ["ALL"]
    assert api["security_opt"] == ["no-new-privileges:true"]
    assert api["pids_limit"] == "${O2_MODEL_PIDS:-256}"
    assert api["mem_limit"] == "${O2_MODEL_MEMORY:-2g}"
    assert api["cpus"] == "${O2_MODEL_CPUS:-2.0}"
    assert api["networks"] == ["backend"]
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


def test_environment_template_and_ignore_policy_are_safe() -> None:
    template = (ROOT / "deploy" / ".env.production.example").read_text(
        encoding="utf-8"
    )
    assert "O2_MODEL_DOMAIN=model.example.org" in template
    assert "O2_MODEL_CORS_ORIGINS=" in template
    assert "CADDY_IMAGE=caddy:2.11.4-alpine" in template
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
    assert api["ports"] == ["127.0.0.1:${ATMO_MOD_LOCAL_PORT:-18000}:8000"]
    assert api["read_only"] is True
    assert api["cap_drop"] == ["ALL"]
    assert api["security_opt"] == ["no-new-privileges:true"]
    assert api["pids_limit"] == "${ATMO_MOD_PIDS:-128}"
    assert api["mem_limit"] == "${ATMO_MOD_MEMORY:-768m}"
    assert api["cpus"] == "${ATMO_MOD_CPUS:-0.75}"
    assert api["networks"] == ["traefik-global-proxy"]
    assert compose["networks"]["traefik-global-proxy"] == {
        "external": True,
        "name": "traefik-global-proxy",
    }

    labels = dict(label.split("=", 1) for label in api["labels"])
    assert labels["traefik.enable"] == "${ATMO_MOD_TRAEFIK_ENABLE:-false}"
    assert "Path(`/atmo-mod`)" in labels["traefik.http.routers.atmo-mod.rule"]
    assert "PathPrefix(`/atmo-mod/`)" in labels["traefik.http.routers.atmo-mod.rule"]
    assert labels["traefik.http.routers.atmo-mod.middlewares"] == (
        "atmo-mod-slash,atmo-mod-strip"
    )
    assert labels["traefik.http.middlewares.atmo-mod-strip.stripprefix.prefixes"] == (
        "/atmo-mod"
    )
    assert labels["traefik.http.services.atmo-mod.loadbalancer.server.port"] == "8000"

    template = (ROOT / "deploy" / ".env.traefik.example").read_text(
        encoding="utf-8"
    )
    assert "ATMO_MOD_TRAEFIK_ENABLE=false" in template
    assert "ATMO_MOD_HOST=model.example.org" in template
    assert "ATMO_MOD_MEMORY=768m" in template
    assert "PASSWORD=" not in template
    assert "TOKEN=" not in template


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
