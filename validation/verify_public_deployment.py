"""Verify a running public deployment against the publication contract."""

from __future__ import annotations

import argparse
import json
from math import isclose
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


EXPECTED_MODEL_ID = "atmospheric_o2_triple_isotope_model_v1"
EXPECTED_MODERN_DELTA17_PERMIL = -0.42635313046373885


def _request_json(base_url: str, path: str, payload: dict[str, Any] | None = None) -> dict:
    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=30.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"deployment request failed for {url}: {exc}") from exc


def verify_deployment(base_url: str) -> dict[str, Any]:
    health = _request_json(base_url, "/api/v1/health")
    metadata = _request_json(base_url, "/api/v1/model")
    forward = _request_json(
        base_url,
        "/api/v1/forward",
        {"p_o2_pal": 1.0, "p_co2_ppm": 294.0, "gpp_pgC_per_year": 290.0},
    )
    model_ids = {
        health.get("publication_model_id"),
        metadata.get("publication_model_id"),
        forward.get("publication_model_id"),
    }
    if model_ids != {EXPECTED_MODEL_ID}:
        raise RuntimeError(f"unexpected deployed model identity: {sorted(model_ids)}")
    modern = forward["result"]["central_cap_delta17_prime_permil"]
    if not isclose(modern, EXPECTED_MODERN_DELTA17_PERMIL, abs_tol=1.0e-12):
        raise RuntimeError(
            "deployed modern Delta-prime-17O differs from the release contract: "
            f"{modern:.15g}"
        )
    return {
        "status": "pass",
        "base_url": base_url.rstrip("/"),
        "publication_model_id": EXPECTED_MODEL_ID,
        "api_version": health["api_version"],
        "modern_Delta_prime_17O_permil": modern,
        "operational_domain": metadata["operational_domain"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify_deployment(arguments.base_url), indent=2))


if __name__ == "__main__":
    main()
