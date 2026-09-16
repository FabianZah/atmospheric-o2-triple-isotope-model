"""Verify a running public deployment against the publication contract."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from math import isclose
from time import monotonic
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from xml.etree import ElementTree
from zipfile import ZipFile


EXPECTED_MODEL_ID = "oxytib_publication_model_v1"
EXPECTED_MODERN_DELTA17_PERMIL = -0.42635313046373885


def _request_bytes(
    base_url: str, path: str, payload: dict[str, Any] | None = None,
    *, timeout: float = 30.0,
) -> bytes:
    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"deployment request failed for {url}: {exc}") from exc


def _request_json(
    base_url: str, path: str, payload: dict[str, Any] | None = None,
    *, timeout: float = 30.0,
) -> dict:
    return json.loads(_request_bytes(base_url, path, payload, timeout=timeout))


def verify_dense_export(base_url: str) -> dict[str, Any]:
    """Opt-in staging load check using a genuinely refined posterior field."""
    payload = {
        "solve_for": "pCO2",
        "target_air_cap_delta17_permil": -8.0,
        "measurement_sigma_permil": 0.015,
        "gpp_constraint": {"kind": "normal", "center": 58.0, "sigma": 29.0},
        "po2_constraint": {"kind": "normal", "center": 0.2, "sigma": 0.2},
    }
    start = monotonic()
    envelope = _request_json(
        base_url, "/api/v1/inference/coordinate", payload, timeout=600.0,
    )
    solve_seconds = monotonic() - start
    if envelope.get("publication_model_id") != EXPECTED_MODEL_ID:
        raise RuntimeError("unexpected model identity in dense inference")
    shape = envelope["result"].get("field_shape")
    if not shape or len(shape) != 2 or shape[0] * shape[1] < 250_000:
        raise RuntimeError("dense-export check did not exercise a refined field")
    start = monotonic()
    content = _request_bytes(
        base_url, "/api/v1/export/coordinate.xlsx",
        {"inference": payload, "context": {"isotope_source": "Direct air O2"}},
        timeout=600.0,
    )
    export_seconds = monotonic() - start
    with ZipFile(BytesIO(content)) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("dense XLSX archive failed its integrity check")
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        names = {node.attrib["name"] for node in workbook.findall("s:sheets/s:sheet", namespace)}
        if not {"Summary", "Posterior", "Joint probability"} <= names:
            raise RuntimeError("dense XLSX is missing a required worksheet")
        sheet_bytes = {
            info.filename: info.file_size for info in archive.infolist()
            if info.filename.startswith("xl/worksheets/sheet")
        }
    health = _request_json(base_url, "/api/v1/health")
    if health.get("status") != "ok" or health.get("publication_model_id") != EXPECTED_MODEL_ID:
        raise RuntimeError("deployment is not healthy after dense export")
    return {
        "status": "pass", "field_shape": shape,
        "posterior_median_ppm": envelope["result"]["posterior_median"],
        "solve_seconds": solve_seconds, "export_seconds": export_seconds,
        "xlsx_bytes": len(content), "worksheet_xml_bytes": sheet_bytes,
    }


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
    parser.add_argument(
        "--check-dense-export", action="store_true",
        help="Run a resource-intensive refined-field and XLSX check against staging.",
    )
    arguments = parser.parse_args()
    report = verify_deployment(arguments.base_url)
    if arguments.check_dense_export:
        report["dense_export"] = verify_dense_export(arguments.base_url)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
