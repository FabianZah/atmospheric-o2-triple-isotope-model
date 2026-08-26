"""Fast end-to-end smoke test for the public scientific package."""

from __future__ import annotations

import importlib.metadata
import json
from math import isfinite
from pathlib import Path
import py_compile
import sys
from typing import Any


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
for subdirectory in ("code", "validation"):
    module_path = str(ROOT / subdirectory)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

from audit_publication_model_acceptance import build_report  # noqa: E402
from updated_output_surface import (  # noqa: E402
    UpdatedOutputSurfaceInput,
    load_updated_output_surface,
)
from updated_output_surface_inverse import (  # noqa: E402
    UpdatedSurfaceInverseInput,
    invert_updated_output_surface,
)


CONTRACT_PATH = ROOT / "model_data" / "publication_model_contract_v1.json"
REQUIRED_FILES = (
    CONTRACT_PATH,
    ROOT / "model_data" / "updated_r7_response_surface_v1.json",
    ROOT / "model_data" / "updated_molecular_output_surface_v1.json",
    ROOT / "code" / "updated_molecular_forward_model.py",
    ROOT / "code" / "updated_output_surface.py",
    ROOT / "code" / "updated_output_surface_inverse.py",
    ROOT / "code" / "public_model_service.py",
    ROOT / "code" / "model_result_workbook.py",
    ROOT / "code" / "web_api.py",
    ROOT / "web" / "index.html",
    ROOT / "web" / "styles.css",
    ROOT / "web" / "app.js",
    ROOT / "CITATION.cff",
    ROOT / "CITATION.bib",
    ROOT / "CITATION.ris",
)


def build_smoke_report() -> dict[str, Any]:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required publication files are missing: {missing}")

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    surface = load_updated_output_surface()
    expected_surface_id = contract["deterministic_model"]["numerical_accelerator"][
        "surface_data_id"
    ]
    if surface.surface_data_id != expected_surface_id:
        raise RuntimeError("runtime output surface differs from the publication contract")

    modern_request = UpdatedOutputSurfaceInput(
        p_o2_pal=1.0,
        p_co2_ppm=294.0,
        gpp_pgC_per_year=290.0,
    )
    modern = surface.evaluate(modern_request)
    nonmodern = surface.evaluate(
        UpdatedOutputSurfaceInput(
            p_o2_pal=0.5,
            p_co2_ppm=10_000.0,
            gpp_pgC_per_year=72.5,
        )
    )
    values = (
        modern.central_cap_delta17_prime_permil,
        modern.central_delta18_prime_permil,
        nonmodern.central_cap_delta17_prime_permil,
        nonmodern.central_delta18_prime_permil,
    )
    if not all(isfinite(value) for value in values):
        raise RuntimeError("publication surface returned a non-finite isotope value")
    if modern.extrapolation_applied or nonmodern.extrapolation_applied:
        raise RuntimeError("publication smoke points unexpectedly used extrapolation")

    inverse = invert_updated_output_surface(
        UpdatedSurfaceInverseInput(
            target_air_cap_delta17_permil=modern.central_cap_delta17_prime_permil,
            solve_for="pCO2",
            p_o2_pal=modern_request.p_o2_pal,
            p_co2_ppm=modern_request.p_co2_ppm,
            gpp_pgC_per_year=modern_request.gpp_pgC_per_year,
        ),
        verify_live_root=False,
    )
    if inverse.central_root is None or abs(inverse.central_root - 294.0) > 1.0e-5:
        raise RuntimeError(
            "surface forward-to-inverse round trip did not recover modern pCO2"
        )

    py_compile.compile(str(ROOT / "code" / "public_model_service.py"), doraise=True)
    py_compile.compile(str(ROOT / "code" / "web_api.py"), doraise=True)
    index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    if (
        'href="assets/styles.css?v=' not in index
        or 'src="assets/app.js?v=' not in index
        or 'href="/assets/' in index
        or 'src="/assets/' in index
    ):
        raise RuntimeError(
            "browser interface does not load prefix-safe versioned assets"
        )
    for citation_link in (
        'href="citation/model.bib"',
        'href="citation/model.ris"',
        'href="citation/CITATION.cff"',
    ):
        if citation_link not in index:
            raise RuntimeError(f"browser interface is missing {citation_link}")
    acceptance = build_report()
    if acceptance["release_blocker_count"] != 0:
        raise RuntimeError("integrated acceptance audit reports release blockers")

    return {
        "status": "pass",
        "runtime_dependencies": {
            "PyYAML": importlib.metadata.version("PyYAML"),
        },
        "publication_model_id": contract["publication_model_id"],
        "surface_data_id": surface.surface_data_id,
        "public_interface": "FastAPI with static browser frontend",
        "domain": surface.domain,
        "modern": {
            "Delta_prime_17O_permil": modern.central_cap_delta17_prime_permil,
            "delta_prime_18O_permil": modern.central_delta18_prime_permil,
        },
        "nonmodern_point": {
            "pO2_PAL": 0.5,
            "pCO2_ppm": 10_000.0,
            "GPP_PgC_per_year": 72.5,
            "Delta_prime_17O_permil": nonmodern.central_cap_delta17_prime_permil,
        },
        "inverse_roundtrip_pCO2_ppm": inverse.central_root,
        "acceptance_verdict": acceptance["verdict"],
    }


def main() -> None:
    print(json.dumps(build_smoke_report(), indent=2))


if __name__ == "__main__":
    main()
