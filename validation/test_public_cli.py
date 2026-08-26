"""Public command-line entry-point tests."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = next(path for path in Path(__file__).resolve().parents if (path / ".project-root").exists())
RUNNER = ROOT / "run_model.py"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), "calculate", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_forward_json_stdout_uses_publication_model() -> None:
    completed = _run("forward", "--po2", "1", "--pco2", "294", "--gpp", "290")
    payload = json.loads(completed.stdout)
    assert payload["publication_model_id"] == "oxytib_publication_model_v1"
    assert payload["calculation"] == "steady_forward"
    assert payload["result"]["inputs"]["p_co2_ppm"] == 294.0


def test_inverse_roundtrip() -> None:
    forward_payload = json.loads(_run("forward").stdout)
    target = forward_payload["result"]["central_cap_delta17_prime_permil"]
    completed = _run(
        "infer",
        "--target-d17o",
        str(target),
        "--solve-for",
        "pCO2",
        "--po2",
        "1",
        "--gpp",
        "290",
    )
    result = json.loads(completed.stdout)["result"]
    assert result["status"] in {"solved", "admissible_interval_found"}
    assert abs(result["central_root"] - 294.0) < 1.0e-6


def test_transient_csv_export() -> None:
    output = ROOT / ".public_cli_test_step.csv"
    try:
        completed = _run(
            "transient",
            "--experiment",
            "pCO2",
            "--final-value",
            "400",
            "--duration",
            "100",
            "--samples",
            "5",
            "--format",
            "csv",
            "--output",
            str(output),
        )
        assert str(output) in completed.stdout
        lines = output.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 6
        assert "O2_Delta_prime_17O_permil" in lines[0]
    finally:
        output.unlink(missing_ok=True)


def test_gradual_pco2_trajectory_json() -> None:
    completed = _run(
        "transient",
        "--experiment",
        "pCO2-trajectory",
        "--initial-pco2",
        "285.5",
        "--final-value",
        "422.8",
        "--trajectory-duration",
        "174",
        "--duration",
        "200",
        "--equilibrium-horizon",
        "200",
        "--samples",
        "9",
    )
    payload = json.loads(completed.stdout)
    assert payload["calculation"] == "pco2_trajectory_transient"
    assert payload["result"]["pco2_ppm"][-1] == 422.8
