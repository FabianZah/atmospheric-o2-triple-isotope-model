"""Regenerate interpolation holdouts against the OXYTIB physical solver."""

from pathlib import Path

from audit_updated_output_surface import run as isotope_holdouts
from audit_updated_delta18_surface import run as oxygen18_holdouts
from audit_updated_output_surface_lowco2 import run as low_co2_holdouts
from audit_updated_output_surface_shape import run as surface_shape

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".project-root").exists())


def require_accepted(report, name):
    if report.get("status") != "accepted":
        raise RuntimeError(f"{name} failed its scientific acceptance gates; see its output report")


def main():
    surface = ROOT / "model_data/updated_molecular_output_surface_v1.json"
    outputs = ROOT / "outputs"
    print("Checking isotope interpolation holdouts", flush=True)
    report = isotope_holdouts(candidate_path=surface,
                     report_path=outputs / "updated_molecular_output_surface_audit.json",
                     validated_surface_path=None, maximum_holdouts=None)
    require_accepted(report, "Isotope interpolation")
    print("Checking oxygen-18 holdouts in every surface cell", flush=True)
    report = oxygen18_holdouts(surface_path=surface,
                      report_path=outputs / "updated_molecular_output_surface_delta18_every_cell_audit.json")
    require_accepted(report, "Oxygen-18 interpolation")
    print("Checking low-CO2 interpolation", flush=True)
    report = low_co2_holdouts(surface, outputs / "updated_molecular_output_surface_50ppm_audit.json",
                    outputs / "updated_molecular_output_surface_50ppm_validated.json",
                    response_bundle_path=ROOT / "model_data/updated_r7_response_surface_v1.json")
    require_accepted(report, "Low-CO2 interpolation")
    print("Checking surface shape", flush=True)
    report = surface_shape(surface_path=surface,
                  report_path=outputs / "updated_molecular_output_surface_50ppm_shape_audit.json")
    require_accepted(report, "Surface shape")


if __name__ == "__main__":
    main()
