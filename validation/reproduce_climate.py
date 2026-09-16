"""Recompute climate-isotope comparisons from archived native atmosphere arrays."""

from pathlib import Path

from audit_clima_global_o2_response import run

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".project-root").exists())


def main():
    for name, po2, suffix in (
        ("1pal_additive", 1.0, ""),
        ("1pal_fixed_total", 1.0, "_fixed_total"),
        ("0p1pal_additive", 0.1, "_0p1pal"),
        ("2pal_additive", 2.0, "_2pal"),
    ):
        source = ROOT / "validation/reference_data/climate" / name
        print(f"Recomputing isotope response: {name}", flush=True)
        run(source / "manifest.json", source,
            ROOT / "model_data/updated_r7_response_surface_v1.json",
            ROOT / "outputs" / f"clima_global_o2_response{suffix}.json", po2_pal=po2)


if __name__ == "__main__":
    main()
