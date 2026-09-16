from __future__ import annotations

# --- path bootstrap (direct execution) ---
import sys as _sys
from pathlib import Path as _Path
_root = next((p for p in _Path(__file__).resolve().parents if (p / ".project-root").exists()), None)
if _root is not None:
    for _sub in ("code", "validation"):
        _p = str(_root / _sub)
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
# --- end path bootstrap ---
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from spherule_to_air_d17o import FISCHER_TABLE5_D18O, air_d17o_from_spherule


HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = next(
    (p for p in (HERE, *HERE.parents) if (p / ".project-root").exists()),
    HERE,
)
_PROJECT_OUTPUTS = _PROJECT_ROOT / "outputs"
# Optional raw Young Fig. 8 page scan, only needed to re-run the pixel
# digitization. It is intentionally NOT bundled in the public archive; the
# published pipeline reads the cached digitized curves below instead.
FIG8_IMAGE = _PROJECT_ROOT / "young_page_images" / "page_16_img_1.png"
# Cached digitized Fig. 8 curves shipped with the archive (derived data).
FIG8_DIGITIZED_CSV = _PROJECT_OUTPUTS / "young_fig8_digitized_curves.csv"

# Pixel calibration for the extracted Young et al. Fig. 8 image.
# The plot spans [CO2] = 0..30000 ppmv and D17O = 0..-14 per mil.
AX_LEFT = 414
AX_RIGHT = 2943
AX_TOP = 43
AX_BOTTOM = 2576
X_MIN = 0.0
X_MAX = 30000.0
Y_TOP = 0.0
Y_BOTTOM = -14.0

# Crop just inside the axes so axis spines and tick labels do not dominate.
CROP_LEFT = 430
CROP_RIGHT = 2925
CROP_TOP = 55
CROP_BOTTOM = 2555


def px_to_pco2(x_px: float) -> float:
    return X_MIN + (x_px - AX_LEFT) / (AX_RIGHT - AX_LEFT) * (X_MAX - X_MIN)


def px_to_d17o(y_px: float) -> float:
    return Y_TOP + (y_px - AX_TOP) / (AX_BOTTOM - AX_TOP) * (Y_BOTTOM - Y_TOP)


def largest_connected_component(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return the largest 8-connected component as local (y, x) pixels."""
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    best: list[tuple[int, int]] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue

            stack = [(y, x)]
            seen[y, x] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                pts.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            if len(pts) > len(best):
                best = pts
    return best


def split_curve_pixels(component: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Split the connected Fig. 8 curve pixels into upper and lower curve traces."""
    by_x: dict[int, list[int]] = defaultdict(list)
    for y_local, x_local in component:
        x = x_local + CROP_LEFT
        y = y_local + CROP_TOP
        by_x[x].append(y)

    upper: list[tuple[float, float]] = []
    lower: list[tuple[float, float]] = []

    for x, ys_raw in sorted(by_x.items()):
        ys = sorted(ys_raw)
        clusters: list[list[int]] = []
        cur = [ys[0]]
        for y in ys[1:]:
            if y - cur[-1] > 8:
                clusters.append(cur)
                cur = [y]
            else:
                cur.append(y)
        clusters.append(cur)

        # Single-pixel clusters are usually remnants of ticks or scan noise.
        medians = [float(np.median(c)) for c in clusters if len(c) >= 2]
        if not medians:
            continue

        upper.append((px_to_pco2(x), px_to_d17o(min(medians))))
        if len(medians) > 1:
            lower.append((px_to_pco2(x), px_to_d17o(max(medians))))

    return np.array(upper), np.array(lower)


def thin_curve(curve: np.ndarray, every_n_pixels: int = 10) -> np.ndarray:
    """Average the dense digitized trace into cleaner bins."""
    bins = np.floor(curve[:, 0] / (30000 / ((AX_RIGHT - AX_LEFT) / every_n_pixels))).astype(int)
    rows = []
    for b in sorted(set(bins)):
        sub = curve[bins == b]
        rows.append([float(np.mean(sub[:, 0])), float(np.mean(sub[:, 1]))])
    return np.array(rows)


def load_cached_fig8_curves() -> dict[int, np.ndarray] | None:
    """Load the digitized Fig. 8 curves from the cached CSV, if present.

    The cache is derived data shipped in ``outputs/`` so the validation
    pipeline is reproducible without redistributing the raw Young page scan.
    Returns ``None`` if the cache file is missing.
    """
    if not FIG8_DIGITIZED_CSV.exists():
        return None
    by_gpp: dict[int, list[tuple[float, float]]] = defaultdict(list)
    with FIG8_DIGITIZED_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            gpp = int(round(float(row["GPP_percent_modern"])))
            by_gpp[gpp].append((float(row["pCO2_ppmv"]), float(row["D17O_permil"])))
    curves: dict[int, np.ndarray] = {}
    for gpp, pts in by_gpp.items():
        arr = np.array(sorted(pts))
        curves[gpp] = arr
    return curves


def digitize_fig8(*, force_image: bool = False) -> dict[int, np.ndarray]:
    """Return the digitized Young Fig. 8 curves.

    By default this loads the cached CSV bundled in ``outputs/``. Set
    ``force_image=True`` (and provide the raw page scan at ``FIG8_IMAGE``) to
    re-run the pixel digitization and refresh the cache.
    """
    if not force_image:
        cached = load_cached_fig8_curves()
        if cached is not None:
            return cached
    if not FIG8_IMAGE.exists():
        raise FileNotFoundError(
            "Fig. 8 digitization needs either the cached CSV "
            f"({FIG8_DIGITIZED_CSV}) or the raw page scan ({FIG8_IMAGE}). "
            "The raw scan is not bundled in the public archive; run "
            "young_fig8_reconstruction.py with the scan available to "
            "regenerate the cache."
        )
    image = Image.open(FIG8_IMAGE).convert("L")
    arr = np.array(image)
    crop = arr[CROP_TOP:CROP_BOTTOM, CROP_LEFT:CROP_RIGHT]
    mask = crop < 80
    component = largest_connected_component(mask)
    upper, lower = split_curve_pixels(component)

    curves = {
        100: thin_curve(upper),
        50: thin_curve(lower),
    }

    # Add textual/nominal anchors so the cropped curve does not lose the
    # near-modern starting point at the upper-left corner. The 50% value is
    # consistent with Young's Fig. 7 294-ppmv isopleth; it is not meant to
    # imply that the separate Fig. 9 pO2/NPP transient is included in Fig. 8.
    anchors = {
        100: (294.0, -0.410),
        50: (294.0, -0.539),
    }
    for gpp, anchor in anchors.items():
        curves[gpp] = curves[gpp][curves[gpp][:, 0] > 400.0]
        curves[gpp] = np.vstack([np.array(anchor), curves[gpp]])
        curves[gpp] = curves[gpp][np.argsort(curves[gpp][:, 0])]

    return curves


def d17o_from_pco2(pco2_ppmv: float | np.ndarray, gpp_percent: int, curves: dict[int, np.ndarray]) -> float | np.ndarray:
    curve = curves[gpp_percent]
    return np.interp(pco2_ppmv, curve[:, 0], curve[:, 1])


def fit_tail(curve: np.ndarray) -> np.ndarray:
    """Estimate an endpoint-anchored saturating tail from the local terminal slope."""
    last_x = curve[-1, 0]
    last_y = curve[-1, 1]
    tail = curve[curve[:, 0] >= last_x - 5000.0]
    slope, _intercept = np.polyfit(tail[:, 0], tail[:, 1], 1)
    # The extracted terminal pixels are a little steeper than the visual trend
    # just before the plot edge. Damp the endpoint slope so the extrapolated
    # tail rotates counter-clockwise around the fixed final digitized point.
    slope *= 0.68

    # A visually conservative asymptote: keep the 50% curve flattening strongly
    # and make the 100% curve less steep than a broad nonlinear fit.
    if last_y < -11:
        y_inf = -14.0
    else:
        y_inf = -12.5

    # y = y_inf + (last_y - y_inf) exp(-k (x-last_x))
    k = -slope / (last_y - y_inf)
    return np.array([last_x, last_y, y_inf, k])


def extrapolated_curve(curve: np.ndarray, x_max: float = 60000.0) -> tuple[np.ndarray, np.ndarray]:
    """Return extension using a high-CO2 saturating tail fit."""
    params = fit_tail(curve)
    last_x = curve[-1, 0]
    extra_x = np.linspace(last_x, x_max, 260)
    last_x, last_y, y_inf, k = params
    extra_y = y_inf + (last_y - y_inf) * np.exp(-k * (extra_x - last_x))
    return params, np.column_stack([extra_x, extra_y])


def pco2_from_d17o(d17o_permil: float, gpp_percent: int, curves: dict[int, np.ndarray]) -> float:
    curve = curves[gpp_percent]
    # D17O decreases monotonically with pCO2 in Young Fig. 8. Sort by D17O for inverse interpolation.
    order = np.argsort(curve[:, 1])
    d_sorted = curve[order, 1]
    if d17o_permil < d_sorted[0] or d17o_permil > d_sorted[-1]:
        return float("nan")
    return float(np.interp(d17o_permil, d_sorted, curve[order, 0]))


def pco2_from_d17o_with_tail(d17o_permil: float, gpp_percent: int, curves: dict[int, np.ndarray]) -> float:
    direct = pco2_from_d17o(d17o_permil, gpp_percent, curves)
    if not np.isnan(direct):
        return direct
    curve = curves[gpp_percent]
    # Only extrapolate beyond the high-CO2, more-negative-D17O end.
    if d17o_permil > np.max(curve[:, 1]):
        return float("nan")
    params = fit_tail(curve)
    last_x, last_y, y_inf, k = params
    if d17o_permil <= y_inf:
        return float("nan")
    return float(last_x - np.log((d17o_permil - y_inf) / (last_y - y_inf)) / k)


def write_csv(curves: dict[int, np.ndarray]) -> None:
    _PROJECT_OUTPUTS.mkdir(parents=True, exist_ok=True)
    with FIG8_DIGITIZED_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["GPP_percent_modern", "pCO2_ppmv", "D17O_permil"])
        for gpp, curve in curves.items():
            for pco2, d17o in curve:
                writer.writerow([gpp, pco2, d17o])

    with (_PROJECT_OUTPUTS / "young_fig8_extrapolated_curves.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["GPP_percent_modern", "pCO2_ppmv", "D17O_permil", "source"])
        for gpp, curve in curves.items():
            for pco2, d17o in curve:
                writer.writerow([gpp, pco2, d17o, "digitized"])
            _params, extra = extrapolated_curve(curve)
            for pco2, d17o in extra:
                writer.writerow([gpp, pco2, d17o, "saturating_exponential_tail"])


def make_svg(curves: dict[int, np.ndarray]) -> None:
    width, height = 900, 560
    left, right, top, bottom = 84, 26, 48, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_plot_max = 60000.0
    y_plot_min = -18.0

    def xm(x: float) -> float:
        return left + x / x_plot_max * plot_w

    def ym(y: float) -> float:
        return top + (0 - y) / abs(y_plot_min) * plot_h

    def polyline(curve: np.ndarray, color: str) -> str:
        pts = " ".join(f"{xm(x):.2f},{ym(y):.2f}" for x, y in curve)
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>'

    def dashed_polyline(curve: np.ndarray, color: str) -> str:
        pts = " ".join(f"{xm(x):.2f},{ym(y):.2f}" for x, y in curve)
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5" stroke-dasharray="8 8"/>'

    def assumed_polyline(curve: np.ndarray, color: str) -> str:
        pts = " ".join(f"{xm(x):.2f},{ym(y):.2f}" for x, y in curve)
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="2 6"/>'

    def ticks_x(values: list[int]) -> str:
        out = []
        for v in values:
            x = xm(v)
            out.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#222"/>')
            out.append(f'<text x="{x:.2f}" y="{top + plot_h + 26}" text-anchor="middle">{v}</text>')
        return "\n".join(out)

    def ticks_y(values: list[int]) -> str:
        out = []
        for v in values:
            y = ym(v)
            out.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#222"/>')
            out.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e7e1d8"/>')
            out.append(f'<text x="{left - 10}" y="{y + 5:.2f}" text-anchor="end">{v}</text>')
        return "\n".join(out)

    p100_m10 = pco2_from_d17o_with_tail(-10.0, 100, curves)
    p50_m10 = pco2_from_d17o_with_tail(-10.0, 50, curves)
    air_range = [
        air_d17o_from_spherule(-10.0, min(FISCHER_TABLE5_D18O)),
        air_d17o_from_spherule(-10.0, max(FISCHER_TABLE5_D18O)),
    ]
    air_low, air_high = min(air_range), max(air_range)
    p100_low = pco2_from_d17o_with_tail(air_low, 100, curves)
    p100_high = pco2_from_d17o_with_tail(air_high, 100, curves)
    p50_low = pco2_from_d17o_with_tail(air_low, 50, curves)
    p50_high = pco2_from_d17o_with_tail(air_high, 50, curves)
    extra100 = extrapolated_curve(curves[100])[1]
    extra50 = extrapolated_curve(curves[50])[1]
    d_grid = np.linspace(-0.6, -14.0, 360)
    low_gpp_curves = {}
    for gpp in [25, 10, 5]:
        pts = []
        for d17o in d_grid:
            p100 = pco2_from_d17o_with_tail(float(d17o), 100, curves)
            if not np.isnan(p100):
                pts.append((p100 * gpp / 100.0, float(d17o)))
        low_gpp_curves[gpp] = np.array(pts)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: Arial, Helvetica, sans-serif; fill: #222; font-size: 14px; }}
  .title {{ font-size: 20px; font-weight: 700; }}
  .subtitle {{ font-size: 13px; fill: #555; }}
  .small {{ font-size: 13px; }}
  .axis-label {{ font-size: 15px; font-weight: 700; }}
</style>
<rect width="100%" height="100%" fill="#fbfaf7"/>
<text x="{left}" y="25" class="title">Digitized reconstruction of Young et al. Fig. 8</text>
<text x="{left}" y="44" class="subtitle">Solid lines are digitized from the paper; dashed tails are independent saturating fits to each high-CO2 curve.</text>
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="white" stroke="#d9d1c7"/>
{ticks_y([0,-2,-4,-6,-8,-10,-12,-14,-16,-18])}
{ticks_x([0,10000,20000,30000,40000,50000,60000])}
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>
<rect x="{left}" y="{ym(air_high):.2f}" width="{plot_w}" height="{ym(air_low) - ym(air_high):.2f}" fill="#7a6a43" opacity="0.13"/>
{polyline(curves[100], "#234f7a")}
{polyline(curves[50], "#b3263a")}
{dashed_polyline(extra100, "#234f7a")}
{dashed_polyline(extra50, "#b3263a")}
{assumed_polyline(low_gpp_curves[25], "#7b3f98")}
{assumed_polyline(low_gpp_curves[10], "#7b3f98")}
{assumed_polyline(low_gpp_curves[5], "#7b3f98")}
<line x1="{left}" y1="{ym(air_low):.2f}" x2="{left + plot_w}" y2="{ym(air_low):.2f}" stroke="#7a6a43" stroke-width="1" stroke-dasharray="3 5"/>
<line x1="{left}" y1="{ym(air_high):.2f}" x2="{left + plot_w}" y2="{ym(air_high):.2f}" stroke="#7a6a43" stroke-width="1" stroke-dasharray="3 5"/>
<line x1="{xm(p100_high):.2f}" y1="{ym(air_high):.2f}" x2="{xm(p100_low):.2f}" y2="{ym(air_low):.2f}" stroke="#234f7a" stroke-width="5" stroke-linecap="round"/>
<line x1="{xm(p50_high):.2f}" y1="{ym(air_high):.2f}" x2="{xm(p50_low):.2f}" y2="{ym(air_low):.2f}" stroke="#b3263a" stroke-width="5" stroke-linecap="round"/>
<text x="{xm(15100):.2f}" y="{ym(-5.3):.2f}" class="axis-label" fill="#234f7a">100% GPP</text>
<text x="{xm(16500):.2f}" y="{ym(-9.1):.2f}" class="axis-label" fill="#b3263a">50% GPP</text>
<text x="{xm(5200):.2f}" y="{ym(-8.4):.2f}" class="small" fill="#7b3f98">25%</text>
<text x="{xm(2600):.2f}" y="{ym(-9.7):.2f}" class="small" fill="#7b3f98">10%</text>
<text x="{xm(1450):.2f}" y="{ym(-10.7):.2f}" class="small" fill="#7b3f98">5%</text>
<text x="{xm(42000):.2f}" y="{ym(air_high) - 8:.2f}" class="small" fill="#6f5d2c">Spherule Δ′<tspan baseline-shift="super" font-size="10">17</tspan>O = -10‰</text>
<text x="{xm(2300):.2f}" y="{ym(-13.2):.2f}" class="small" fill="#7b3f98">assumed pCO<tspan baseline-shift="sub" font-size="10">2</tspan> ∝ GPP</text>
<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle" class="axis-label">pCO<tspan baseline-shift="sub" font-size="11">2</tspan> (ppmv)</text>
<text transform="translate(24,{top + plot_h / 2}) rotate(-90)" text-anchor="middle" class="axis-label">Δ′<tspan baseline-shift="super" font-size="11">17</tspan>O O<tspan baseline-shift="sub" font-size="11">2</tspan> (‰)</text>
</svg>
"""
    (HERE / "young_fig8_digitized_reconstruction.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    curves = digitize_fig8()
    write_csv(curves)
    make_svg(curves)

    print("Wrote young_fig8_digitized_curves.csv")
    print("Wrote young_fig8_extrapolated_curves.csv")
    print("Wrote young_fig8_digitized_reconstruction.svg")
    for d17o in [-0.410, -0.539, -7.0, -10.0]:
        p100 = pco2_from_d17o_with_tail(d17o, 100, curves)
        p50 = pco2_from_d17o_with_tail(d17o, 50, curves)
        p100_txt = "outside range" if np.isnan(p100) else f"{p100:7.0f} ppmv" + ("*" if np.isnan(pco2_from_d17o(d17o, 100, curves)) else "")
        p50_txt = "outside range" if np.isnan(p50) else f"{p50:7.0f} ppmv" + ("*" if np.isnan(pco2_from_d17o(d17o, 50, curves)) else "")
        print(f"D17O={d17o:6.3f} per mil -> pCO2={p100_txt} at 100% GPP; {p50_txt} at 50% GPP")
    print("* extrapolated beyond digitized Fig. 8 range")


if __name__ == "__main__":
    main()
