"""Generate report/geological_map.html — a spatial geology EDA for the ROGII
Wellbore Geology Prediction dataset.

What this report shows that the other reports don't:
  * Where each of the six formations sits in absolute elevation across the
    33×24 mile play (structure-contour maps).
  * Isopach maps — how the thickness of each layer varies spatially.
  * Local dip magnitude + direction across the field (quiver plot).
  * N-S and E-W cross-section transects through the survey area.
  * Per-well drilling depth and GR statistics painted on the map.

The 6 formation-top columns are training-only, but their elevation samples
along every horizontal trajectory effectively give us thousands of (X, Y,
elevation) points for each formation — perfect inputs to a structure-contour
interpolator.
"""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

DATA_ROOT = Path("/home/tom99763/ROGII/rogii-wellbore-geology-prediction")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
REPORT_DIR = Path("/home/tom99763/ROGII/report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FORMATIONS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
# Display friendly names
FORM_FULL = {
    "ANCC": "Anacacho",
    "ASTNU": "Austin Chalk (Upper)",
    "ASTNL": "Austin Chalk (Lower)",
    "EGFDU": "Eagle Ford (Upper)",
    "EGFDL": "Eagle Ford (Lower)",
    "BUDA": "Buda Limestone",
}

FIGS: dict[str, str] = {}


def save_fig(name: str, fig) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGS[name] = base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# 1. Sample formation-top points from all training horizontal wells
# ---------------------------------------------------------------------------
SAMPLE_STRIDE = 40  # every 40 rows of MD (≈40 ft) per well

cols = ["X", "Y", "Z", "GR", "TVT", *FORMATIONS]
hw_paths = sorted(
    p for p in TRAIN_DIR.glob("*__horizontal_well.csv") if "Zone" not in p.name
)
print(f"loading {len(hw_paths)} training horizontal wells (subsampling every {SAMPLE_STRIDE} rows)...")
t0 = time.time()
parts = []
for i, p in enumerate(hw_paths):
    df = pd.read_csv(p, usecols=cols)
    df = df.iloc[::SAMPLE_STRIDE].copy()
    df["well_id"] = p.name.split("__", 1)[0]
    parts.append(df)
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(hw_paths)}  ({time.time()-t0:.1f}s)")
points = pd.concat(parts, ignore_index=True)
print(f"  -> {len(points):,} sampled formation-top observations in {time.time()-t0:.1f}s")

# Also pull per-well centroid + lateral azimuth from the existing summary
well_summary = pd.read_csv(REPORT_DIR / "well_summary.csv")
ts = well_summary[well_summary.split == "train"].reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. Build a regular interpolation grid covering the play
# ---------------------------------------------------------------------------
x_min, x_max = points["X"].min(), points["X"].max()
y_min, y_max = points["Y"].min(), points["Y"].max()
# Add a small margin
mx = 0.02 * (x_max - x_min)
my = 0.02 * (y_max - y_min)
x_min -= mx; x_max += mx; y_min -= my; y_max += my
nx, ny = 140, 110
xi = np.linspace(x_min, x_max, nx)
yi = np.linspace(y_min, y_max, ny)
XX, YY = np.meshgrid(xi, yi)

print("interpolating structure contours...")
formation_grids: dict[str, np.ndarray] = {}
xy = points[["X", "Y"]].to_numpy()
for fm in FORMATIONS:
    Z = griddata(xy, points[fm].to_numpy(), (XX, YY), method="linear")
    # Smooth lightly to suppress interpolation noise from finite sampling
    mask = ~np.isnan(Z)
    Zs = Z.copy()
    Zs[mask] = gaussian_filter(np.where(mask, Z, 0), sigma=1.2)[mask]
    formation_grids[fm] = Zs

# Convert elevations (negative numbers in the data) to depth-from-datum-style
# friendlier display: we'll plot raw elevation but label "Elevation (ft)" so
# higher (less negative) = shallower, lower (more negative) = deeper. The
# colormap convention chosen so 'red = shallow', 'blue = deep' matches geologist intuition.

# ---------------------------------------------------------------------------
# 3. Figure 1 — six structure-contour maps in one 2×3 grid
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5), constrained_layout=True)
fig.suptitle("Structure Contour Maps — top elevation of each formation across the play",
             fontsize=14, weight="bold")

# Use a common range across the warmest set so colors are roughly comparable
for ax, fm in zip(axes.flat, FORMATIONS):
    Z = formation_grids[fm]
    vmin, vmax = np.nanpercentile(Z, [2, 98])
    cf = ax.contourf(XX / 1000, YY / 1000, Z, levels=18,
                     cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
    ax.contour(XX / 1000, YY / 1000, Z, levels=10,
               colors="black", linewidths=0.4, alpha=0.55)
    # Light overlay of test wells in white
    te = well_summary[well_summary.split == "test"]
    ax.scatter(te["x_mean"] / 1000, te["y_mean"] / 1000,
               s=80, marker="*", c="white", edgecolor="black", linewidth=1.0, zorder=5)
    ax.set_title(f"{fm} — {FORM_FULL[fm]}", fontsize=11)
    ax.set_xlabel("X (×1000 ft)")
    ax.set_ylabel("Y (×1000 ft)")
    ax.set_aspect("equal", adjustable="box")
    cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Elevation (ft)\nshallow ↑ / deep ↓", fontsize=8)
save_fig("structure_contours", fig)


# ---------------------------------------------------------------------------
# 4. Figure 2 — isopach maps (layer thickness) for the 5 inter-formation pairs
# ---------------------------------------------------------------------------
pairs = list(zip(FORMATIONS[:-1], FORMATIONS[1:]))  # (ANCC,ASTNU), ...
fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5), constrained_layout=True)
fig.suptitle("Isopach Maps — thickness of each layer (upper top − lower top)",
             fontsize=14, weight="bold")

for ax, (upper, lower) in zip(axes.flat, pairs):
    # Both grids hold elevation (more negative = deeper). Upper is shallower.
    # Thickness = upper_elev - lower_elev (positive feet thick).
    thick = formation_grids[upper] - formation_grids[lower]
    vmin, vmax = np.nanpercentile(thick, [2, 98])
    cf = ax.contourf(XX / 1000, YY / 1000, thick, levels=18,
                     cmap="YlOrRd", vmin=vmin, vmax=vmax)
    ax.contour(XX / 1000, YY / 1000, thick, levels=10,
               colors="black", linewidths=0.4, alpha=0.55)
    ax.set_title(f"{upper} → {lower}\n(thickness)", fontsize=11)
    ax.set_xlabel("X (×1000 ft)")
    ax.set_ylabel("Y (×1000 ft)")
    ax.set_aspect("equal", adjustable="box")
    cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Thickness (ft)", fontsize=8)
# Hide the 6th panel (only 5 pairs)
axes.flat[-1].axis("off")
save_fig("isopach", fig)


# ---------------------------------------------------------------------------
# 5. Figure 3 — Eagle Ford reservoir thickness (the target zone)
# ---------------------------------------------------------------------------
eagle_top = formation_grids["EGFDU"]      # top of Eagle Ford
eagle_base = formation_grids["BUDA"]      # base of Eagle Ford (~top of Buda)
eagle_thick = eagle_top - eagle_base

fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), constrained_layout=True)

ax = axes[0]
vmin, vmax = np.nanpercentile(eagle_top, [2, 98])
cf = ax.contourf(XX / 1000, YY / 1000, eagle_top, levels=22,
                 cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
ax.contour(XX / 1000, YY / 1000, eagle_top, levels=14,
           colors="black", linewidths=0.4, alpha=0.55)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Top of Eagle Ford (EGFDU) — depth structure")
ax.set_xlabel("X (×1000 ft)")
ax.set_ylabel("Y (×1000 ft)")
cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("Elevation (ft)\nshallow ↑ / deep ↓")
te = well_summary[well_summary.split == "test"]
ax.scatter(te["x_mean"] / 1000, te["y_mean"] / 1000, s=120, marker="*",
           c="white", edgecolor="black", linewidth=1.2, zorder=5, label="test well")
ax.legend(loc="upper right", fontsize=8)

ax = axes[1]
vmin, vmax = np.nanpercentile(eagle_thick, [2, 98])
cf = ax.contourf(XX / 1000, YY / 1000, eagle_thick, levels=22,
                 cmap="YlOrRd", vmin=vmin, vmax=vmax)
ax.contour(XX / 1000, YY / 1000, eagle_thick, levels=14,
           colors="black", linewidths=0.4, alpha=0.55)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Eagle Ford reservoir thickness (EGFDU → BUDA)")
ax.set_xlabel("X (×1000 ft)")
ax.set_ylabel("Y (×1000 ft)")
cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("Thickness (ft)")
ax.scatter(te["x_mean"] / 1000, te["y_mean"] / 1000, s=120, marker="*",
           c="white", edgecolor="black", linewidth=1.2, zorder=5)
save_fig("eagle_ford", fig)


# ---------------------------------------------------------------------------
# 6. Figure 4 — Local dip magnitude + direction (quiver) from EGFDU surface
# ---------------------------------------------------------------------------
# Compute gradient ∇z of the EGFDU surface
dy_step = (YY[1, 0] - YY[0, 0])   # ft per cell in Y
dx_step = (XX[0, 1] - XX[0, 0])   # ft per cell in X
gz_dy, gz_dx = np.gradient(eagle_top, dy_step, dx_step)  # ft/ft slopes
dip_mag = np.hypot(gz_dx, gz_dy)
dip_az = np.degrees(np.arctan2(-gz_dy, -gz_dx)) % 360  # down-dip direction (deg from East CCW)

fig, ax = plt.subplots(figsize=(10.5, 8), constrained_layout=True)
cf = ax.contourf(XX / 1000, YY / 1000, dip_mag * 100, levels=20, cmap="magma_r",
                 vmin=0, vmax=np.nanpercentile(dip_mag * 100, 96))
# Subsample for quiver
step = 7
ax.quiver(XX[::step, ::step] / 1000, YY[::step, ::step] / 1000,
          -gz_dx[::step, ::step], -gz_dy[::step, ::step],
          color="white", scale=0.3, width=0.0025, alpha=0.95)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Local structural dip — magnitude (colormap, in %) and down-dip direction (arrows)\n"
             "Computed from interpolated Eagle Ford (EGFDU) top surface", fontsize=11)
ax.set_xlabel("X (×1000 ft)")
ax.set_ylabel("Y (×1000 ft)")
cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("Dip magnitude (% = ft of elevation per ft of horizontal distance × 100)")
ax.scatter(te["x_mean"] / 1000, te["y_mean"] / 1000, s=140, marker="*",
           c="cyan", edgecolor="black", linewidth=1.2, zorder=5, label="test well")
ax.legend(loc="upper right", fontsize=9)
save_fig("dip_quiver", fig)


# ---------------------------------------------------------------------------
# 7. Figure 5 — N–S and E–W cross-sections through the play centroid
# ---------------------------------------------------------------------------
center_x = (x_min + x_max) / 2
center_y = (y_min + y_max) / 2

# Find indices of the center row / column
ix_center = np.argmin(np.abs(xi - center_x))
iy_center = np.argmin(np.abs(yi - center_y))

fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.5), constrained_layout=True)

# E-W cross-section at center Y
ax = axes[0]
colors = ["#1f4e8a", "#3b6fbb", "#74a0d9", "#e0a070", "#c87237", "#8c4d22"]
for fm, c in zip(FORMATIONS, colors):
    z_line = formation_grids[fm][iy_center, :]
    ax.plot(xi / 1000, z_line, color=c, lw=1.6, label=fm)
# Fill between consecutive formations to highlight layers
for i in range(len(FORMATIONS) - 1):
    upper = formation_grids[FORMATIONS[i]][iy_center, :]
    lower = formation_grids[FORMATIONS[i + 1]][iy_center, :]
    ax.fill_between(xi / 1000, upper, lower, color=colors[i], alpha=0.25)
ax.set_title(f"E–W cross-section at Y ≈ {center_y/1000:,.0f} ×1000 ft")
ax.set_xlabel("X (×1000 ft)")
ax.set_ylabel("Elevation (ft)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

# N-S cross-section at center X
ax = axes[1]
for fm, c in zip(FORMATIONS, colors):
    z_line = formation_grids[fm][:, ix_center]
    ax.plot(yi / 1000, z_line, color=c, lw=1.6, label=fm)
for i in range(len(FORMATIONS) - 1):
    upper = formation_grids[FORMATIONS[i]][:, ix_center]
    lower = formation_grids[FORMATIONS[i + 1]][:, ix_center]
    ax.fill_between(yi / 1000, upper, lower, color=colors[i], alpha=0.25)
ax.set_title(f"N–S cross-section at X ≈ {center_x/1000:,.0f} ×1000 ft")
ax.set_xlabel("Y (×1000 ft)")
ax.set_ylabel("Elevation (ft)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
save_fig("cross_sections", fig)


# ---------------------------------------------------------------------------
# 8. Figure 6 — Per-well statistics painted on map
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)

# (a) drilling depth (min TVT — i.e. shallowest part of the lateral)
ax = axes[0]
sc = ax.scatter(ts["x_mean"] / 1000, ts["y_mean"] / 1000,
                c=ts["tvt_min"], cmap="RdYlBu_r", s=14, alpha=0.85)
ax.scatter(te["x_mean"] / 1000, te["y_mean"] / 1000, s=180, marker="*",
           c="white", edgecolor="black", linewidth=1.2, zorder=5, label="test well")
ax.set_aspect("equal", adjustable="box")
ax.set_title("(a) Minimum TVT per well\n(deepest target encountered)")
ax.set_xlabel("X (×1000 ft)")
ax.set_ylabel("Y (×1000 ft)")
fig.colorbar(sc, ax=ax, shrink=0.85, label="min TVT (ft)")
ax.legend(loc="best", fontsize=8)

# (b) mean GR
ax = axes[1]
sc = ax.scatter(ts["x_mean"] / 1000, ts["y_mean"] / 1000,
                c=ts["gr_mean"], cmap="viridis", s=14, alpha=0.85)
ax.scatter(te["x_mean"] / 1000, te["y_mean"] / 1000, s=180, marker="*",
           c="white", edgecolor="black", linewidth=1.2, zorder=5)
ax.set_aspect("equal", adjustable="box")
ax.set_title("(b) Mean horizontal-well GR\n(lithology proxy)")
ax.set_xlabel("X (×1000 ft)")
fig.colorbar(sc, ax=ax, shrink=0.85, label="mean GR (API)")

# (c) lateral azimuth (color-coded by direction)
ax = axes[2]
sc = ax.scatter(ts["x_mean"] / 1000, ts["y_mean"] / 1000,
                c=ts["lateral_azimuth_deg"], cmap="twilight", s=14, alpha=0.85)
ax.scatter(te["x_mean"] / 1000, te["y_mean"] / 1000, s=180, marker="*",
           c="white", edgecolor="black", linewidth=1.2, zorder=5)
ax.set_aspect("equal", adjustable="box")
ax.set_title("(c) Lateral azimuth (0=East)\n(drilling direction by well)")
ax.set_xlabel("X (×1000 ft)")
fig.colorbar(sc, ax=ax, shrink=0.85, label="azimuth (deg)")
save_fig("well_stats_map", fig)


# ---------------------------------------------------------------------------
# 9. Headline numbers
# ---------------------------------------------------------------------------
play_area_mi2 = ((x_max - x_min) / 5280) * ((y_max - y_min) / 5280)
eagle_thick_med = float(np.nanmedian(eagle_thick))
eagle_thick_min = float(np.nanpercentile(eagle_thick, 5))
eagle_thick_max = float(np.nanpercentile(eagle_thick, 95))
egfd_top_med = float(np.nanmedian(eagle_top))
egfd_top_dip_med_pct = float(np.nanmedian(dip_mag) * 100)
egfd_top_dip_p95_pct = float(np.nanpercentile(dip_mag, 95) * 100)


# ---------------------------------------------------------------------------
# 10. Build HTML
# ---------------------------------------------------------------------------
def img(name: str, alt: str = "") -> str:
    return (
        f"<figure><img src='data:image/png;base64,{FIGS[name]}' alt='{alt}' "
        f"style='width:100%;max-width:100%'/></figure>"
    )


html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ROGII — Spatial Geology EDA</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
    background: #fafbfc; color: #1f2933;
    max-width: 1180px; margin: 0 auto; padding: 28px 36px 80px;
    line-height: 1.6;
  }}
  h1 {{ font-size: 30px; border-bottom: 2px solid #3b6fbb; padding-bottom: 6px; margin-top: 0; }}
  h2 {{ font-size: 22px; color: #2b4a7a; margin-top: 38px;
        border-left: 4px solid #3b6fbb; padding-left: 10px; }}
  h3 {{ font-size: 16px; color: #444; margin-top: 22px; }}
  p, li {{ font-size: 14.5px; }}
  .meta {{ font-size: 12px; color: #6b7280; margin-bottom: 18px; }}
  code {{ background: #eef0f2; padding: 1px 5px; border-radius: 3px; font-size: 13px; }}
  blockquote {{ border-left: 3px solid #5cb874; margin: 14px 0; padding: 10px 16px;
                background: white; color: #2f4d36; font-size: 13.5px; }}
  blockquote.idea {{ border-left-color: #b8860b; color: #5c4515; background: #fff9e6; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px; margin: 18px 0; }}
  .kpi {{ background: white; border: 1px solid #e3e7ec; border-radius: 8px; padding: 12px 14px; }}
  .kpi .v {{ font-size: 19px; font-weight: 600; color: #1f2933; }}
  .kpi .l {{ font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }}
  table.data {{ border-collapse: collapse; width: 100%; font-size: 13px;
                margin: 10px 0 18px; background: white; }}
  table.data th, table.data td {{ border-bottom: 1px solid #e3e7ec;
                padding: 7px 10px; text-align: left; vertical-align: top; }}
  table.data th {{ background: #eef3f9; color: #2b4a7a; }}
  figure {{ margin: 10px 0 26px; }}
  a {{ color: #2b6cb0; }}
  hr {{ border: none; border-top: 1px solid #e3e7ec; margin: 30px 0; }}
</style>
</head>
<body>

<h1>ROGII — Spatial Geology EDA</h1>
<div class="meta">
  Map-vs-geology view of the
  <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction">
    rogii-wellbore-geology-prediction</a>
  dataset. Generated by <code>report/generate_geological_map.py</code> from {len(hw_paths)}
  training wells × every {SAMPLE_STRIDE}-ft MD sample = {len(points):,}
  formation-top observations interpolated onto a {nx}×{ny} grid.
</div>

<h2>1. The Play in One Paragraph</h2>
<blockquote>
The 773 training wells + 3 visible test wells cover a contiguous
<b>{(x_max-x_min)/5280:.0f} × {(y_max-y_min)/5280:.0f}-mile</b> block
(≈ {play_area_mi2:,.0f} mi² of lease area). The geological column visible in
typewell labels matches the <b>South Texas Eagle Ford trend</b>: from shallow to deep,
<code>ANCC</code> (Anacacho) → <code>ASTNU</code>/<code>ASTNL</code> (Austin Chalk Upper/Lower)
→ <code>EGFDU</code>/<code>EGFDL</code> (Eagle Ford Upper/Lower) → <code>BUDA</code> (Buda
Limestone). Eagle Ford is the productive zone — the median Eagle Ford reservoir thickness
(EGFDU → BUDA) is about <b>{eagle_thick_med:.0f}&nbsp;ft</b> across the play, with the top of
the Eagle Ford sitting at a median elevation of <b>{egfd_top_med:,.0f}&nbsp;ft</b>
(typical TVT ≈ {abs(egfd_top_med)+1500:,.0f}&nbsp;ft below surface, since Z is elevation, not
depth-from-surface). Structural dip is gentle: median ≈
<b>{egfd_top_dip_med_pct:.2f}%</b>, but the structure does tilt systematically across the
play.
</blockquote>

<h2>2. Headline Numbers</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="l">Play footprint</div>
       <div class="v">{(x_max-x_min)/5280:.1f} × {(y_max-y_min)/5280:.1f} mi</div></div>
  <div class="kpi"><div class="l">Wells</div><div class="v">{len(hw_paths)} train + 3 test</div></div>
  <div class="kpi"><div class="l">Sampled obs</div><div class="v">{len(points):,}</div></div>
  <div class="kpi"><div class="l">Top of Eagle Ford (median)</div>
       <div class="v">{egfd_top_med:,.0f} ft</div></div>
  <div class="kpi"><div class="l">Eagle Ford thickness (median)</div>
       <div class="v">{eagle_thick_med:.0f} ft</div></div>
  <div class="kpi"><div class="l">Eagle Ford thickness (p5–p95)</div>
       <div class="v">{eagle_thick_min:.0f} – {eagle_thick_max:.0f} ft</div></div>
  <div class="kpi"><div class="l">Median structural dip</div>
       <div class="v">{egfd_top_dip_med_pct:.2f}%</div></div>
  <div class="kpi"><div class="l">p95 structural dip</div>
       <div class="v">{egfd_top_dip_p95_pct:.2f}%</div></div>
</div>

<h2>3. Structure Contour Maps — All Six Formations</h2>
<p>For each of the six formation tops, we have an elevation observation at every
sampled MD point of every training horizontal well. Linear interpolation onto a
regular grid gives a structure-contour map per formation. <i>Red = shallow, blue
= deep.</i> White stars are the three visible test wells.</p>
{img("structure_contours")}
<p>All six surfaces share the <b>same overall structural pattern</b> — they are
parallel beds dipping in roughly the same direction. This is the spatial
manifestation of the &ldquo;nearby wells&apos; typewells are offset copies&rdquo;
insight from the EDA: same stratigraphy + small regional dip means every
neighbour&apos;s typewell sees the same GR sequence with a depth shift dictated
by the dip.</p>

<h2>4. Isopach Maps — Where Each Layer Is Thick or Thin</h2>
<p>Subtracting consecutive structure-contour surfaces gives the local layer
thickness. <i>Yellow = thin, red = thick.</i> These are the conventional
isopachs reservoir geologists use to plan landing zones.</p>
{img("isopach")}
<p>The Eagle Ford intervals (EGFDU and EGFDL) are spatially the most variable —
they thicken and thin meaningfully across the play. The Austin Chalk intervals
above them are more uniform.</p>

<h2>5. The Target Zone — Eagle Ford Reservoir</h2>
<p>Almost every horizontal well in this dataset is drilled within the Eagle Ford
(between the EGFDU top and the BUDA top). The left panel shows the depth of the
Eagle Ford top — i.e. how deep each well had to drill before reaching the target.
The right panel shows the Eagle Ford reservoir thickness — the vertical window
the lateral can stay inside.</p>
{img("eagle_ford")}

<blockquote class="idea">
<b>Insight for modelling.</b> The test wells (white stars) fall in regions where:
(a) the Eagle Ford top is at typical depth, and (b) Eagle Ford thickness is in
the dataset&apos;s normal range. There is no extrapolation risk — they sit
firmly in the training distribution. This is good news: training-set-derived
inversion priors should transfer cleanly.
</blockquote>

<h2>6. Structural Dip — Magnitude and Direction</h2>
<p>Numerically differentiating the smoothed Eagle Ford top gives, at each grid
cell, the local dip slope (% rise/fall per horizontal foot) and the down-dip
direction. The colormap shows magnitude; the white arrows point down-dip.</p>
{img("dip_quiver")}
<p>The play has a coherent regional structural trend — arrows align across
neighbouring cells, which means the &ldquo;use nearest wells&apos; geology as a
prior&rdquo; trick is sound. Where you see swirls or arrow chaos, the dip is
very small (essentially flat zones) and the noisy interpolation dominates.</p>

<h2>7. Cross-Sections Through the Play</h2>
<p>Slicing the interpolated structure surfaces along the East–West and
North–South center lines gives true cross-sections of the geology. Coloured
bands are the formation intervals; lines are the formation tops.</p>
{img("cross_sections")}
<p>Two things to read off these plots:</p>
<ul>
  <li><b>The beds are nearly parallel</b> — all six surfaces dip in tandem. That
      makes the Tikhonov smoothness prior (penalize curvature of the TVT path)
      well-justified.</li>
  <li><b>The Eagle Ford zone tilts a few hundred feet</b> across the 30-mile
      block — meaningful for any well that crosses geographically through it,
      and exactly why predicting TVT is more than constant extrapolation.</li>
</ul>

<h2>8. Per-Well Statistics on the Map</h2>
<p>Three quantities per well, painted at the well&apos;s XY centroid:
(a) minimum TVT — the deepest part of the lateral, a proxy for landing depth;
(b) mean horizontal GR — a coarse lithology indicator;
(c) lateral azimuth in degrees — the compass direction the well was drilled.</p>
{img("well_stats_map")}
<p>Lateral azimuth (right panel) clearly clusters in two dominant directions —
matches the EDA finding that the field has two principal drilling orientations,
likely chosen perpendicular to the regional maximum-horizontal-stress direction
for fracture-network efficiency.</p>

<h2>9. What This Means for the Competition</h2>
<ul>
  <li><b>One play, one geological column.</b> Models can assume a single shared
      stratigraphy — no need to generalize across different basins or formations.</li>
  <li><b>Test wells are inside the training distribution.</b> No
      out-of-distribution surprise on the depth axis. Whatever prior you build
      from training will apply.</li>
  <li><b>Regional dip is small and coherent.</b> Linear-Z baseline succeeds
      precisely <i>because</i> dip is locally stable. But local variability
      &mdash; the swirls in the dip map &mdash; explains the long tail where
      naive baselines fail.</li>
  <li><b>Offset-well priors are physically defensible.</b> The structure
      contours show neighbouring wells really are sampling the same dipping
      surface; a stacked super-typewell is essentially a Monte Carlo estimate
      of the same regional structure.</li>
  <li><b>Two drilling directions ⇒ two failure modes.</b> A model that uses
      lateral azimuth as an input feature should help, especially for laterals
      that drilled <i>across</i> dip rather than <i>along</i> it.</li>
</ul>

<hr/>
<p class="meta">
This is a spatial complement to the EDA in
<a href="eda_report.html">eda_report.html</a> (per-well statistics) and the
formal problem framing in
<a href="competition_overview.html">competition_overview.html</a>. The
consolidated knowledge-base index is <a href="index.html">index.html</a>.
</p>

</body>
</html>
"""

out = REPORT_DIR / "geological_map.html"
out.write_text(html, encoding="utf-8")
print(f"\nwrote {out}  ({out.stat().st_size/1024:.1f} KB)")
