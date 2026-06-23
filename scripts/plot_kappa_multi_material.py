#!/usr/bin/env python3
"""Compute and plot temperature-dependent thermal conductivity for multiple materials.

Uses per-material scattering parameters from ``solver_params.toml`` (if
available) or the flat ``[scattering]`` section as fallback.

Usage::

    python scripts/plot_kappa_multi_material.py --input-dir input_two_material_dmm_test
    python scripts/plot_kappa_multi_material.py --input-dir input_two_material_dmm_test --T-min 200 --T-max 500 --num-points 61
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_scattering_rate_vs_energy import (
    branch_rates,
    build_opts,
    compute_thermal_conductivity,
    resolve_material_entries,
    sanitize_name,
)
from phonon_mc import build_spectral_grid

# Distinct colors for up to ~10 materials.
MATERIAL_COLORS = [
    "#0f4c81",  # dark blue
    "#bf360c",  # dark red
    "#2e7d32",  # green
    "#6a1b9a",  # purple
    "#e65100",  # orange
    "#00695c",  # teal
    "#c62828",  # crimson
    "#283593",  # indigo
    "#4e342e",  # brown
    "#37474f",  # blue-grey
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot κ(T) for all materials in an input directory using per-material scattering parameters."
    )
    parser.add_argument(
        "--input-dir", default="input_two_material_dmm_test",
        help="Input directory containing solver_params.toml and dispersion files. "
             "Default: input_two_material_dmm_test",
    )
    parser.add_argument(
        "--materials", nargs="*", default=[],
        help="Optional material filter, e.g. IGZO SILICON. Default: auto-discover all.",
    )
    parser.add_argument("--T-min", type=float, default=200.0, help="Minimum temperature in K. Default: 200")
    parser.add_argument("--T-max", type=float, default=500.0, help="Maximum temperature in K. Default: 500")
    parser.add_argument("--num-points", type=int, default=61,
                        help="Number of temperature points. Default: 61")
    parser.add_argument("--cos-beta", type=float, default=1.0,
                        help="Cosine term for thin-film boundary scattering. Default: 1.0")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1),
                        help="Thread count for temperature sweep. Default: min(8, cpu_count)")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI. Default: 200")
    parser.add_argument("--output-dir", default="",
                        help="Output directory. Default: output/kappa_multi_material")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation, only write CSV.")
    return parser.parse_args()


def resolve_output_dir(output_dir: str) -> Path:
    if output_dir:
        path = Path(output_dir).expanduser().resolve()
    else:
        path = ROOT / "output" / "kappa_multi_material"
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_kappa_curve(
    material_entry: dict[str, object],
    input_dir: Path,
    temperatures: np.ndarray,
    cos_beta: float,
    jobs: int,
) -> pd.DataFrame:
    """Compute κ(T) for a single material across all temperatures."""
    mk = str(material_entry.get("key", ""))
    name = str(material_entry.get("name", mk))
    # Build opts once (the T0 in opts is just for spectral grid; we override T in branch_rates).
    opts = build_opts(input_dir, float(np.median(temperatures)))
    spec = build_spectral_grid(dict(material_entry["mat"]), opts)

    def _kappa_one(T: float) -> float:
        rates = branch_rates(spec, opts, T, cos_beta, material_key_name=mk)
        table = compute_thermal_conductivity(material_entry, spec, rates, T)
        return float(table.loc[table["branch"] == "TOTAL", "thermal_conductivity_W_mK"].iloc[0])

    workers = max(1, int(jobs))
    if workers == 1:
        values = [_kappa_one(float(T)) for T in temperatures]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            values = list(executor.map(lambda T: _kappa_one(float(T)), temperatures))

    return pd.DataFrame({
        "material": name,
        "material_key": mk,
        "temperature_K": np.asarray(temperatures, dtype=np.float64),
        "thermal_conductivity_W_mK": np.asarray(values, dtype=np.float64),
    })


def plot_all_curves(
    curves_df: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    dpi: int,
) -> None:
    """Plot κ(T) for all materials on a single log-scale figure."""
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "axes.linewidth": 1.2,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 10.5,
    })

    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)

    materials = sorted(curves_df["material"].unique())
    for i, mat_name in enumerate(materials):
        sub = curves_df[curves_df["material"] == mat_name]
        x = sub["temperature_K"].to_numpy(dtype=np.float64)
        y = sub["thermal_conductivity_W_mK"].to_numpy(dtype=np.float64)
        color = MATERIAL_COLORS[i % len(MATERIAL_COLORS)]
        ax.plot(x, y, color=color, linewidth=2.4, solid_capstyle="round", label=mat_name)
        # Add subtle fill below curve.
        y_pos = y[y > 0] if np.any(y > 0) else np.array([1e-6])
        y_floor = float(np.min(y_pos)) / 1.5
        ax.fill_between(x, y, np.full_like(y, y_floor), color=color, alpha=0.08)

    # Axes.
    all_y = curves_df["thermal_conductivity_W_mK"].to_numpy(dtype=np.float64)
    y_pos_all = all_y[np.isfinite(all_y) & (all_y > 0.0)]
    if y_pos_all.size > 0:
        ax.set_yscale("log")
        ax.set_ylim(float(np.min(y_pos_all)) / 1.4, float(np.max(y_pos_all)) * 1.4)

    ax.set_xlim(
        float(curves_df["temperature_K"].min()),
        float(curves_df["temperature_K"].max()),
    )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Thermal Conductivity, $\kappa$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title("Thermal Conductivity vs Temperature — Per-Material Scattering Parameters")
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.28)
    ax.minorticks_on()
    ax.legend(loc="upper right", frameon=False)

    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        print(f"FAIL: input directory not found: {input_dir}")
        return 1

    # Discover materials.
    entries = resolve_material_entries(input_dir, list(args.materials))
    if not entries:
        print(f"FAIL: no materials discovered in {input_dir}")
        return 1

    print(f"Materials discovered ({len(entries)}):")
    for e in entries:
        mk = e.get("key", "?")
        mn = e.get("name", mk)
        mat_scat = {}
        # Try to get per-material scattering for display.
        try:
            opts_tmp = build_opts(input_dir, 300.0)
            ms = opts_tmp.get("material_scattering", {})
            mat_scat = ms.get(mk, {})
        except Exception:
            pass
        print(f"  {mk:20s}  name={mn}  branches={e['mat'].get('B','?')}  "
              f"scat_keys={list(mat_scat.keys()) if mat_scat else '(global)'}")

    temperatures = np.linspace(args.T_min, args.T_max, args.num_points, dtype=np.float64)
    output_dir = resolve_output_dir(args.output_dir)

    # Compute curves.
    all_curves: list[pd.DataFrame] = []
    for entry in entries:
        mk = entry.get("key", str(entry.get("name", "?")))
        print(f"Computing κ(T) for {mk} ...")
        curve = compute_kappa_curve(entry, input_dir, temperatures, args.cos_beta, args.jobs)
        all_curves.append(curve)

    curves_df = pd.concat(all_curves, ignore_index=True)

    # Write CSV.
    csv_path = output_dir / "kappa_multi_material.csv"
    curves_df.to_csv(csv_path, index=False)
    print(f"[ok] CSV -> {csv_path}")

    # Print summary table.
    print()
    print(f"{'Material':20s} {'T=200K':>10s} {'T=300K':>10s} {'T=400K':>10s} {'T=500K':>10s}")
    print("-" * 60)
    for entry in entries:
        mk = entry.get("key", "?")
        name = str(entry.get("name", mk))
        sub = curves_df[curves_df["material"] == name]
        if sub.empty:
            continue
        vals = []
        for T_ref in [200, 300, 400, 500]:
            idx = np.argmin(np.abs(sub["temperature_K"].to_numpy() - T_ref))
            vals.append(f"{sub.iloc[idx]['thermal_conductivity_W_mK']:10.4f}")
        print(f"{name:20s} {' '.join(vals)}")

    # Plot.
    if not args.no_plot:
        tag = "_".join(sanitize_name(str(e.get("name", e.get("key", "?")))) for e in entries)
        png_path = output_dir / f"kappa_multi_{tag}.png"
        pdf_path = output_dir / f"kappa_multi_{tag}.pdf"
        plot_all_curves(curves_df, png_path, pdf_path, args.dpi)
        print(f"[ok] PNG -> {png_path}")
        print(f"[ok] PDF -> {pdf_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
