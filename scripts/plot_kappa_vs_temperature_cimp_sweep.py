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
    build_spectral_grid,
    compute_thermal_conductivity,
    resolve_material_entries,
    sanitize_name,
)
from scripts.plot_kappa_vs_temperature import paper_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot thermal conductivity vs temperature for multiple C_imp values on one figure "
            "and overlay experimental reference points."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Input directory containing solver_params.toml.")
    parser.add_argument("--material", default="", help="Optional material name/key. Default: first material used by the case.")
    parser.add_argument("--T-min", type=float, default=18.0, help="Minimum temperature in K. Default: 18")
    parser.add_argument("--T-max", type=float, default=373.0, help="Maximum temperature in K. Default: 373")
    parser.add_argument("--num-points", type=int, default=356, help="Number of temperatures sampled on the curve. Default: 356")
    parser.add_argument(
        "--c-imp",
        nargs="+",
        type=float,
        default=[1e7, 1e8, 1e9, 1e10, 1e11],
        help="One or more C_imp values to compare. Default: 1e7 1e8 1e9 1e10 1e11",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Reference point in the form T:kappa. Default points: 300:1.76, 323:1.08, 373:0.79",
    )
    parser.add_argument("--cos-beta", type=float, default=1.0, help="Same definition as export_scattering_rate_vs_energy.py")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="Thread count per C_imp curve. Default: min(8, cpu_count)")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI. Default: 300")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for PNG/PDF/CSV outputs. Default: output/kappa_vs_temperature_cimp_sweep",
    )
    return parser.parse_args()


def resolve_output_dir(output_dir: str) -> Path:
    if output_dir:
        path = Path(output_dir).expanduser().resolve()
    else:
        path = ROOT / "output" / "kappa_vs_temperature_cimp_sweep"
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_targets(raw_targets: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    items = raw_targets if raw_targets else ["300:1.76", "323:1.08", "373:0.79"]
    for item in items:
        if ":" not in item:
            raise ValueError(f"invalid target format: {item!r}, expected T:kappa")
        left, right = item.split(":", 1)
        rows.append({"temperature_K": float(left), "thermal_conductivity_W_mK": float(right)})
    return pd.DataFrame(rows).sort_values("temperature_K", kind="stable").reset_index(drop=True)


def choose_material(input_dir: Path, requested_name: str) -> dict[str, object]:
    entries = resolve_material_entries(input_dir, [requested_name] if requested_name else [])
    if not entries:
        raise ValueError(f"no material resolved from {input_dir}")
    return entries[0]


def compute_total_kappa_at_temperature(material_entry: dict[str, object], input_dir: Path, temperature: float, cos_beta: float, c_imp: float) -> float:
    opts = build_opts(input_dir, float(temperature))
    opts["C_imp"] = float(c_imp)
    spec = build_spectral_grid(dict(material_entry["mat"]), opts)
    rates = branch_rates(spec, opts, float(temperature), float(cos_beta))
    table = compute_thermal_conductivity(material_entry, spec, rates, float(temperature))
    return float(table.loc[table["branch"] == "TOTAL", "thermal_conductivity_W_mK"].iloc[0])


def compute_curve(material_entry: dict[str, object], input_dir: Path, temperatures: np.ndarray, cos_beta: float, c_imp: float, jobs: int) -> pd.DataFrame:
    workers = max(1, int(jobs))
    if workers == 1:
        values = [compute_total_kappa_at_temperature(material_entry, input_dir, float(T), cos_beta, c_imp) for T in temperatures]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            values = list(
                executor.map(
                    lambda T: compute_total_kappa_at_temperature(material_entry, input_dir, float(T), cos_beta, c_imp),
                    temperatures,
                )
            )
    return pd.DataFrame(
        {
            "series": f"C_imp={float(c_imp):.0e}",
            "temperature_K": np.asarray(temperatures, dtype=np.float64),
            "thermal_conductivity_W_mK": np.asarray(values, dtype=np.float64),
            "C_imp": float(c_imp),
        }
    )


def plot_curves(material_name: str, curves_df: pd.DataFrame, target_df: pd.DataFrame, output_png: Path, output_pdf: Path, dpi: int) -> None:
    paper_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    cimp_values = sorted(curves_df["C_imp"].drop_duplicates().astype(float).tolist())
    colors = [cmap(i) for i in np.linspace(0.08, 0.92, len(cimp_values))]
    for color, c_imp in zip(colors, cimp_values, strict=True):
        sub = curves_df.loc[curves_df["C_imp"] == c_imp].sort_values("temperature_K", kind="stable")
        ax.plot(
            sub["temperature_K"].to_numpy(dtype=np.float64),
            sub["thermal_conductivity_W_mK"].to_numpy(dtype=np.float64),
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
            label=rf"$C_{{imp}}={c_imp:.0e}$",
        )
    tx = target_df["temperature_K"].to_numpy(dtype=np.float64)
    ty = target_df["thermal_conductivity_W_mK"].to_numpy(dtype=np.float64)
    ax.scatter(
        tx,
        ty,
        s=72,
        marker="D",
        color="#bf360c",
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
        label="Experimental points",
    )
    for idx, (xt, yt) in enumerate(zip(tx, ty, strict=True)):
        ax.annotate(
            f"{xt:.0f} K, {yt:.2f}",
            xy=(xt, yt),
            xytext=(8, 8 if idx % 2 == 0 else -14),
            textcoords="offset points",
            fontsize=9.5,
            color="#bf360c",
            arrowprops={"arrowstyle": "-", "color": "#bf360c", "lw": 0.8, "alpha": 0.85},
        )
    x_all = curves_df["temperature_K"].to_numpy(dtype=np.float64)
    y_all = np.r_[curves_df["thermal_conductivity_W_mK"].to_numpy(dtype=np.float64), ty]
    y_pos = y_all[np.isfinite(y_all) & (y_all > 0.0)]
    ax.set_xlim(float(np.nanmin(x_all)), float(np.nanmax(x_all)))
    ax.set_yscale("log")
    ax.set_ylim(float(np.min(y_pos)) / 1.25, float(np.max(y_pos)) * 1.35)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Thermal Conductivity, $\kappa$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title(f"{material_name} Thermal Conductivity vs Temperature | $C_{{imp}}$ Sweep")
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.28)
    ax.minorticks_on()
    ax.legend(loc="upper right", frameon=False)
    ax.text(
        0.02,
        0.04,
        "Lines: model curves for different $C_{imp}$\nMarkers: experimental reference points",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#f8fafc", "edgecolor": "#d1d5db", "alpha": 0.92},
    )
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    output_dir = resolve_output_dir(args.output_dir)
    material_entry = choose_material(input_dir, str(args.material))
    temperatures = np.linspace(float(args.T_min), float(args.T_max), int(args.num_points), dtype=np.float64)
    target_df = parse_targets(list(args.target))
    c_imp_values = [float(v) for v in args.c_imp]

    curves = [
        compute_curve(material_entry, input_dir, temperatures, float(args.cos_beta), c_imp, int(args.jobs))
        for c_imp in c_imp_values
    ]
    curves_df = pd.concat(curves, ignore_index=True)

    safe_name = sanitize_name(str(material_entry["name"]))
    cimp_tag = "_".join(f"{float(v):.0e}" for v in c_imp_values)
    stem = f"kappa_vs_temperature_Cimp_sweep_{safe_name}_{temperatures[0]:.0f}K_{temperatures[-1]:.0f}K_{cimp_tag}"
    plot_csv = output_dir / f"{stem}.csv"
    plot_png = output_dir / f"{stem}.png"
    plot_pdf = output_dir / f"{stem}.pdf"

    export_df = curves_df.copy()
    target_export = target_df.copy()
    target_export.insert(0, "series", "target")
    target_export["C_imp"] = np.nan
    export_df = pd.concat([export_df, target_export], ignore_index=True)
    export_df.to_csv(plot_csv, index=False)
    plot_curves(str(material_entry["name"]), curves_df, target_df, plot_png, plot_pdf, int(args.dpi))

    print(f"[ok] csv -> {plot_csv}")
    print(f"[ok] png -> {plot_png}")
    print(f"[ok] pdf -> {plot_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
