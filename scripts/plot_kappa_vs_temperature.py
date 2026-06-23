from __future__ import annotations

import argparse
import os
import re
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot thermal conductivity vs temperature using the current scattering parameters "
            "and overlay selected literature target points."
        )
    )
    parser.add_argument("--input-dir", default="input", help="Input directory containing solver_params.toml. Default: input")
    parser.add_argument("--material", default="", help="Optional material name/key. Default: first material used by the case.")
    parser.add_argument("--T-min", type=float, default=290.0, help="Minimum temperature in K. Default: 290")
    parser.add_argument("--T-max", type=float, default=400.0, help="Maximum temperature in K. Default: 400")
    parser.add_argument("--num-points", type=int, default=111, help="Number of temperatures sampled on the curve. Default: 111")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Reference point in the form T:kappa, for example 300:1.76 . Repeat for multiple points.",
    )
    parser.add_argument("--cos-beta", type=float, default=1.0, help="Same definition as export_scattering_rate_vs_energy.py")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="Thread count for temperature sweep. Default: min(8, cpu_count)")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI. Default: 300")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for PNG/PDF/CSV outputs. Default: output/kappa_vs_temperature",
    )
    return parser.parse_args()


def resolve_output_dir(output_dir: str) -> Path:
    if output_dir:
        path = Path(output_dir).expanduser().resolve()
    else:
        path = ROOT / "output" / "kappa_vs_temperature"
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_targets(raw_targets: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for item in raw_targets:
        if ":" not in item:
            raise ValueError(f"invalid target format: {item!r}, expected T:kappa")
        left, right = item.split(":", 1)
        rows.append(
            {
                "temperature_K": float(left),
                "thermal_conductivity_W_mK": float(right),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["temperature_K", "thermal_conductivity_W_mK"])
    target_df = pd.DataFrame(rows)
    target_df = target_df.sort_values("temperature_K", kind="stable").reset_index(drop=True)
    return target_df


def choose_material(input_dir: Path, requested_name: str) -> dict[str, object]:
    entries = resolve_material_entries(input_dir, [requested_name] if requested_name else [])
    if not entries:
        raise ValueError(f"no material resolved from {input_dir}")
    return entries[0]


def compute_total_kappa_at_temperature(material_entry: dict[str, object], input_dir: Path, temperature: float, cos_beta: float) -> float:
    opts = build_opts(input_dir, float(temperature))
    spec = build_spectral_grid(dict(material_entry["mat"]), opts)
    mk = str(material_entry.get("key", ""))
    rates = branch_rates(spec, opts, float(temperature), float(cos_beta), material_key_name=mk)
    table = compute_thermal_conductivity(material_entry, spec, rates, float(temperature))
    return float(table.loc[table["branch"] == "TOTAL", "thermal_conductivity_W_mK"].iloc[0])


def compute_curve(material_entry: dict[str, object], input_dir: Path, temperatures: np.ndarray, cos_beta: float, jobs: int) -> pd.DataFrame:
    workers = max(1, int(jobs))
    if workers == 1:
        values = [compute_total_kappa_at_temperature(material_entry, input_dir, float(T), cos_beta) for T in temperatures]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            values = list(executor.map(lambda T: compute_total_kappa_at_temperature(material_entry, input_dir, float(T), cos_beta), temperatures))
    return pd.DataFrame(
        {
            "series": "model",
            "temperature_K": np.asarray(temperatures, dtype=np.float64),
            "thermal_conductivity_W_mK": np.asarray(values, dtype=np.float64),
        }
    )


def paper_style() -> None:
    plt.rcParams.update(
        {
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
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "xtick.minor.size": 3.0,
            "ytick.minor.size": 3.0,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_curve(material_name: str, curve_df: pd.DataFrame, target_df: pd.DataFrame, output_png: Path, output_pdf: Path, dpi: int) -> None:
    paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    model_color = "#0f4c81"
    band_color = "#9ec5e6"
    target_color = "#bf360c"

    x = curve_df["temperature_K"].to_numpy(dtype=np.float64)
    y = curve_df["thermal_conductivity_W_mK"].to_numpy(dtype=np.float64)
    y_valid = y[np.isfinite(y) & (y > 0.0)]
    if y_valid.size == 0:
        raise ValueError("thermal conductivity curve must contain positive values for log-scale plotting")
    y_floor = float(np.min(y_valid)) / 1.25
    ax.plot(x, y, color=model_color, linewidth=2.8, solid_capstyle="round", label="Current parameter set")
    ax.fill_between(x, y, np.full_like(y, y_floor), color=band_color, alpha=0.12)

    if not target_df.empty:
        tx = target_df["temperature_K"].to_numpy(dtype=np.float64)
        ty = target_df["thermal_conductivity_W_mK"].to_numpy(dtype=np.float64)
        ax.scatter(
            tx,
            ty,
            s=72,
            marker="D",
            color=target_color,
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
            label="Literature points",
        )
        for idx, (xt, yt) in enumerate(zip(tx, ty, strict=True)):
            ax.annotate(
                f"{xt:.0f} K, {yt:.2f}",
                xy=(xt, yt),
                xytext=(8, 8 if idx % 2 == 0 else -14),
                textcoords="offset points",
                fontsize=9.5,
                color=target_color,
                arrowprops={"arrowstyle": "-", "color": target_color, "lw": 0.8, "alpha": 0.85},
            )

    ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    y_all = np.r_[y, target_df["thermal_conductivity_W_mK"].to_numpy(dtype=np.float64)] if not target_df.empty else y
    y_pos = y_all[np.isfinite(y_all) & (y_all > 0.0)]
    if y_pos.size == 0:
        raise ValueError("thermal conductivity plot requires positive y values for log scale")
    y_min = float(np.min(y_pos))
    y_max = float(np.max(y_pos))
    ax.set_yscale("log")
    ax.set_ylim(y_min / 1.25, y_max * 1.25)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Thermal Conductivity, $\kappa$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title(f"{material_name} Thermal Conductivity vs Temperature")
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.28)
    ax.minorticks_on()
    ax.legend(loc="upper right", frameon=False)
    ax.text(
        0.02,
        0.04,
        "Line: current scattering parameters\nMarkers: literature targets",
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
    curve_df = compute_curve(material_entry, input_dir, temperatures, float(args.cos_beta), int(args.jobs))

    safe_name = sanitize_name(str(material_entry["name"]))
    stem = f"kappa_vs_temperature_{safe_name}_{temperatures[0]:.0f}K_{temperatures[-1]:.0f}K"
    plot_csv = output_dir / f"{stem}.csv"
    plot_png = output_dir / f"{stem}.png"
    plot_pdf = output_dir / f"{stem}.pdf"

    export_df = curve_df.copy()
    if not target_df.empty:
        target_export = target_df.copy()
        target_export.insert(0, "series", "target")
        export_df = pd.concat([export_df, target_export], ignore_index=True)
    export_df.to_csv(plot_csv, index=False)
    plot_curve(str(material_entry["name"]), curve_df, target_df, plot_png, plot_pdf, int(args.dpi))

    print(f"[ok] csv -> {plot_csv}")
    print(f"[ok] png -> {plot_png}")
    print(f"[ok] pdf -> {plot_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
