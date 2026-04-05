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
from scripts.fit_scattering_params_to_kappa import FIT_PARAM_BOUNDS
from scripts.plot_kappa_vs_temperature import paper_style


DEFAULT_TARGETS = ("300:1.76", "323:1.08", "373:0.79")
DEFAULT_SCAN_PARAMS = ("BL", "A_imp", "B_imp", "BTN", "BTU")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sensitivity analysis with fixed C_imp: scan each other scattering parameter "
            "across its search interval while keeping the remaining parameters fixed."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Baseline input directory.")
    parser.add_argument(
        "--baseline-params-csv",
        required=True,
        help="CSV containing the baseline parameter set, e.g. output/refit_cimp_1e10/fitted_parameters.csv",
    )
    parser.add_argument("--material", default="IGZO", help="Material name/key. Default: IGZO")
    parser.add_argument("--fixed-c-imp", type=float, default=1.0e10, help="Fixed C_imp value. Default: 1e10")
    parser.add_argument("--T-min", type=float, default=18.0, help="Minimum temperature in K. Default: 18")
    parser.add_argument("--T-max", type=float, default=373.0, help="Maximum temperature in K. Default: 373")
    parser.add_argument("--num-points", type=int, default=356, help="Temperature samples. Default: 356")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of scan values per parameter. Default: 5")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Reference point in the form T:kappa. Default: 300:1.76, 323:1.08, 373:0.79",
    )
    parser.add_argument(
        "--scan-param",
        action="append",
        default=[],
        choices=tuple(k for k in FIT_PARAM_BOUNDS if k != "C_imp"),
        help="Parameter to scan. Repeat to override defaults.",
    )
    parser.add_argument(
        "--fixed-param",
        action="append",
        default=[],
        help="Fix a parameter to a specific value, e.g. A_imp=1e-46. Repeatable.",
    )
    parser.add_argument(
        "--scan-bound",
        action="append",
        default=[],
        help="Override scan bounds for one parameter, e.g. BL=1e-23:1e-21. Repeatable.",
    )
    parser.add_argument("--cos-beta", type=float, default=1.0, help="Same definition as export_scattering_rate_vs_energy.py")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="Thread count per curve. Default: min(8, cpu_count)")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI. Default: 300")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for outputs. Default: output/fixed_cimp_param_sensitivity",
    )
    return parser.parse_args()


def resolve_output_dir(output_dir: str) -> Path:
    if output_dir:
        path = Path(output_dir).expanduser().resolve()
    else:
        path = ROOT / "output" / "fixed_cimp_param_sensitivity"
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_targets(raw_targets: list[str]) -> pd.DataFrame:
    items = raw_targets if raw_targets else list(DEFAULT_TARGETS)
    rows: list[dict[str, float]] = []
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


def load_baseline_params(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    params: dict[str, float] = {}
    for row in df.itertuples(index=False):
        params[str(row.parameter)] = float(row.value)
    return params


def parse_fixed_params(items: list[str]) -> dict[str, float]:
    fixed: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --fixed-param format: {item!r}, expected NAME=VALUE")
        name, value = item.split("=", 1)
        key = name.strip()
        if key not in FIT_PARAM_BOUNDS:
            raise ValueError(f"unknown fixed parameter: {key}")
        fixed[key] = float(value)
    return fixed


def parse_scan_bounds(items: list[str]) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --scan-bound format: {item!r}, expected NAME=LOW:HIGH")
        name, spec = item.split("=", 1)
        key = name.strip()
        if key not in FIT_PARAM_BOUNDS or key == "C_imp":
            raise ValueError(f"unknown or unsupported scan parameter: {key}")
        if ":" in spec:
            low_text, high_text = spec.split(":", 1)
        elif "," in spec:
            low_text, high_text = spec.split(",", 1)
        else:
            raise ValueError(f"invalid bound spec for {key}: {spec!r}, expected LOW:HIGH")
        low = float(low_text)
        high = float(high_text)
        if low <= 0.0 or high <= 0.0 or low >= high:
            raise ValueError(f"invalid scan bounds for {key}: {(low, high)}")
        bounds[key] = (low, high)
    return bounds


def compute_total_kappa_at_temperature(
    material_entry: dict[str, object],
    input_dir: Path,
    temperature: float,
    cos_beta: float,
    param_overrides: dict[str, float],
) -> float:
    opts = build_opts(input_dir, float(temperature))
    opts.update(param_overrides)
    spec = build_spectral_grid(dict(material_entry["mat"]), opts)
    rates = branch_rates(spec, opts, float(temperature), float(cos_beta))
    table = compute_thermal_conductivity(material_entry, spec, rates, float(temperature))
    return float(table.loc[table["branch"] == "TOTAL", "thermal_conductivity_W_mK"].iloc[0])


def compute_curve(
    material_entry: dict[str, object],
    input_dir: Path,
    temperatures: np.ndarray,
    cos_beta: float,
    param_overrides: dict[str, float],
    jobs: int,
) -> np.ndarray:
    workers = max(1, int(jobs))
    if workers == 1:
        values = [compute_total_kappa_at_temperature(material_entry, input_dir, float(T), cos_beta, param_overrides) for T in temperatures]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            values = list(
                executor.map(
                    lambda T: compute_total_kappa_at_temperature(material_entry, input_dir, float(T), cos_beta, param_overrides),
                    temperatures,
                )
            )
    return np.asarray(values, dtype=np.float64)


def plot_parameter_sweep(
    material_name: str,
    parameter_name: str,
    curves_df: pd.DataFrame,
    target_df: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    dpi: int,
) -> None:
    paper_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    series_values = sorted(curves_df["scan_value"].drop_duplicates().astype(float).tolist())
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0.08, 0.92, len(series_values))]
    for color, scan_value in zip(colors, series_values, strict=True):
        sub = curves_df.loc[curves_df["scan_value"] == scan_value].sort_values("temperature_K", kind="stable")
        ax.plot(
            sub["temperature_K"].to_numpy(dtype=np.float64),
            sub["thermal_conductivity_W_mK"].to_numpy(dtype=np.float64),
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
            label=rf"{parameter_name}={scan_value:.2e}",
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
    y_all = np.r_[curves_df["thermal_conductivity_W_mK"].to_numpy(dtype=np.float64), ty]
    y_pos = y_all[np.isfinite(y_all) & (y_all > 0.0)]
    ax.set_xlim(float(curves_df["temperature_K"].min()), float(curves_df["temperature_K"].max()))
    ax.set_yscale("log")
    ax.set_ylim(float(np.min(y_pos)) / 1.25, float(np.max(y_pos)) * 1.35)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Thermal Conductivity, $\kappa$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title(f"{material_name} | Sensitivity to {parameter_name} | fixed $C_{{imp}}$")
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.28)
    ax.minorticks_on()
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    baseline_params_csv = Path(args.baseline_params_csv).expanduser().resolve()
    output_dir = resolve_output_dir(args.output_dir)
    material_entry = choose_material(input_dir, str(args.material))
    baseline_params = load_baseline_params(baseline_params_csv)
    fixed_params = parse_fixed_params(list(args.fixed_param))
    bound_overrides = parse_scan_bounds(list(args.scan_bound))
    baseline_params["C_imp"] = float(args.fixed_c_imp)
    baseline_params.update(fixed_params)

    temperatures = np.linspace(float(args.T_min), float(args.T_max), int(args.num_points), dtype=np.float64)
    target_df = parse_targets(list(args.target))
    if args.scan_param:
        scan_params = tuple(args.scan_param)
    else:
        scan_params = tuple(name for name in DEFAULT_SCAN_PARAMS if name not in fixed_params)

    manifest_rows: list[dict[str, object]] = []
    for parameter_name in scan_params:
        bounds = bound_overrides.get(parameter_name, FIT_PARAM_BOUNDS[parameter_name])
        scan_values = np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), int(args.num_samples), dtype=np.float64)
        curve_rows: list[dict[str, object]] = []
        for scan_value in scan_values:
            overrides = dict(baseline_params)
            overrides[parameter_name] = float(scan_value)
            values = compute_curve(material_entry, input_dir, temperatures, float(args.cos_beta), overrides, int(args.jobs))
            curve_rows.extend(
                {
                    "parameter": parameter_name,
                    "scan_value": float(scan_value),
                    "temperature_K": float(temp),
                    "thermal_conductivity_W_mK": float(kappa),
                    "fixed_C_imp": float(args.fixed_c_imp),
                }
                for temp, kappa in zip(temperatures, values, strict=True)
            )
        curves_df = pd.DataFrame(curve_rows)
        safe_name = sanitize_name(str(material_entry["name"]))
        stem = f"sensitivity_{parameter_name}_{safe_name}_{temperatures[0]:.0f}K_{temperatures[-1]:.0f}K"
        csv_path = output_dir / f"{stem}.csv"
        png_path = output_dir / f"{stem}.png"
        pdf_path = output_dir / f"{stem}.pdf"
        curves_df.to_csv(csv_path, index=False)
        plot_parameter_sweep(str(material_entry["name"]), parameter_name, curves_df, target_df, png_path, pdf_path, int(args.dpi))
        manifest_rows.append(
            {
                "parameter": parameter_name,
                "scan_low": float(bounds[0]),
                "scan_high": float(bounds[1]),
                "num_samples": int(args.num_samples),
                "csv_path": str(csv_path.resolve()),
                "png_path": str(png_path.resolve()),
                "pdf_path": str(pdf_path.resolve()),
            }
        )
        print(f"[ok] {parameter_name} -> {png_path}")

    pd.DataFrame(manifest_rows).to_csv(output_dir / "manifest.csv", index=False)
    pd.DataFrame(
        [{"parameter": k, "value": baseline_params[k]} for k in sorted(baseline_params) if k in FIT_PARAM_BOUNDS]
    ).to_csv(output_dir / "baseline_parameters.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
