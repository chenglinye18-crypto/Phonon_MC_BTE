from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_heat_flux import aggregate_flux_tables, load_step_flux_table, resolve_steps
from scripts.plot_kappa_vs_temperature import paper_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For the 280 nm gradient-dependence dataset, take the middle monitor plane, average "
            "heat flux over a step window, convert to thermal conductivity, and plot one curve "
            "per base temperature."
        )
    )
    parser.add_argument("--step-start", type=int, default=32000, help="Averaging window start step. Default: 32000")
    parser.add_argument("--step-end", type=int, default=40000, help="Averaging window end step. Default: 40000")
    parser.add_argument("--length-nm", type=float, default=280.0, help="Transport length used in deltaT/L. Default: 280 nm")
    parser.add_argument("--middle-label", default="flux_plane_003", help="Middle monitor-plane label. Default: flux_plane_003")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI. Default: 300")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for PNG/PDF/CSV outputs. Default: output/y280_center_plane_kappa_vs_gradient",
    )
    return parser.parse_args()


def resolve_output_dir(output_dir: str) -> Path:
    if output_dir:
        path = Path(output_dir).expanduser().resolve()
    else:
        path = ROOT / "output" / "y280_center_plane_kappa_vs_gradient"
    path.mkdir(parents=True, exist_ok=True)
    return path


def case_mapping() -> list[dict[str, object]]:
    return [
        {"temperature_K": 300.0, "delta_half_K": 2.5, "case_name": "T300K_pm2p5K", "run_dir": ROOT / "output" / "run_y280nm_300K_pm2p5K_rerun"},
        {"temperature_K": 300.0, "delta_half_K": 5.0, "case_name": "T300K_pm5K", "run_dir": ROOT / "output" / "run_run_input_y280nm_10nm_Eeff5e-19_T300K"},
        {"temperature_K": 300.0, "delta_half_K": 10.0, "case_name": "T300K_pm10K", "run_dir": ROOT / "output" / "run_y280nm_300K_pm10K"},
        {"temperature_K": 300.0, "delta_half_K": 20.0, "case_name": "T300K_pm20K", "run_dir": ROOT / "output" / "run_y280nm_300K_pm20K"},
        {"temperature_K": 323.0, "delta_half_K": 2.5, "case_name": "T323K_pm2p5K", "run_dir": ROOT / "output" / "run_y280nm_323K_pm2p5K"},
        {"temperature_K": 323.0, "delta_half_K": 5.0, "case_name": "T323K_pm5K", "run_dir": ROOT / "output" / "run_run_input_y280nm_10nm_Eeff5e-19_T323K"},
        {"temperature_K": 323.0, "delta_half_K": 10.0, "case_name": "T323K_pm10K", "run_dir": ROOT / "output" / "run_y280nm_323K_pm10K"},
        {"temperature_K": 323.0, "delta_half_K": 20.0, "case_name": "T323K_pm20K", "run_dir": ROOT / "output" / "run_y280nm_323K_pm20K"},
        {"temperature_K": 373.0, "delta_half_K": 2.5, "case_name": "T373K_pm2p5K", "run_dir": ROOT / "output" / "run_y280nm_373K_pm2p5K"},
        {"temperature_K": 373.0, "delta_half_K": 5.0, "case_name": "T373K_pm5K", "run_dir": ROOT / "output" / "run_run_input_y280nm_10nm_Eeff5e-19_T373K"},
        {"temperature_K": 373.0, "delta_half_K": 10.0, "case_name": "T373K_pm10K", "run_dir": ROOT / "output" / "run_y280nm_373K_pm10K"},
        {"temperature_K": 373.0, "delta_half_K": 20.0, "case_name": "T373K_pm20K", "run_dir": ROOT / "output" / "run_y280nm_373K_pm20K"},
    ]


def averaged_middle_plane_flux(run_dir: Path, step_start: int, step_end: int, middle_label: str) -> dict[str, object]:
    steps = resolve_steps(run_dir, [int(step_start), int(step_end)])
    step_tables = [load_step_flux_table(run_dir, step, "interval") for step in steps]
    tbl = aggregate_flux_tables(step_tables, steps, "interval")
    row = tbl.loc[tbl["label"].astype(str) == str(middle_label)]
    if row.empty:
        raise RuntimeError(f"{run_dir}: label {middle_label!r} not found")
    out = row.iloc[0].to_dict()
    out["step_count"] = int(out["step_count"])
    return out


def build_dataset(step_start: int, step_end: int, length_nm: float, middle_label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    L_m = float(length_nm) * 1e-9
    for case in case_mapping():
        avg_row = averaged_middle_plane_flux(Path(case["run_dir"]), step_start, step_end, middle_label)
        delta_half = float(case["delta_half_K"])
        delta_total = 2.0 * delta_half
        gradient = delta_total / L_m
        heat_flux = float(avg_row["heat_flux_W_m2"])
        rows.append(
            {
                "case_name": str(case["case_name"]),
                "temperature_K": float(case["temperature_K"]),
                "delta_half_K": delta_half,
                "delta_total_K": delta_total,
                "length_nm": float(length_nm),
                "gradient_K_per_m": gradient,
                "middle_label": str(middle_label),
                "heat_flux_W_m2": heat_flux,
                "kappa_div_grad_W_mK": heat_flux / gradient,
                "kappa_fourier_W_mK": -heat_flux / gradient,
                "step_start": int(step_start),
                "step_end": int(step_end),
                "step_count": int(avg_row["step_count"]),
                "steps_used": str(avg_row["steps_used"]),
                "run_dir": str(case["run_dir"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["temperature_K", "delta_half_K"], kind="stable").reset_index(drop=True)


def plot_dataset(df: pd.DataFrame, output_png: Path, output_pdf: Path, dpi: int) -> None:
    paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    colors = {300.0: "#0f4c81", 323.0: "#b45309", 373.0: "#9f1239"}
    markers = {300.0: "o", 323.0: "s", 373.0: "D"}
    for temperature_K in sorted(df["temperature_K"].drop_duplicates().astype(float).tolist()):
        sub = df.loc[df["temperature_K"] == temperature_K].sort_values("gradient_K_per_m", kind="stable")
        ax.plot(
            sub["gradient_K_per_m"].to_numpy(dtype=np.float64),
            sub["kappa_fourier_W_mK"].to_numpy(dtype=np.float64),
            color=colors.get(temperature_K, None),
            marker=markers.get(temperature_K, "o"),
            linewidth=2.2,
            markersize=6.0,
            label=f"{temperature_K:.0f} K",
        )
    ax.set_xlabel(r"Temperature Gradient, $\Delta T/L$ (K m$^{-1}$)")
    ax.set_ylabel(r"Thermal Conductivity, $\kappa$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title("280 nm Center-Plane Thermal Conductivity vs Temperature Gradient")
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.28)
    ax.minorticks_on()
    ax.legend(loc="best", frameon=False)
    ax.text(
        0.02,
        0.04,
        "Middle monitor plane only\nConductivity uses Fourier sign: $\\kappa=-q/(\\Delta T/L)$",
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
    output_dir = resolve_output_dir(args.output_dir)
    df = build_dataset(int(args.step_start), int(args.step_end), float(args.length_nm), str(args.middle_label))
    stem = f"y280_center_plane_kappa_vs_gradient_{int(round(float(args.step_start))):05d}_{int(round(float(args.step_end))):05d}"
    csv_path = output_dir / f"{stem}.csv"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    df.to_csv(csv_path, index=False)
    plot_dataset(df, png_path, pdf_path, int(args.dpi))
    print(f"[ok] csv -> {csv_path}")
    print(f"[ok] png -> {png_path}")
    print(f"[ok] pdf -> {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
