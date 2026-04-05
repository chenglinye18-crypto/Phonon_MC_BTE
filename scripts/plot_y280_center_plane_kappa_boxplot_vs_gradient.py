from __future__ import annotations

import argparse
import re
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

from scripts.plot_kappa_vs_temperature import paper_style


COLORS = {
    300.0: "#0f4c81",
    323.0: "#b45309",
    373.0: "#9f1239",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build boxplots for the 280 nm temperature-gradient dataset using the center monitor, "
            "interval heat flux, and rolling-window samples."
        )
    )
    parser.add_argument("--step-min", type=int, default=32000, help="Use output steps >= this step. Default: 32000")
    parser.add_argument("--tail-count", type=int, default=5, help="Keep only the last N qualifying output steps. Default: 5")
    parser.add_argument("--window-max", type=int, default=5, help="Maximum rolling window length. Default: 5")
    parser.add_argument("--length-nm", type=float, default=280.0, help="Transport length. Default: 280 nm")
    parser.add_argument("--middle-label", default="flux_plane_003", help="Center monitor-plane label. Default: flux_plane_003")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI. Default: 300")
    parser.add_argument(
        "--output-dir",
        default="output/y280_center_plane_kappa_boxplot_gradient_interval_tail5",
        help="Output directory.",
    )
    return parser.parse_args()


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


def list_output_steps(run_dir: Path, step_min: int) -> list[int]:
    steps: list[int] = []
    for step_dir in sorted((run_dir / "steps").glob("step_*")):
        match = re.fullmatch(r"step_(\d+)", step_dir.name)
        if match is None:
            continue
        step = int(match.group(1))
        if step >= step_min:
            steps.append(step)
    if not steps:
        raise FileNotFoundError(f"no output steps >= {step_min} found under {run_dir / 'steps'}")
    return steps


def load_interval_flux_for_label(run_dir: Path, step: int, label: str) -> float:
    path = run_dir / "steps" / f"step_{step:05d}" / "heat_flux.txt"
    table = pd.read_csv(path)
    if "stat_type" in table.columns:
        table = table.loc[table["stat_type"].astype(str).str.lower() == "interval"].copy()
    table = table.loc[table["label"].astype(str) == str(label)].copy()
    if table.empty:
        raise RuntimeError(f"{path}: label {label!r} not found")
    row = table.iloc[0]
    if "net_W_m2" in row.index:
        return float(row["net_W_m2"])
    if "flux_interval_W_m2" in row.index:
        return float(row["flux_interval_W_m2"])
    if "flux_interval_net_W_m2" in row.index:
        return float(row["flux_interval_net_W_m2"])
    raise RuntimeError(f"unsupported heat_flux.txt format: {path}")


def build_rolling_samples(steps: list[int], values: list[float], window_max: int) -> list[dict[str, object]]:
    n = len(steps)
    if n != len(values):
        raise ValueError("steps and values length mismatch")
    out: list[dict[str, object]] = []
    sample_index = 1
    for win in range(1, min(max(int(window_max), 1), n) + 1):
        for start in range(0, n - win + 1):
            stop = start + win
            step_block = [int(v) for v in steps[start:stop]]
            val_block = [float(v) for v in values[start:stop]]
            out.append(
                {
                    "sample_index": sample_index,
                    "window_len": win,
                    "step_start": int(step_block[0]),
                    "step_end": int(step_block[-1]),
                    "step_count": int(len(step_block)),
                    "steps_used": ";".join(f"{step:05d}" for step in step_block),
                    "heat_flux_avg_W_m2": float(np.mean(val_block)),
                }
            )
            sample_index += 1
    return out


def build_dataset(step_min: int, tail_count: int, window_max: int, length_nm: float, middle_label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    L_m = float(length_nm) * 1e-9
    for case in case_mapping():
        run_dir = Path(case["run_dir"])
        steps = list_output_steps(run_dir, int(step_min))
        if tail_count > 0 and len(steps) > tail_count:
            steps = steps[-tail_count:]
        flux_values = [load_interval_flux_for_label(run_dir, step, str(middle_label)) for step in steps]
        samples = build_rolling_samples(steps, flux_values, int(window_max))
        delta_half = float(case["delta_half_K"])
        delta_total = 2.0 * delta_half
        gradient = delta_total / L_m
        for sample in samples:
            heat_flux = float(sample["heat_flux_avg_W_m2"])
            rows.append(
                {
                    "case_name": str(case["case_name"]),
                    "temperature_K": float(case["temperature_K"]),
                    "delta_half_K": delta_half,
                    "delta_total_K": delta_total,
                    "length_nm": float(length_nm),
                    "gradient_K_per_m": gradient,
                    "middle_label": str(middle_label),
                    "sample_index": int(sample["sample_index"]),
                    "window_len": int(sample["window_len"]),
                    "step_start": int(sample["step_start"]),
                    "step_end": int(sample["step_end"]),
                    "step_count": int(sample["step_count"]),
                    "steps_used": str(sample["steps_used"]),
                    "heat_flux_avg_W_m2": heat_flux,
                    "kappa_div_grad_W_mK": heat_flux / gradient,
                    "kappa_fourier_W_mK": -heat_flux / gradient,
                    "run_dir": str(run_dir),
                }
            )
    return pd.DataFrame(rows).sort_values(["temperature_K", "delta_half_K", "sample_index"], kind="stable").reset_index(drop=True)


def plot_dataset(df: pd.DataFrame, output_png: Path, output_pdf: Path, dpi: int) -> None:
    paper_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    delta_values = sorted(df["delta_half_K"].drop_duplicates().astype(float).tolist())
    base_positions = np.arange(len(delta_values), dtype=np.float64)
    offsets = {300.0: -0.24, 323.0: 0.0, 373.0: 0.24}
    width = 0.20
    gradient_lookup = {dv: 2.0 * dv / (280e-9) / 1e7 for dv in delta_values}

    for temperature_K in sorted(df["temperature_K"].drop_duplicates().astype(float).tolist()):
        median_x: list[float] = []
        median_y: list[float] = []
        for i, delta_half in enumerate(delta_values):
            sub = df.loc[(df["temperature_K"] == temperature_K) & (df["delta_half_K"] == delta_half)]
            if sub.empty:
                continue
            pos = base_positions[i] + offsets[temperature_K]
            data = sub["kappa_fourier_W_mK"].to_numpy(dtype=np.float64)
            ax.boxplot(
                [data],
                positions=[pos],
                widths=width,
                patch_artist=True,
                showfliers=True,
                medianprops={"color": "black", "linewidth": 1.1},
                boxprops={"facecolor": COLORS[temperature_K], "edgecolor": COLORS[temperature_K], "alpha": 0.55, "linewidth": 1.0},
                whiskerprops={"color": COLORS[temperature_K], "linewidth": 1.0},
                capprops={"color": COLORS[temperature_K], "linewidth": 1.0},
                flierprops={"marker": "o", "markersize": 3, "markerfacecolor": COLORS[temperature_K], "markeredgecolor": COLORS[temperature_K], "alpha": 0.55},
            )
            median_x.append(pos)
            median_y.append(float(np.median(data)))
        if median_x:
            ax.plot(median_x, median_y, color=COLORS[temperature_K], linewidth=1.2, marker="o", markersize=3.5, label=f"{temperature_K:.0f} K")

    ax.set_xticks(base_positions)
    ax.set_xticklabels([f"{gradient_lookup[dv]:.2f}" for dv in delta_values])
    ax.set_xlabel(r"Temperature Gradient, $\Delta T/L$ ($10^7$ K m$^{-1}$)")
    ax.set_ylabel(r"Thermal Conductivity, $\kappa$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title("280 nm Center-Plane Thermal Conductivity vs Temperature Gradient\n(interval + rolling window)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = build_dataset(int(args.step_min), int(args.tail_count), int(args.window_max), float(args.length_nm), str(args.middle_label))
    stem = f"y280_center_plane_kappa_boxplot_gradient_interval_tail{int(args.tail_count)}"
    csv_path = output_dir / f"{stem}.csv"
    summary_path = output_dir / f"{stem}_summary.csv"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    df.to_csv(csv_path, index=False)
    summary = (
        df.groupby(["temperature_K", "delta_half_K"], sort=True)["kappa_fourier_W_mK"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n_samples", "mean": "mean_W_mK", "std": "std_W_mK", "median": "median_W_mK", "min": "min_W_mK", "max": "max_W_mK"})
    )
    q = df.groupby(["temperature_K", "delta_half_K"], sort=True)["kappa_fourier_W_mK"].quantile([0.25, 0.75]).unstack().reset_index()
    q = q.rename(columns={0.25: "q1_W_mK", 0.75: "q3_W_mK"})
    summary = summary.merge(q, on=["temperature_K", "delta_half_K"], how="left")
    summary.to_csv(summary_path, index=False)
    plot_dataset(df, png_path, pdf_path, int(args.dpi))
    print(f"[ok] csv -> {csv_path}")
    print(f"[ok] summary -> {summary_path}")
    print(f"[ok] png -> {png_path}")
    print(f"[ok] pdf -> {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
