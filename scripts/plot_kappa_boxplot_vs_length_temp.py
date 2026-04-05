from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LENGTHS_NM = [10, 20, 50, 70, 140, 280, 560, 1120]
TEMPS_K = [300, 323, 373]
COLORS = {
    300: "#1f77b4",
    323: "#ff7f0e",
    373: "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build boxplots of thermal conductivity versus length and temperature from existing MC outputs. "
            "Samples are formed from output steps >= step_min."
        )
    )
    parser.add_argument("--output-dir", default="output/kappa_boxplot_length_temp", help="Output directory.")
    parser.add_argument("--step-min", type=int, default=32000, help="Use output steps >= this step. Default: 32000")
    parser.add_argument(
        "--stat-type",
        choices=("interval", "cumulative"),
        default="interval",
        help="Use interval or cumulative heat-flux rows. Default: interval",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("grouped", "individual", "rolling"),
        default="grouped",
        help="Use grouped averages or each output step as one sample. Default: grouped",
    )
    parser.add_argument(
        "--group-count",
        type=int,
        default=5,
        help="Number of consecutive averaged sample groups per case when sample-mode=grouped. Default: 5",
    )
    parser.add_argument(
        "--tail-count",
        type=int,
        default=0,
        help="If > 0, keep only the last N output steps after step-min. Useful for comparable interval statistics. Default: 0",
    )
    parser.add_argument(
        "--window-max",
        type=int,
        default=5,
        help="Maximum rolling window length when sample-mode=rolling. Default: 5",
    )
    parser.add_argument(
        "--mixed-1120-cumulative",
        action="store_true",
        help=(
            "Use cumulative individual samples for (1120 nm, 300/323 K), "
            "and interval rolling-window samples for all other cases."
        ),
    )
    return parser.parse_args()


def resolve_run_dir(root: Path, length_nm: int, temp_k: int) -> Path:
    primary = root / "output" / f"run_run_input_y{length_nm}nm_{'1' if length_nm in {10,20,50,70} else '10'}nm_Eeff5e-19_T{temp_k}K"
    if primary.is_dir():
        return primary
    if length_nm == 560 and temp_k == 300:
        fallback = root / "output" / "run_test_y560_10nm_300K_Eeff5e-19_Tloc_after_debug"
        if fallback.is_dir():
            return fallback
    if length_nm == 1120 and temp_k == 300:
        fallback = root / "output" / "run_test_y1120_10nm_300K_Eeff5e-19_Tloc_after_debug"
        if fallback.is_dir():
            return fallback
    raise FileNotFoundError(f"run dir not found for L={length_nm} nm, T={temp_k} K")


def load_monitor_manifest(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "heat_flux_monitors_manifest.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    table = table.copy()
    if "label" not in table.columns:
        raise RuntimeError(f"invalid monitor manifest: {path}")
    return table


def pick_monitor_label(run_dir: Path, length_nm: int) -> str:
    table = load_monitor_manifest(run_dir)
    labels = table["label"].astype(str).tolist()
    if not labels:
        raise RuntimeError(f"no monitors found in {run_dir}")
    if length_nm == 1120:
        if len(labels) < 6:
            raise RuntimeError(f"1120 nm run does not have a sixth monitor: {run_dir}")
        return labels[5]
    return labels[len(labels) // 2]


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


def load_flux_for_label(run_dir: Path, step: int, label: str, stat_type: str) -> float:
    path = run_dir / "steps" / f"step_{step:05d}" / "heat_flux.txt"
    table = pd.read_csv(path)
    if "stat_type" in table.columns:
        table = table.loc[table["stat_type"].astype(str).str.lower() == stat_type.lower()].copy()
    table = table.loc[table["label"].astype(str) == label].copy()
    if table.empty:
        raise RuntimeError(f"monitor {label} with stat_type={stat_type} not found in {path}")
    row = table.iloc[0]
    if "net_W_m2" in row.index:
        return float(row["net_W_m2"])
    if stat_type.lower() == "interval" and "flux_interval_W_m2" in row.index:
        return float(row["flux_interval_W_m2"])
    if stat_type.lower() == "interval" and "flux_interval_net_W_m2" in row.index:
        return float(row["flux_interval_net_W_m2"])
    if stat_type.lower() == "cumulative" and "flux_cumulative_W_m2" in row.index:
        return float(row["flux_cumulative_W_m2"])
    if stat_type.lower() == "cumulative" and "flux_cumulative_net_W_m2" in row.index:
        return float(row["flux_cumulative_net_W_m2"])
    raise RuntimeError(f"unsupported heat_flux.txt format: {path}")


def load_delta_t_from_snapshot(run_dir: Path) -> float:
    candidates = [
        run_dir / "inputs" / "initial_temperature__initial_temperature.csv",
        run_dir / "initial_temperature.csv",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        table = pd.read_csv(path)
        col = "Temperature" if "Temperature" in table.columns else "temperature_K" if "temperature_K" in table.columns else None
        if col is None:
            continue
        vals = table[col].to_numpy(dtype=np.float64)
        if vals.size:
            return float(vals.max() - vals.min())
    return 10.0


def build_group_samples(steps: list[int], values: list[float], group_count: int) -> list[dict[str, object]]:
    if len(steps) != len(values):
        raise ValueError("steps and values length mismatch")
    groups = np.array_split(np.arange(len(steps)), min(max(group_count, 1), len(steps)))
    out: list[dict[str, object]] = []
    for sample_index, idxs in enumerate(groups, start=1):
        idx_list = idxs.tolist()
        step_block = [int(steps[i]) for i in idx_list]
        val_block = [float(values[i]) for i in idx_list]
        out.append(
            {
                "sample_index": sample_index,
                "step_start": int(step_block[0]),
                "step_end": int(step_block[-1]),
                "step_count": int(len(step_block)),
                "steps_used": ";".join(f"{step:05d}" for step in step_block),
                "heat_flux_avg_W_m2": float(np.mean(val_block)),
            }
        )
    return out


def build_individual_samples(steps: list[int], values: list[float]) -> list[dict[str, object]]:
    if len(steps) != len(values):
        raise ValueError("steps and values length mismatch")
    out: list[dict[str, object]] = []
    for sample_index, (step, value) in enumerate(zip(steps, values), start=1):
        out.append(
            {
                "sample_index": sample_index,
                "step_start": int(step),
                "step_end": int(step),
                "step_count": 1,
                "steps_used": f"{int(step):05d}",
                "heat_flux_avg_W_m2": float(value),
            }
        )
    return out


def build_rolling_samples(steps: list[int], values: list[float], window_max: int) -> list[dict[str, object]]:
    if len(steps) != len(values):
        raise ValueError("steps and values length mismatch")
    n = len(steps)
    if n == 0:
        return []
    wmax = min(max(int(window_max), 1), n)
    out: list[dict[str, object]] = []
    sample_index = 1
    for win in range(1, wmax + 1):
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


def case_sampling_strategy(length_nm: int, temp_k: int, args: argparse.Namespace) -> tuple[str, str, int, int]:
    if args.mixed_1120_cumulative and length_nm == 1120 and temp_k in {300, 323}:
        return "cumulative", "individual", 0, 1
    tail_count = int(args.tail_count)
    if args.mixed_1120_cumulative and tail_count <= 0:
        tail_count = 5
    return str(args.stat_type), str(args.sample_mode), tail_count, int(args.window_max)


def make_plot(raw: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, title_suffix: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, ax = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
    base_positions = np.arange(len(LENGTHS_NM), dtype=np.float64)
    offsets = {300: -0.24, 323: 0.0, 373: 0.24}
    width = 0.20

    for temp_k in TEMPS_K:
        median_x: list[float] = []
        median_y: list[float] = []
        for i, length_nm in enumerate(LENGTHS_NM):
            case = raw.loc[(raw["temperature_K"] == temp_k) & (raw["length_nm"] == length_nm)]
            if case.empty:
                continue
            pos = base_positions[i] + offsets[temp_k]
            data = case["kappa_fourier_W_mK"].to_numpy(dtype=np.float64)
            ax.boxplot(
                [data],
                positions=[pos],
                widths=width,
                patch_artist=True,
                showfliers=True,
                medianprops={"color": "black", "linewidth": 1.1},
                boxprops={"facecolor": COLORS[temp_k], "edgecolor": COLORS[temp_k], "alpha": 0.55, "linewidth": 1.0},
                whiskerprops={"color": COLORS[temp_k], "linewidth": 1.0},
                capprops={"color": COLORS[temp_k], "linewidth": 1.0},
                flierprops={"marker": "o", "markersize": 3, "markerfacecolor": COLORS[temp_k], "markeredgecolor": COLORS[temp_k], "alpha": 0.55},
            )
            median_x.append(pos)
            median_y.append(float(np.median(data)))
        if median_x:
            ax.plot(median_x, median_y, color=COLORS[temp_k], linewidth=1.2, marker="o", markersize=3.5, label=f"{temp_k} K")

    ax.set_xticks(base_positions)
    ax.set_xticklabels([str(v) for v in LENGTHS_NM])
    ax.set_xlabel("Length (nm)")
    ax.set_ylabel("Thermal Conductivity (W/mK)")
    ax.set_title(f"Length Dependence of Thermal Conductivity ({title_suffix})")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="best")

    png = output_dir / "kappa_boxplot_vs_length_temperature.png"
    pdf = output_dir / "kappa_boxplot_vs_length_temperature.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []
    for length_nm in LENGTHS_NM:
        for temp_k in TEMPS_K:
            run_dir = resolve_run_dir(root, length_nm, temp_k)
            monitor_label = pick_monitor_label(run_dir, length_nm)
            steps = list_output_steps(run_dir, args.step_min)
            stat_type_use, sample_mode_use, tail_count_use, window_max_use = case_sampling_strategy(length_nm, temp_k, args)
            if tail_count_use > 0 and len(steps) > tail_count_use:
                steps = steps[-tail_count_use:]
            flux_values = [load_flux_for_label(run_dir, step, monitor_label, stat_type_use) for step in steps]
            delta_t_k = load_delta_t_from_snapshot(run_dir)
            transport_length_m = float(length_nm) * 1e-9
            gradient_k_per_m = delta_t_k / transport_length_m
            if sample_mode_use == "individual":
                samples = build_individual_samples(steps, flux_values)
            elif sample_mode_use == "rolling":
                samples = build_rolling_samples(steps, flux_values, window_max_use)
            else:
                samples = build_group_samples(steps, flux_values, args.group_count)
            for sample in samples:
                heat_flux = float(sample["heat_flux_avg_W_m2"])
                raw_rows.append(
                    {
                        "length_nm": int(length_nm),
                        "temperature_K": int(temp_k),
                        "stat_type": stat_type_use,
                        "sample_mode": sample_mode_use,
                        "tail_count_used": int(tail_count_use),
                        "window_max_used": int(window_max_use),
                        "run_dir": str(run_dir),
                        "monitor_label": monitor_label,
                        "sample_index": int(sample["sample_index"]),
                        "window_len": int(sample.get("window_len", 1)),
                        "step_start": int(sample["step_start"]),
                        "step_end": int(sample["step_end"]),
                        "step_count": int(sample["step_count"]),
                        "steps_used": str(sample["steps_used"]),
                        "deltaT_K": float(delta_t_k),
                        "transport_length_m": float(transport_length_m),
                        "gradient_K_per_m": float(gradient_k_per_m),
                        "heat_flux_avg_W_m2": heat_flux,
                        "kappa_signed_W_mK": heat_flux / gradient_k_per_m,
                        "kappa_fourier_W_mK": -heat_flux / gradient_k_per_m,
                    }
                )

    raw = pd.DataFrame(raw_rows).sort_values(["temperature_K", "length_nm", "sample_index"], kind="stable").reset_index(drop=True)
    raw.to_csv(output_dir / "kappa_boxplot_raw_samples.csv", index=False)

    summary = (
        raw.groupby(["temperature_K", "length_nm"], sort=True)["kappa_fourier_W_mK"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n_samples", "std": "std_W_mK", "mean": "mean_W_mK", "median": "median_W_mK", "min": "min_W_mK", "max": "max_W_mK"})
    )
    q = raw.groupby(["temperature_K", "length_nm"], sort=True)["kappa_fourier_W_mK"].quantile([0.25, 0.75]).unstack().reset_index()
    q = q.rename(columns={0.25: "q1_W_mK", 0.75: "q3_W_mK"})
    summary = summary.merge(q, on=["temperature_K", "length_nm"], how="left")
    summary.to_csv(output_dir / "kappa_boxplot_summary.csv", index=False)

    if args.mixed_1120_cumulative:
        title_suffix = "mixed: 1120nm(300/323K)=cumulative, others=interval rolling"
    else:
        title_suffix = f"{args.stat_type}, {args.sample_mode}"
    make_plot(raw, summary, output_dir, title_suffix)
    print(f"saved {output_dir / 'kappa_boxplot_raw_samples.csv'}")
    print(f"saved {output_dir / 'kappa_boxplot_summary.csv'}")
    print(f"saved {output_dir / 'kappa_boxplot_vs_length_temperature.png'}")


if __name__ == "__main__":
    main()
