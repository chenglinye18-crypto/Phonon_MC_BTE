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


TEMPS_K = [300.0, 323.0, 373.0]
COLORS = {
    300.0: "#0f4c81",
    323.0: "#b45309",
    373.0: "#9f1239",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build in-plane SCATTER thermal-conductivity boxplots versus x width from existing MC outputs. "
            "Samples use interval heat flux with rolling-window averages over steps >= step-min."
        )
    )
    parser.add_argument(
        "--manifest-csv",
        default="output/in_plane_scatter_full_batch/case_manifest.csv",
        help="Batch manifest CSV produced by the in-plane sweep script.",
    )
    parser.add_argument("--step-min", type=int, default=32000, help="Use output steps >= this step. Default: 32000")
    parser.add_argument("--tail-count", type=int, default=5, help="Keep only the last N qualifying output steps. Default: 5")
    parser.add_argument("--window-max", type=int, default=5, help="Maximum rolling-window length. Default: 5")
    parser.add_argument(
        "--monitor-label",
        default="flux_plane_004",
        help="Monitor label used for conductivity extraction. Default: flux_plane_004",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI. Default: 300")
    parser.add_argument(
        "--output-dir",
        default="output/in_plane_scatter_width_boxplot_interval_tail5",
        help="Output directory.",
    )
    return parser.parse_args()


def list_output_steps(run_dir: Path, step_min: int) -> list[int]:
    steps: list[int] = []
    for step_dir in sorted((run_dir / "steps").glob("step_*")):
        name = step_dir.name
        if not name.startswith("step_"):
            continue
        try:
            step = int(name.split("_", 1)[1])
        except ValueError:
            continue
        if step >= step_min:
            steps.append(step)
    if not steps:
        raise FileNotFoundError(f"no output steps >= {step_min} found under {run_dir / 'steps'}")
    return steps


def rerun_index(run_dir: Path, base_name: str) -> int:
    if run_dir.name == base_name:
        return 0
    match = re.fullmatch(re.escape(base_name) + r"_(\d+)", run_dir.name)
    if match is None:
        return -1
    return int(match.group(1))


def candidate_run_dirs(run_dir: Path) -> list[Path]:
    parent = run_dir.parent
    base_name = run_dir.name
    candidates: list[Path] = []
    for cand in parent.glob(base_name + "*"):
        if not cand.is_dir():
            continue
        if rerun_index(cand, base_name) >= 0:
            candidates.append(cand.resolve())
    return sorted(set(candidates), key=lambda path: rerun_index(path, base_name))


def resolve_run_dir(run_dir: Path, step_min: int) -> tuple[Path, list[int]]:
    last_error: Exception | None = None
    for cand in reversed(candidate_run_dirs(run_dir)):
        try:
            steps = list_output_steps(cand, step_min)
        except FileNotFoundError as exc:
            last_error = exc
            continue
        if steps:
            return cand, steps
    if last_error is not None:
        raise FileNotFoundError(f"no output steps >= {step_min} for {run_dir} or its reruns") from last_error
    raise FileNotFoundError(f"no output steps >= {step_min} for {run_dir} or its reruns")


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
    if len(steps) != len(values):
        raise ValueError("steps and values length mismatch")
    n = len(steps)
    if n == 0:
        return []
    out: list[dict[str, object]] = []
    sample_index = 1
    for win in range(1, min(max(int(window_max), 1), n) + 1):
        for start in range(0, n - win + 1):
            stop = start + win
            step_block = [int(v) for v in steps[start:stop]]
            value_block = [float(v) for v in values[start:stop]]
            out.append(
                {
                    "sample_index": sample_index,
                    "window_len": win,
                    "step_start": int(step_block[0]),
                    "step_end": int(step_block[-1]),
                    "step_count": int(len(step_block)),
                    "steps_used": ";".join(f"{step:05d}" for step in step_block),
                    "heat_flux_avg_W_m2": float(np.mean(value_block)),
                }
            )
            sample_index += 1
    return out


def load_manifest(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"run_tag", "run_dir", "width_nm", "width_label", "temperature_K", "deltaT_K", "transport_length_m"}
    missing = required.difference(table.columns)
    if missing:
        raise RuntimeError(f"manifest missing columns: {sorted(missing)}")
    return table.copy()


def build_dataset(manifest: pd.DataFrame, step_min: int, tail_count: int, window_max: int, monitor_label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rec in manifest.to_dict(orient="records"):
        run_dir = Path(str(rec["run_dir"])).expanduser().resolve()
        resolved_run_dir, steps = resolve_run_dir(run_dir, int(step_min))
        if tail_count > 0 and len(steps) > tail_count:
            steps = steps[-tail_count:]
        flux_values = [load_interval_flux_for_label(resolved_run_dir, step, str(monitor_label)) for step in steps]
        samples = build_rolling_samples(steps, flux_values, int(window_max))
        delta_t = float(rec["deltaT_K"])
        transport_length_m = float(rec["transport_length_m"])
        gradient = delta_t / max(transport_length_m, np.finfo(np.float64).eps)
        for sample in samples:
            heat_flux = float(sample["heat_flux_avg_W_m2"])
            rows.append(
                {
                    "run_tag": str(rec["run_tag"]),
                    "width_nm": float(rec["width_nm"]),
                    "width_label": str(rec["width_label"]),
                    "width_um": float(rec["width_um"]) if "width_um" in rec else float(rec["width_nm"]) * 1e-3,
                    "temperature_K": float(rec["temperature_K"]),
                    "deltaT_K": delta_t,
                    "transport_length_m": transport_length_m,
                    "gradient_K_per_m": gradient,
                    "monitor_label": str(monitor_label),
                    "sample_index": int(sample["sample_index"]),
                    "window_len": int(sample["window_len"]),
                    "step_start": int(sample["step_start"]),
                    "step_end": int(sample["step_end"]),
                    "step_count": int(sample["step_count"]),
                    "steps_used": str(sample["steps_used"]),
                    "heat_flux_avg_W_m2": heat_flux,
                    "kappa_div_W_mK": heat_flux / gradient,
                    "kappa_fourier_W_mK": -heat_flux / gradient,
                    "run_dir": str(resolved_run_dir),
                }
            )
    return pd.DataFrame(rows).sort_values(["temperature_K", "width_nm", "sample_index"], kind="stable").reset_index(drop=True)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["temperature_K", "width_nm", "width_label"], sort=True)["kappa_fourier_W_mK"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "n_samples",
                "mean": "mean_W_mK",
                "std": "std_W_mK",
                "median": "median_W_mK",
                "min": "min_W_mK",
                "max": "max_W_mK",
            }
        )
    )
    q = df.groupby(["temperature_K", "width_nm", "width_label"], sort=True)["kappa_fourier_W_mK"].quantile([0.25, 0.75]).unstack().reset_index()
    q = q.rename(columns={0.25: "q1_W_mK", 0.75: "q3_W_mK"})
    return summary.merge(q, on=["temperature_K", "width_nm", "width_label"], how="left")


def plot_dataset(df: pd.DataFrame, output_png: Path, output_pdf: Path, dpi: int) -> None:
    paper_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    width_values = sorted(df["width_nm"].drop_duplicates().astype(float).tolist())
    width_labels = [f"{int(round(v))}" if abs(v - round(v)) < 1e-9 else f"{v:g}" for v in width_values]
    base_positions = np.arange(len(width_values), dtype=np.float64)
    offsets = {300.0: -0.24, 323.0: 0.0, 373.0: 0.24}
    width = 0.20

    for temperature_K in TEMPS_K:
        median_x: list[float] = []
        median_y: list[float] = []
        for i, width_nm in enumerate(width_values):
            sub = df.loc[(df["temperature_K"] == temperature_K) & (df["width_nm"] == width_nm)]
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
    ax.set_xticklabels(width_labels)
    ax.set_xlabel("In-Plane Width (nm)")
    ax.set_ylabel(r"Thermal Conductivity, $\kappa$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title("In-Plane SCATTER Thermal Conductivity vs Width\n(interval + rolling window)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    manifest_path = (ROOT / args.manifest_csv).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    df = build_dataset(manifest, int(args.step_min), int(args.tail_count), int(args.window_max), str(args.monitor_label))
    stem = f"in_plane_scatter_kappa_boxplot_interval_tail{int(args.tail_count)}"
    csv_path = output_dir / f"{stem}.csv"
    summary_path = output_dir / f"{stem}_summary.csv"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    df.to_csv(csv_path, index=False)
    build_summary(df).to_csv(summary_path, index=False)
    plot_dataset(df, png_path, pdf_path, int(args.dpi))
    print(f"[ok] csv -> {csv_path}")
    print(f"[ok] summary -> {summary_path}")
    print(f"[ok] png -> {png_path}")
    print(f"[ok] pdf -> {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
