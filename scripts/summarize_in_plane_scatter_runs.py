from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize in-plane SCATTER batch runs into averaged heat flux and conductivity.")
    parser.add_argument("--manifest-csv", required=True, help="CSV created by run_in_plane_scatter_x_sweep.sh")
    parser.add_argument("--step", nargs=2, type=int, default=(32000, 40000), help="Inclusive averaging window")
    parser.add_argument("--output-dir", default="", help="Summary output directory")
    return parser.parse_args()


def list_output_steps(run_dir: Path) -> list[int]:
    steps: list[int] = []
    for step_dir in sorted((run_dir / "steps").glob("step_*")):
        match = re.fullmatch(r"step_(\d+)", step_dir.name)
        if match is not None:
            steps.append(int(match.group(1)))
    if not steps:
        raise FileNotFoundError(f"no step directories found under {run_dir / 'steps'}")
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


def resolve_steps(run_dir: Path, step_lo: int, step_hi: int) -> tuple[Path, list[int]]:
    lo, hi = sorted((int(step_lo), int(step_hi)))
    last_error: Exception | None = None
    for cand in reversed(candidate_run_dirs(run_dir)):
        try:
            selected = [step for step in list_output_steps(cand) if lo <= step <= hi]
        except FileNotFoundError as exc:
            last_error = exc
            continue
        if selected:
            return cand, selected
    if last_error is not None:
        raise FileNotFoundError(f"no output steps found between {lo} and {hi} for {run_dir} or its reruns") from last_error
    raise FileNotFoundError(f"no output steps found between {lo} and {hi} for {run_dir} or its reruns")


def load_heat_flux_table(run_dir: Path, step: int) -> pd.DataFrame:
    path = run_dir / "steps" / f"step_{step:05d}" / "heat_flux.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    if "stat_type" in table.columns:
        table = table.loc[table["stat_type"].astype(str).str.lower() == "interval"].copy()
    if "net_W_m2" in table.columns:
        table["heat_flux_W_m2"] = table["net_W_m2"].astype(float)
    elif "flux_interval_W_m2" in table.columns:
        table["heat_flux_W_m2"] = table["flux_interval_W_m2"].astype(float)
    elif "flux_interval_net_W_m2" in table.columns:
        table["heat_flux_W_m2"] = table["flux_interval_net_W_m2"].astype(float)
    else:
        raise RuntimeError(f"unsupported heat flux file format: {path}")
    needed = [col for col in ("label", "heat_flux_W_m2") if col in table.columns]
    return table.loc[:, needed].copy()


def summarize_one_run(run_dir: Path, step_lo: int, step_hi: int) -> pd.DataFrame:
    resolved_run_dir, steps = resolve_steps(run_dir, step_lo, step_hi)
    tables = []
    for step in steps:
        table = load_heat_flux_table(resolved_run_dir, step)
        table["step"] = step
        tables.append(table)
    long_table = pd.concat(tables, ignore_index=True)
    out = (
        long_table.groupby("label", sort=False)["heat_flux_W_m2"]
        .mean()
        .reset_index()
        .rename(columns={"heat_flux_W_m2": "heat_flux_avg_W_m2"})
    )
    out["actual_run_dir"] = str(resolved_run_dir)
    out["step_start"] = int(steps[0])
    out["step_end"] = int(steps[-1])
    out["step_count"] = int(len(steps))
    out["steps_used"] = ";".join(f"{step:05d}" for step in steps)
    return out


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_csv).expanduser().resolve()
    manifest = pd.read_csv(manifest_path)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else manifest_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    step_lo, step_hi = (int(args.step[0]), int(args.step[1]))

    per_case_dir = output_dir / "per_case"
    per_case_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    for rec in manifest.to_dict(orient="records"):
        run_dir = Path(str(rec["run_dir"])).expanduser().resolve()
        summary = summarize_one_run(run_dir, step_lo, step_hi)
        summary.insert(0, "run_tag", str(rec["run_tag"]))
        summary.insert(1, "temperature_K", float(rec["temperature_K"]))
        summary.insert(2, "width_um", float(rec["width_um"]))
        summary.insert(3, "deltaT_K", float(rec["deltaT_K"]))
        summary.insert(4, "transport_length_m", float(rec["transport_length_m"]))
        grad = float(rec["deltaT_K"]) / max(float(rec["transport_length_m"]), np.finfo(np.float64).eps)
        summary.insert(5, "gradient_K_per_m", grad)
        summary["kappa_div_W_mK"] = summary["heat_flux_avg_W_m2"] / grad
        summary["kappa_fourier_W_mK"] = -summary["heat_flux_avg_W_m2"] / grad
        rows.append(summary)
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(rec["run_tag"]))
        summary.to_csv(per_case_dir / f"{safe_tag}__heat_flux_interval_steps_{step_lo:05d}_{step_hi:05d}_avg.csv", index=False)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(output_dir / "heat_flux_kappa_summary.csv", index=False)
    print(f"saved {output_dir / 'heat_flux_kappa_summary.csv'}")


if __name__ == "__main__":
    main()
