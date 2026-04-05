from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-monitor net heat flux from solver output. "
            "With one --step value, export that output step. "
            "With two --step values, average all available output steps in the inclusive range."
        )
    )
    parser.add_argument("--run-dir", required=True, help="Run output directory, e.g. output/run_20260316_154407")
    parser.add_argument(
        "--step",
        required=True,
        nargs="+",
        type=int,
        help="One step number, or two step numbers defining an inclusive averaging window over all available output steps",
    )
    parser.add_argument(
        "--stat-type",
        choices=("interval", "cumulative"),
        default="interval",
        help="Which heat-flux statistic to export. Default: interval",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Output CSV path. Default: <run-dir>/heat_flux_step_*.csv or heat_flux_steps_*_avg.csv",
    )
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


def resolve_steps(run_dir: Path, step_args: list[int]) -> list[int]:
    if len(step_args) not in (1, 2):
        raise ValueError("--step requires one value, or two values defining an averaging window")
    available = list_output_steps(run_dir)
    if len(step_args) == 1:
        step = int(step_args[0])
        if step not in available:
            raise FileNotFoundError(f"requested step not found: {step}")
        return [step]
    lo, hi = sorted(int(v) for v in step_args)
    selected = [step for step in available if lo <= step <= hi]
    if not selected:
        raise FileNotFoundError(f"no output steps found between {lo} and {hi}")
    return selected


def build_output_csv(run_dir: Path, steps: list[int], stat_type: str, output_csv: str) -> Path:
    if output_csv:
        path = Path(output_csv).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    if len(steps) == 1:
        path = run_dir / f"heat_flux_{stat_type}_step_{steps[0]:05d}.csv"
    else:
        path = run_dir / f"heat_flux_{stat_type}_steps_{steps[0]:05d}_{steps[-1]:05d}_avg.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def select_flux_rows(table: pd.DataFrame, stat_type: str) -> pd.DataFrame:
    out = table.copy()
    if "stat_type" in out.columns:
        out = out.loc[out["stat_type"].astype(str).str.lower() == stat_type.lower()]
    if out.empty:
        raise RuntimeError(f"no rows found for stat_type={stat_type!r}")
    return out


def extract_net_flux(table: pd.DataFrame, stat_type: str) -> pd.DataFrame:
    rows = select_flux_rows(table, stat_type)
    if "net_W_m2" in rows.columns:
        flux = rows["net_W_m2"].to_numpy(dtype=np.float64)
    elif stat_type.lower() == "interval" and "flux_interval_W_m2" in rows.columns:
        flux = rows["flux_interval_W_m2"].to_numpy(dtype=np.float64)
    elif stat_type.lower() == "cumulative" and "flux_cumulative_W_m2" in rows.columns:
        flux = rows["flux_cumulative_W_m2"].to_numpy(dtype=np.float64)
    elif "flux_interval_net_W_m2" in rows.columns and stat_type.lower() == "interval":
        flux = rows["flux_interval_net_W_m2"].to_numpy(dtype=np.float64)
    elif "flux_cumulative_net_W_m2" in rows.columns and stat_type.lower() == "cumulative":
        flux = rows["flux_cumulative_net_W_m2"].to_numpy(dtype=np.float64)
    else:
        raise RuntimeError("unsupported heat_flux.txt format: unable to locate net heat-flux column")
    out = rows.copy()
    out["heat_flux_W_m2"] = flux
    return out


def load_step_flux_table(run_dir: Path, step: int, stat_type: str) -> pd.DataFrame:
    heat_file = run_dir / "steps" / f"step_{step:05d}" / "heat_flux.txt"
    if not heat_file.is_file():
        raise FileNotFoundError(heat_file)
    table = pd.read_csv(heat_file)
    table = extract_net_flux(table, stat_type)
    table["step"] = int(step)
    return table


def aggregate_flux_tables(step_tables: list[pd.DataFrame], steps: list[int], stat_type: str) -> pd.DataFrame:
    long_table = pd.concat(step_tables, ignore_index=True)
    group_cols = ["label"]
    keep_cols = [col for col in ("requested_direction", "effective_normal", "warning") if col in long_table.columns]
    agg_rows: list[dict[str, object]] = []
    for label, group in long_table.groupby("label", sort=False):
        row: dict[str, object] = {
            "aggregation": "single" if len(steps) == 1 else "mean",
            "stat_type": stat_type,
            "step_start": int(steps[0]),
            "step_end": int(steps[-1]),
            "step_count": int(len(steps)),
            "steps_used": ";".join(f"{step:05d}" for step in steps),
            "label": str(label),
            "heat_flux_W_m2": float(group["heat_flux_W_m2"].mean()),
        }
        for col in keep_cols:
            values = group[col].dropna().astype(str).unique().tolist()
            row[col] = values[0] if len(values) == 1 else ";".join(values)
        agg_rows.append(row)
    out = pd.DataFrame(agg_rows)
    if not out.empty:
        out = out.sort_values("label", kind="stable").reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    steps = resolve_steps(run_dir, args.step)
    output_csv = build_output_csv(run_dir, steps, args.stat_type, args.output_csv)
    step_tables = [load_step_flux_table(run_dir, step, args.stat_type) for step in steps]
    out = aggregate_flux_tables(step_tables, steps, args.stat_type)
    out.to_csv(output_csv, index=False)
    print(f"saved {output_csv}")


if __name__ == "__main__":
    main()
