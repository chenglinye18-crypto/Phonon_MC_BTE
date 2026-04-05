from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export full temperature-field CSV from solver output. "
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
        "--output-csv",
        default="",
        help="Output CSV path. Default: <run-dir>/temperature_step_*.csv or temperature_steps_*_avg.csv",
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


def build_output_csv(run_dir: Path, steps: list[int], output_csv: str) -> Path:
    if output_csv:
        path = Path(output_csv).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    if len(steps) == 1:
        path = run_dir / f"temperature_step_{steps[0]:05d}.csv"
    else:
        path = run_dir / f"temperature_steps_{steps[0]:05d}_{steps[-1]:05d}_avg.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_step_temperature(run_dir: Path, step: int) -> pd.DataFrame:
    temp_file = run_dir / "steps" / f"step_{step:05d}" / "temperature.txt"
    if not temp_file.is_file():
        raise FileNotFoundError(temp_file)
    table = pd.read_csv(temp_file)
    needed = ["idxcell", "idycell", "idzcell", "Temperature"]
    missing = [col for col in needed if col not in table.columns]
    if missing:
        raise RuntimeError(f"{temp_file} is missing columns: {missing}")
    return table.loc[:, needed].sort_values(["idxcell", "idycell", "idzcell"], kind="stable").reset_index(drop=True)


def aggregate_temperature_tables(step_tables: list[pd.DataFrame], steps: list[int]) -> pd.DataFrame:
    base = step_tables[0].copy()
    temp_sum = base["Temperature"].to_numpy(dtype=np.float64).copy()
    ref_idx = base.loc[:, ["idxcell", "idycell", "idzcell"]].to_numpy(dtype=np.int64)
    for table, step in zip(step_tables[1:], steps[1:]):
        idx = table.loc[:, ["idxcell", "idycell", "idzcell"]].to_numpy(dtype=np.int64)
        if idx.shape != ref_idx.shape or not np.array_equal(idx, ref_idx):
            raise RuntimeError(f"inconsistent temperature field indexing at step {step}")
        temp_sum += table["Temperature"].to_numpy(dtype=np.float64)
    temp_avg = temp_sum / float(len(step_tables))
    out = base.loc[:, ["idxcell", "idycell", "idzcell"]].copy()
    out.insert(0, "aggregation", "single" if len(steps) == 1 else "mean")
    out.insert(1, "step_start", int(steps[0]))
    out.insert(2, "step_end", int(steps[-1]))
    out.insert(3, "step_count", int(len(steps)))
    out.insert(4, "steps_used", ";".join(f"{step:05d}" for step in steps))
    out["temperature_K"] = temp_avg
    return out


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    steps = resolve_steps(run_dir, args.step)
    output_csv = build_output_csv(run_dir, steps, args.output_csv)
    step_tables = [load_step_temperature(run_dir, step) for step in steps]
    out = aggregate_temperature_tables(step_tables, steps)
    out.to_csv(output_csv, index=False)
    print(f"saved {output_csv}")


if __name__ == "__main__":
    main()
