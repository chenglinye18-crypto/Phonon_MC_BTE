from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phonon_mc import init_mesh_from_geom, resolve_input_dir, setup_case_from_ldg_lgrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a 1D temperature line by fixing two dimensions and sweeping along the remaining axis. "
            "Use --fix twice, e.g. --fix x=1 --fix z=1."
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
    parser.add_argument("--fix", action="append", required=True, help="Fixed dimension and value, repeated twice. Example: --fix x=1 --fix z=1")
    parser.add_argument("--mode", choices=("index", "coord"), default="index", help="Interpret fixed values as cell indices or physical coordinates")
    parser.add_argument("--input-dir", default="", help="Optional input directory containing ldg.txt and lgrid.txt")
    parser.add_argument("--output", default="", help="Output PNG path. Default: <run-dir>/plots/temperature_line_*.png")
    parser.add_argument("--output-csv", default="", help="Output CSV path. Default: <run-dir>/plots/temperature_line_*.csv")
    parser.add_argument("--show", action="store_true", help="Also show the figure interactively")
    return parser.parse_args()


def parse_fixed(fix_args: list[str]) -> dict[str, float]:
    fixed: dict[str, float] = {}
    for item in fix_args:
        match = re.fullmatch(r"\s*([xyzXYZ])\s*=\s*([-+0-9.eE]+)\s*", item)
        if match is None:
            raise ValueError(f"invalid --fix value: {item!r}")
        axis = match.group(1).lower()
        fixed[axis] = float(match.group(2))
    if len(fixed) != 2:
        raise ValueError("exactly two fixed dimensions are required")
    return fixed


def resolve_case_files(run_dir: Path, input_dir: str) -> tuple[Path, Path]:
    if input_dir:
        base = Path(input_dir).expanduser().resolve()
        return base / "ldg.txt", base / "lgrid.txt"
    inputs_dir = run_dir / "inputs"
    ldg_matches = sorted(inputs_dir.glob("layout_ldg__*.txt"))
    lgrid_matches = sorted(inputs_dir.glob("grid_lgrid__*.txt"))
    if ldg_matches and lgrid_matches:
        return ldg_matches[0], lgrid_matches[0]
    fallback = resolve_input_dir()
    return fallback / "ldg.txt", fallback / "lgrid.txt"


def nearest_index(centers: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(centers - value)) + 1)


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


def build_output_paths(run_dir: Path, steps: list[int], fixed: dict[str, float], output_png: str, output_csv: str) -> tuple[Path, Path]:
    fixed_tag = "_".join(f"{axis}{value:g}" for axis, value in fixed.items())
    if len(steps) == 1:
        stem = f"temperature_line_step_{steps[0]:05d}_{fixed_tag}"
    else:
        stem = f"temperature_line_steps_{steps[0]:05d}_{steps[-1]:05d}_avg_{fixed_tag}"
    png_path = Path(output_png).expanduser().resolve() if output_png else (run_dir / "plots" / f"{stem}.png")
    csv_path = Path(output_csv).expanduser().resolve() if output_csv else (run_dir / "plots" / f"{stem}.csv")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    return png_path, csv_path


def load_line_from_step(temp_file: Path, fixed_idx: dict[str, int], free_axis: str) -> tuple[np.ndarray, np.ndarray]:
    temp_tbl = pd.read_csv(temp_file)
    idx_cols = {"x": "idxcell", "y": "idycell", "z": "idzcell"}
    filtered = temp_tbl.copy()
    for axis, idx in fixed_idx.items():
        filtered = filtered.loc[filtered[idx_cols[axis]] == idx]
    if filtered.empty:
        raise RuntimeError(f"no temperature data found for fixed dimensions {fixed_idx} at {temp_file}")
    filtered = filtered.sort_values(idx_cols[free_axis])
    free_idx = filtered[idx_cols[free_axis]].to_numpy(dtype=np.int64)
    temperature = filtered["Temperature"].to_numpy(dtype=np.float64)
    return free_idx, temperature


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    fixed_raw = parse_fixed(args.fix)
    ldg_file, lgrid_file = resolve_case_files(run_dir, args.input_dir)
    cs = setup_case_from_ldg_lgrid(ldg_file, lgrid_file, length_scale=1e-6, input_length_unit="um", verbose=False)
    mesh = init_mesh_from_geom(cs)
    centers_by_axis = {"x": np.asarray(mesh["xc"], dtype=np.float64), "y": np.asarray(mesh["yc"], dtype=np.float64), "z": np.asarray(mesh["zc"], dtype=np.float64)}
    fixed_idx: dict[str, int] = {}
    for axis, value in fixed_raw.items():
        if args.mode == "index":
            fixed_idx[axis] = int(round(value))
        else:
            fixed_idx[axis] = nearest_index(centers_by_axis[axis], float(value))
    if any(idx < 1 for idx in fixed_idx.values()):
        raise ValueError(f"invalid fixed indices: {fixed_idx}")
    free_axis = next(axis for axis in ("x", "y", "z") if axis not in fixed_idx)
    steps = resolve_steps(run_dir, args.step)
    idx_cols = {"x": "idxcell", "y": "idycell", "z": "idzcell"}
    free_idx_ref: np.ndarray | None = None
    temp_stack: list[np.ndarray] = []
    for step in steps:
        temp_file = run_dir / "steps" / f"step_{step:05d}" / "temperature.txt"
        if not temp_file.is_file():
            raise FileNotFoundError(temp_file)
        free_idx_step, temperature_step = load_line_from_step(temp_file, fixed_idx, free_axis)
        if free_idx_ref is None:
            free_idx_ref = free_idx_step
        elif free_idx_ref.shape != free_idx_step.shape or not np.array_equal(free_idx_ref, free_idx_step):
            raise RuntimeError(f"inconsistent line indices across steps, first mismatch at step {step}")
        temp_stack.append(temperature_step)
    if free_idx_ref is None or not temp_stack:
        raise RuntimeError("no temperature data resolved from the requested steps")
    free_idx = free_idx_ref
    x_coord = centers_by_axis[free_axis][free_idx - 1]
    temp_mat = np.vstack(temp_stack)
    temperature = temp_mat.mean(axis=0)
    output_path, output_csv_path = build_output_paths(run_dir, steps, fixed_raw, args.output, args.output_csv)
    csv_tbl = pd.DataFrame(
        {
            "aggregation": "mean" if len(steps) > 1 else "single",
            "step_start": int(steps[0]),
            "step_end": int(steps[-1]),
            "step_count": int(len(steps)),
            "steps_used": ";".join(f"{step:05d}" for step in steps),
            idx_cols[free_axis]: free_idx.astype(np.int64),
            f"{free_axis}_center_m": x_coord.astype(np.float64),
            "temperature_K": temperature.astype(np.float64),
        }
    )
    for axis, idx in fixed_idx.items():
        csv_tbl[idx_cols[axis]] = int(idx)
    csv_tbl.to_csv(output_csv_path, index=False)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x_coord, temperature, marker="o", linewidth=1.5, markersize=4)
    fixed_desc = ", ".join(f"{axis}={fixed_idx[axis]}" for axis in sorted(fixed_idx))
    if len(steps) == 1:
        title = f"Temperature line at step {steps[0]:05d} | fixed {fixed_desc}"
    else:
        title = f"Temperature line mean over steps {steps[0]:05d}-{steps[-1]:05d} ({len(steps)} outputs) | fixed {fixed_desc}"
    plt.title(title)
    plt.xlabel(f"{free_axis}-center (m)")
    plt.ylabel("Temperature (K)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(f"saved {output_path}")
    print(f"saved {output_csv_path}")
    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
