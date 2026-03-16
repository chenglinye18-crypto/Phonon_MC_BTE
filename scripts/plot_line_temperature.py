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
    parser.add_argument("--step", required=True, type=int, help="Step number to plot")
    parser.add_argument("--fix", action="append", required=True, help="Fixed dimension and value, repeated twice. Example: --fix x=1 --fix z=1")
    parser.add_argument("--mode", choices=("index", "coord"), default="index", help="Interpret fixed values as cell indices or physical coordinates")
    parser.add_argument("--input-dir", default="", help="Optional input directory containing ldg.txt and lgrid.txt")
    parser.add_argument("--output", default="", help="Output PNG path. Default: <run-dir>/plots/temperature_line_*.png")
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


def build_output_path(run_dir: Path, step: int, fixed: dict[str, float], output: str) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    fixed_tag = "_".join(f"{axis}{value:g}" for axis, value in fixed.items())
    path = run_dir / "plots" / f"temperature_line_step_{step:05d}_{fixed_tag}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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
    temp_file = run_dir / "steps" / f"step_{args.step:05d}" / "temperature.txt"
    if not temp_file.is_file():
        raise FileNotFoundError(temp_file)
    temp_tbl = pd.read_csv(temp_file)
    idx_cols = {"x": "idxcell", "y": "idycell", "z": "idzcell"}
    filtered = temp_tbl.copy()
    for axis, idx in fixed_idx.items():
        filtered = filtered.loc[filtered[idx_cols[axis]] == idx]
    if filtered.empty:
        raise RuntimeError(f"no temperature data found for fixed dimensions {fixed_idx} at step {args.step}")
    filtered = filtered.sort_values(idx_cols[free_axis])
    free_idx = filtered[idx_cols[free_axis]].to_numpy(dtype=np.int64)
    x_coord = centers_by_axis[free_axis][free_idx - 1]
    temperature = filtered["Temperature"].to_numpy(dtype=np.float64)
    output_path = build_output_path(run_dir, args.step, fixed_raw, args.output)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x_coord, temperature, marker="o", linewidth=1.5, markersize=4)
    fixed_desc = ", ".join(f"{axis}={fixed_idx[axis]}" for axis in sorted(fixed_idx))
    plt.title(f"Temperature line at step {args.step:05d} | fixed {fixed_desc}")
    plt.xlabel(f"{free_axis}-center (m)")
    plt.ylabel("Temperature (K)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(f"saved {output_path}")
    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
