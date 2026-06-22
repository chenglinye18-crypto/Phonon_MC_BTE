from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import phonon_mc as pm
from phonon_mc import init_mesh_from_geom, resolve_input_dir, setup_case_from_ldg_lgrid

IDX_COLS = {"x": "idxcell", "y": "idycell", "z": "idzcell"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export full temperature-field CSV from solver output. "
            "With one --step value, export that output step. "
            "With two --step values, average all available output steps in the inclusive range. "
            "Optionally extract and plot a 1D temperature line from the exported field."
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
    parser.add_argument(
        "--y-block",
        type=int,
        default=1,
        help=(
            "Merge this many consecutive y-cells within each (x, z) column before export. "
            "Example: 2 merges every two neighboring y-cells into one. Default: 1"
        ),
    )
    parser.add_argument(
        "--y-average",
        choices=("energy", "temperature"),
        default="energy",
        help=(
            "How to combine merged y-cells. "
            "energy: volume-average energy density then invert E-T LUT; "
            "temperature: volume-weighted temperature average. Default: energy"
        ),
    )
    parser.add_argument("--line-axis", choices=("x", "y", "z"), default="", help="Optional free axis for 1D line extraction and plotting")
    parser.add_argument("--idxcell", type=int, default=None, help="Fixed x-cell index when extracting a line")
    parser.add_argument("--idycell", type=int, default=None, help="Fixed y-cell index when extracting a line")
    parser.add_argument("--idzcell", type=int, default=None, help="Fixed z-cell index when extracting a line")
    parser.add_argument("--input-dir", default="", help="Optional input directory containing ldg.txt and lgrid.txt")
    parser.add_argument(
        "--line-output-csv",
        default="",
        help="Output CSV path for the extracted line. Default: <run-dir>/temperature_step_*_along_*.csv",
    )
    parser.add_argument(
        "--plot-png",
        default="",
        help="Output PNG path for the extracted line. Default: <run-dir>/temperature_step_*_along_*.png",
    )
    parser.add_argument("--show", action="store_true", help="Also show the line plot interactively")
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


def build_output_csv(run_dir: Path, steps: list[int], output_csv: str, y_block: int) -> Path:
    if output_csv:
        path = Path(output_csv).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    suffix = "" if int(y_block) <= 1 else f"_yblock{int(y_block)}"
    if len(steps) == 1:
        path = run_dir / f"temperature_step_{steps[0]:05d}{suffix}.csv"
    else:
        path = run_dir / f"temperature_steps_{steps[0]:05d}_{steps[-1]:05d}_avg{suffix}.csv"
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


def resolve_case_input_dir(run_dir: Path, input_dir: str) -> Path:
    if input_dir:
        return Path(input_dir).expanduser().resolve()
    inputs_dir = run_dir / "inputs"
    solver_params = inputs_dir / "solver_params.toml"
    if solver_params.is_file():
        return inputs_dir
    return resolve_input_dir()


def load_case_context(run_dir: Path, input_dir: str) -> dict[str, object]:
    ldg_file, lgrid_file = resolve_case_files(run_dir, input_dir)
    cs = setup_case_from_ldg_lgrid(ldg_file, lgrid_file, length_scale=1e-6, input_length_unit="um", verbose=False)
    mesh = init_mesh_from_geom(cs)
    boxes = np.asarray(mesh["boxes"], dtype=np.float64)
    centers = np.asarray(mesh["centers"], dtype=np.float64)
    return {
        "cs": cs,
        "mesh": mesh,
        "boxes": boxes,
        "centers": centers,
    }


def build_energy_lut_map(case_input_dir: Path, ctx: dict[str, object]) -> dict[str, dict[str, object]]:
    mesh = ctx["mesh"]
    cs = ctx["cs"]
    opts = pm.mc_default_opts(case_input_dir)
    opts = pm.resolve_linearization_temperature(mesh, opts)
    materials = pm.resolve_case_materials(cs)
    lut_map: dict[str, dict[str, object]] = {}
    for entry in materials["list"]:
        key = str(entry["key"])
        spec = pm.build_spectral_grid(entry["mat"], opts)
        lut_map[key] = pm.build_E_T_lookup(spec, pm.et_lookup_cfg_from_opts(opts))
    return lut_map


def coarsen_y_temperature_field(
    field_tbl: pd.DataFrame,
    run_dir: Path,
    input_dir: str,
    y_block: int,
    y_average: str,
) -> pd.DataFrame:
    if int(y_block) <= 1:
        return field_tbl
    ctx = load_case_context(run_dir, input_dir)
    mesh = ctx["mesh"]
    boxes = ctx["boxes"]
    ix = field_tbl["idxcell"].to_numpy(dtype=np.int64)
    iy = field_tbl["idycell"].to_numpy(dtype=np.int64)
    iz = field_tbl["idzcell"].to_numpy(dtype=np.int64)
    cid = pm.sub2ind(mesh["Nx"], mesh["Ny"], mesh["Nz"], ix, iy, iz).astype(np.int64)
    grp = ((iy - 1) // int(y_block)).astype(np.int64)
    work = field_tbl.copy()
    work["cid"] = cid
    work["y_block_group"] = grp
    bx = boxes[cid - 1]
    work["y0_m_src"] = bx[:, 2]
    work["y1_m_src"] = bx[:, 3]
    if "cell_vol" in mesh:
        work["V_m3"] = np.asarray(mesh["cell_vol"], dtype=np.float64)[cid - 1]
    else:
        work["V_m3"] = (bx[:, 1] - bx[:, 0]) * (bx[:, 3] - bx[:, 2]) * (bx[:, 5] - bx[:, 4])
    mat_names = np.asarray(mesh.get("cell_material_name", []), dtype=object)
    if mat_names.size >= int(np.max(cid)):
        work["material_key"] = [pm.material_key(str(name)) for name in mat_names[cid - 1]]
    else:
        work["material_key"] = ""
    lut_map = build_energy_lut_map(resolve_case_input_dir(run_dir, input_dir), ctx) if y_average == "energy" else {}
    rows: list[dict[str, object]] = []
    group_cols = ["aggregation", "step_start", "step_end", "step_count", "steps_used", "idxcell", "idzcell", "y_block_group"]
    for _, grp_df in work.groupby(group_cols, sort=False):
        weights = grp_df["V_m3"].to_numpy(dtype=np.float64)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
            weights = np.ones(len(grp_df), dtype=np.float64)
        lut = None
        if y_average == "energy" and grp_df["material_key"].nunique() == 1:
            lut = lut_map.get(str(grp_df["material_key"].iloc[0]))
        if lut is not None:
            tvals = np.clip(grp_df["temperature_K"].to_numpy(dtype=np.float64), float(lut["T"][0]), float(lut["T"][-1]))
            uvals = np.asarray(lut["U_interp"](tvals), dtype=np.float64)
            uavg = float(np.sum(uvals * weights) / np.sum(weights))
            uavg = float(np.clip(uavg, float(lut["U_mono"][0]), float(lut["U_mono"][-1])))
            tgrp = float(lut["inv_interp"](uavg))
        else:
            tgrp = float(np.average(grp_df["temperature_K"].to_numpy(dtype=np.float64), weights=weights))
        y0_m = float(grp_df["y0_m_src"].min())
        y1_m = float(grp_df["y1_m_src"].max())
        rows.append(
            {
                "aggregation": grp_df["aggregation"].iat[0],
                "step_start": int(grp_df["step_start"].iat[0]),
                "step_end": int(grp_df["step_end"].iat[0]),
                "step_count": int(grp_df["step_count"].iat[0]),
                "steps_used": grp_df["steps_used"].iat[0],
                "idxcell": int(grp_df["idxcell"].iat[0]),
                "idycell": int(grp_df["idycell"].min()),
                "idzcell": int(grp_df["idzcell"].iat[0]),
                "idycell_first": int(grp_df["idycell"].min()),
                "idycell_last": int(grp_df["idycell"].max()),
                "group_size": int(len(grp_df)),
                "y0_m": y0_m,
                "y1_m": y1_m,
                "y_center_m": 0.5 * (y0_m + y1_m),
                "y0_um": y0_m * 1e6,
                "y1_um": y1_m * 1e6,
                "y_center_um": 0.5 * (y0_m + y1_m) * 1e6,
                "temperature_K": tgrp,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["idxcell", "idycell", "idzcell"], kind="stable").reset_index(drop=True)


def load_axis_geometry(run_dir: Path, input_dir: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    ldg_file, lgrid_file = resolve_case_files(run_dir, input_dir)
    cs = setup_case_from_ldg_lgrid(ldg_file, lgrid_file, length_scale=1e-6, input_length_unit="um", verbose=False)
    mesh = init_mesh_from_geom(cs)
    centers = {
        "x": np.asarray(mesh["xc"], dtype=np.float64),
        "y": np.asarray(mesh["yc"], dtype=np.float64),
        "z": np.asarray(mesh["zc"], dtype=np.float64),
    }
    edges = {
        "x": np.asarray(mesh["x_edges"], dtype=np.float64),
        "y": np.asarray(mesh["y_edges"], dtype=np.float64),
        "z": np.asarray(mesh["z_edges"], dtype=np.float64),
    }
    return centers, edges


def resolve_fixed_indices(args: argparse.Namespace, line_axis: str) -> dict[str, int]:
    fixed_idx: dict[str, int] = {}
    for axis, col in IDX_COLS.items():
        if axis == line_axis:
            continue
        value = getattr(args, col)
        if value is None:
            raise ValueError(f"--{col} is required when --line-axis={line_axis}")
        if int(value) < 1:
            raise ValueError(f"--{col} must be >= 1")
        fixed_idx[axis] = int(value)
    return fixed_idx


def build_line_outputs(
    run_dir: Path,
    steps: list[int],
    line_axis: str,
    fixed_idx: dict[str, int],
    output_csv: str,
    output_png: str,
    y_block: int,
) -> tuple[Path, Path]:
    fixed_tag = "_".join(f"{IDX_COLS[axis][:-4]}{fixed_idx[axis]}" for axis in sorted(fixed_idx))
    suffix = "" if int(y_block) <= 1 else f"_yblock{int(y_block)}"
    if len(steps) == 1:
        stem = f"temperature_step_{steps[0]:05d}_along_{line_axis}_{fixed_tag}{suffix}"
    else:
        stem = f"temperature_steps_{steps[0]:05d}_{steps[-1]:05d}_avg_along_{line_axis}_{fixed_tag}{suffix}"
    csv_path = Path(output_csv).expanduser().resolve() if output_csv else (run_dir / f"{stem}.csv")
    png_path = Path(output_png).expanduser().resolve() if output_png else (run_dir / f"{stem}.png")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    return csv_path, png_path


def extract_line_table(
    field_tbl: pd.DataFrame,
    line_axis: str,
    fixed_idx: dict[str, int],
    centers_by_axis: dict[str, np.ndarray],
    edges_by_axis: dict[str, np.ndarray],
) -> pd.DataFrame:
    filtered = field_tbl.copy()
    for axis, idx in fixed_idx.items():
        if axis == "y" and {"idycell_first", "idycell_last"}.issubset(filtered.columns):
            filtered = filtered.loc[(filtered["idycell_first"] <= idx) & (filtered["idycell_last"] >= idx)]
        else:
            filtered = filtered.loc[filtered[IDX_COLS[axis]] == idx]
    if filtered.empty:
        raise RuntimeError(f"no temperature data found for line selection axis={line_axis}, fixed={fixed_idx}")
    free_col = IDX_COLS[line_axis]
    filtered = filtered.sort_values(free_col, kind="stable").reset_index(drop=True)
    free_idx = filtered[free_col].to_numpy(dtype=np.int64)
    line_tbl = filtered.loc[:, ["aggregation", "step_start", "step_end", "step_count", "steps_used"]].copy()
    for axis, col in IDX_COLS.items():
        if axis == line_axis:
            line_tbl[col] = free_idx
        else:
            line_tbl[col] = int(fixed_idx[axis])
    if line_axis == "y" and {"y0_m", "y1_m"}.issubset(filtered.columns):
        line_tbl["y0_m"] = filtered["y0_m"].to_numpy(dtype=np.float64)
        line_tbl["y1_m"] = filtered["y1_m"].to_numpy(dtype=np.float64)
        line_tbl["y_center_m"] = filtered.get("y_center_m", 0.5 * (filtered["y0_m"] + filtered["y1_m"])).to_numpy(dtype=np.float64)
        line_tbl["y0_um"] = filtered.get("y0_um", filtered["y0_m"] * 1e6).to_numpy(dtype=np.float64)
        line_tbl["y1_um"] = filtered.get("y1_um", filtered["y1_m"] * 1e6).to_numpy(dtype=np.float64)
        line_tbl["y_center_um"] = filtered.get("y_center_um", 0.5 * (filtered["y0_m"] + filtered["y1_m"]) * 1e6).to_numpy(dtype=np.float64)
        if "idycell_first" in filtered.columns:
            line_tbl["idycell_first"] = filtered["idycell_first"].to_numpy(dtype=np.int64)
        if "idycell_last" in filtered.columns:
            line_tbl["idycell_last"] = filtered["idycell_last"].to_numpy(dtype=np.int64)
        if "group_size" in filtered.columns:
            line_tbl["group_size"] = filtered["group_size"].to_numpy(dtype=np.int64)
    else:
        axis_edges = edges_by_axis[line_axis]
        axis_centers = centers_by_axis[line_axis]
        line_tbl[f"{line_axis}0_m"] = axis_edges[free_idx - 1]
        line_tbl[f"{line_axis}1_m"] = axis_edges[free_idx]
        line_tbl[f"{line_axis}_center_m"] = axis_centers[free_idx - 1]
        line_tbl[f"{line_axis}0_um"] = line_tbl[f"{line_axis}0_m"] * 1e6
        line_tbl[f"{line_axis}1_um"] = line_tbl[f"{line_axis}1_m"] * 1e6
        line_tbl[f"{line_axis}_center_um"] = line_tbl[f"{line_axis}_center_m"] * 1e6
    if line_axis != "y" and {"y0_m", "y1_m"}.issubset(filtered.columns):
        line_tbl["selected_y0_m"] = filtered["y0_m"].to_numpy(dtype=np.float64)
        line_tbl["selected_y1_m"] = filtered["y1_m"].to_numpy(dtype=np.float64)
        line_tbl["selected_y_center_m"] = filtered.get("y_center_m", 0.5 * (filtered["y0_m"] + filtered["y1_m"])).to_numpy(dtype=np.float64)
        line_tbl["selected_y0_um"] = filtered.get("y0_um", filtered["y0_m"] * 1e6).to_numpy(dtype=np.float64)
        line_tbl["selected_y1_um"] = filtered.get("y1_um", filtered["y1_m"] * 1e6).to_numpy(dtype=np.float64)
        line_tbl["selected_y_center_um"] = filtered.get("y_center_um", 0.5 * (filtered["y0_m"] + filtered["y1_m"]) * 1e6).to_numpy(dtype=np.float64)
        if "idycell_first" in filtered.columns:
            line_tbl["selected_idycell_first"] = filtered["idycell_first"].to_numpy(dtype=np.int64)
        if "idycell_last" in filtered.columns:
            line_tbl["selected_idycell_last"] = filtered["idycell_last"].to_numpy(dtype=np.int64)
        if "group_size" in filtered.columns:
            line_tbl["selected_y_group_size"] = filtered["group_size"].to_numpy(dtype=np.int64)
    line_tbl["temperature_K"] = filtered["temperature_K"].to_numpy(dtype=np.float64)
    return line_tbl


def plot_line_table(line_tbl: pd.DataFrame, line_axis: str, fixed_idx: dict[str, int], output_png: Path, show: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for --line-axis plotting. Run this script in an environment with matplotlib installed."
        ) from exc
    x_coord_um = line_tbl[f"{line_axis}_center_um"].to_numpy(dtype=np.float64)
    temperature = line_tbl["temperature_K"].to_numpy(dtype=np.float64)
    fixed_desc = ", ".join(f"{IDX_COLS[axis]}={fixed_idx[axis]}" for axis in sorted(fixed_idx))
    step_start = int(line_tbl["step_start"].iat[0])
    step_end = int(line_tbl["step_end"].iat[0])
    step_count = int(line_tbl["step_count"].iat[0])
    plt.figure(figsize=(8, 4.5))
    plt.plot(x_coord_um, temperature, marker="o", linewidth=1.5, markersize=4)
    if step_count == 1:
        title = f"Temperature along {line_axis} at step {step_start:05d} | fixed {fixed_desc}"
    else:
        title = f"Temperature along {line_axis} mean over steps {step_start:05d}-{step_end:05d} | fixed {fixed_desc}"
    plt.title(title)
    plt.xlabel(f"{line_axis}-center (um)")
    plt.ylabel("Temperature (K)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    if show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if int(args.y_block) < 1:
        raise ValueError("--y-block must be >= 1")
    steps = resolve_steps(run_dir, args.step)
    output_csv = build_output_csv(run_dir, steps, args.output_csv, args.y_block)
    step_tables = [load_step_temperature(run_dir, step) for step in steps]
    out = aggregate_temperature_tables(step_tables, steps)
    out = coarsen_y_temperature_field(out, run_dir, args.input_dir, args.y_block, args.y_average)
    out.to_csv(output_csv, index=False)
    print(f"saved {output_csv}")
    if not args.line_axis:
        if args.line_output_csv or args.plot_png or args.show:
            raise ValueError("--line-axis is required when requesting line CSV or line plotting")
        return
    fixed_idx = resolve_fixed_indices(args, args.line_axis)
    centers_by_axis, edges_by_axis = load_axis_geometry(run_dir, args.input_dir)
    line_tbl = extract_line_table(out, args.line_axis, fixed_idx, centers_by_axis, edges_by_axis)
    line_csv, line_png = build_line_outputs(
        run_dir,
        steps,
        args.line_axis,
        fixed_idx,
        args.line_output_csv,
        args.plot_png,
        args.y_block,
    )
    line_tbl.to_csv(line_csv, index=False)
    print(f"saved {line_csv}")
    plot_line_table(line_tbl, args.line_axis, fixed_idx, line_png, args.show)
    print(f"saved {line_png}")


if __name__ == "__main__":
    main()
