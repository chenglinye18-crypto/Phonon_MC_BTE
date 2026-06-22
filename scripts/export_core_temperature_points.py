#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import phonon_mc as pm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export MC core temperature back onto the original TCAD point cloud. "
            "With one --step value, export that step. "
            "With two --step values, average all available output steps in the inclusive range. "
            "If --step is omitted, the latest output step is used."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--step",
        nargs="+",
        type=int,
        default=None,
        help="One step number, or two step numbers defining an inclusive averaging window over available output steps",
    )
    parser.add_argument(
        "--points-file",
        default=None,
        help="Point-cloud file like Temperature_distribution.txt; default is input-dir/Temperature_distribution.txt",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output txt path. Default: steps/step_XXXXX/Temperature_distribution_MC.txt "
            "for a single step, or run-dir/Temperature_distribution_MC_steps_XXXXX_YYYYY_avg.txt "
            "for an averaged range"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "interp"],
        default="direct",
        help="direct: boundary original + core cell centers; interp: original point cloud with interior interpolation",
    )
    parser.add_argument(
        "--core-y-block",
        type=int,
        default=1,
        help=(
            "Merge this many consecutive core cells along y within each (x, z) column before export. "
            "Example: 2 merges every two neighboring core y-cells into one. Default: 1"
        ),
    )
    parser.add_argument(
        "--core-y-average",
        choices=["energy", "temperature"],
        default="energy",
        help=(
            "How to combine merged y-cells. "
            "energy: volume-average energy density then invert E-T LUT; "
            "temperature: volume-weighted temperature average. Default: energy"
        ),
    )
    return parser.parse_args()


def read_step_temperature(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    need = ["idxcell", "idycell", "idzcell"]
    for key in need:
        if key not in cols:
            raise ValueError(f"{path} missing column {key}")
    tcol = None
    for cand in ("temperature", "t", "temp"):
        if cand in cols:
            tcol = cols[cand]
            break
    if tcol is None:
        raise ValueError(f"{path} missing temperature column")
    out = pd.DataFrame(
        {
            "idxcell": df[cols["idxcell"]].astype(int),
            "idycell": df[cols["idycell"]].astype(int),
            "idzcell": df[cols["idzcell"]].astype(int),
            "T": df[tcol].astype(float),
        }
    )
    return out.sort_values(["idxcell", "idycell", "idzcell"], kind="stable").reset_index(drop=True)


def read_point_cloud(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", sep=r"\s+", names=["x", "y", "z", "T"], engine="python")


def list_output_steps(run_dir: Path) -> list[int]:
    steps: list[int] = []
    for step_dir in sorted((run_dir / "steps").glob("step_*")):
        match = re.fullmatch(r"step_(\d+)", step_dir.name)
        if match is not None:
            steps.append(int(match.group(1)))
    if not steps:
        raise FileNotFoundError(f"no step directories found under {run_dir / 'steps'}")
    return steps


def resolve_steps(run_dir: Path, step_args: list[int] | None) -> list[int]:
    available = list_output_steps(run_dir)
    if not step_args:
        return [available[-1]]
    if len(step_args) not in (1, 2):
        raise ValueError("--step requires one value, or two values defining an averaging window")
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


def build_output_path(run_dir: Path, steps: list[int], output_arg: str | None, core_y_block: int = 1) -> Path:
    if output_arg:
        path = Path(output_arg).expanduser().resolve()
    else:
        suffix = "" if int(core_y_block) <= 1 else f"_yblock{int(core_y_block)}"
        if len(steps) == 1:
            path = run_dir / "steps" / f"step_{steps[0]:05d}" / f"Temperature_distribution_MC{suffix}.txt"
        else:
            path = run_dir / f"Temperature_distribution_MC_steps_{steps[0]:05d}_{steps[-1]:05d}_avg{suffix}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def aggregate_temperature_tables(run_dir: Path, steps: list[int]) -> pd.DataFrame:
    step_tables = [read_step_temperature(run_dir / "steps" / f"step_{step:05d}" / "temperature.txt") for step in steps]
    base = step_tables[0].copy()
    ref_idx = base.loc[:, ["idxcell", "idycell", "idzcell"]].to_numpy(dtype=np.int64)
    temp_sum = base["T"].to_numpy(dtype=np.float64).copy()
    for table, step in zip(step_tables[1:], steps[1:]):
        idx = table.loc[:, ["idxcell", "idycell", "idzcell"]].to_numpy(dtype=np.int64)
        if idx.shape != ref_idx.shape or not np.array_equal(idx, ref_idx):
            raise RuntimeError(f"inconsistent temperature field indexing at step {step}")
        temp_sum += table["T"].to_numpy(dtype=np.float64)
    base["T"] = temp_sum / float(len(step_tables))
    return base


def load_case_context(input_dir: Path) -> dict[str, object]:
    cs = pm.setup_case_from_ldg_lgrid(
        input_dir / "ldg.txt",
        input_dir / "lgrid.txt",
        length_scale=1e-6,
        input_length_unit="um",
        verbose=False,
    )
    mesh = pm.init_mesh_from_geom(cs)
    boxes = np.asarray(mesh["boxes"], dtype=np.float64)
    centers = np.asarray(mesh["centers"], dtype=np.float64)
    reservoir_mask = np.asarray(mesh.get("reservoir_cell_mask", np.zeros(mesh["Nc"], dtype=bool)), dtype=bool)
    core_boxes = boxes[~reservoir_mask]
    meta = {
        "core_x_min": float(np.min(core_boxes[:, 0])),
        "core_x_max": float(np.max(core_boxes[:, 1])),
        "core_y_min": float(np.min(core_boxes[:, 2])),
        "core_y_max": float(np.max(core_boxes[:, 3])),
        "core_z_min": float(np.min(core_boxes[:, 4])),
        "core_z_max": float(np.max(core_boxes[:, 5])),
    }
    return {
        "cs": cs,
        "mesh": mesh,
        "boxes": boxes,
        "centers": centers,
        "reservoir_mask": reservoir_mask,
        "meta": meta,
    }


def build_core_interpolator(core_samples: pd.DataFrame, meta: dict[str, float]) -> tuple[RegularGridInterpolator, dict[str, float]]:
    if core_samples.empty:
        raise ValueError("core region contains no cells")
    xg = np.unique(np.round(core_samples["x_m"].to_numpy(dtype=np.float64), 15))
    yg = np.unique(np.round(core_samples["y_m"].to_numpy(dtype=np.float64), 15))
    zg = np.unique(np.round(core_samples["z_m"].to_numpy(dtype=np.float64), 15))
    grid = np.full((xg.size, yg.size, zg.size), np.nan, dtype=np.float64)
    x_map = {float(v): i for i, v in enumerate(xg)}
    y_map = {float(v): i for i, v in enumerate(yg)}
    z_map = {float(v): i for i, v in enumerate(zg)}
    for row in core_samples.itertuples(index=False):
        ix = x_map[round(float(row.x_m), 15)]
        iy = y_map[round(float(row.y_m), 15)]
        iz = z_map[round(float(row.z_m), 15)]
        grid[ix, iy, iz] = float(row.T)
    if np.any(~np.isfinite(grid)):
        raise RuntimeError("coarsened core grid is incomplete; cannot build interpolator")
    interp = RegularGridInterpolator((xg, yg, zg), grid, bounds_error=False, fill_value=None)
    interp_meta = {
        "core_x_min": float(meta["core_x_min"]),
        "core_x_max": float(meta["core_x_max"]),
        "core_y_min": float(meta["core_y_min"]),
        "core_y_max": float(meta["core_y_max"]),
        "core_z_min": float(meta["core_z_min"]),
        "core_z_max": float(meta["core_z_max"]),
        "x_min": float(xg.min()),
        "x_max": float(xg.max()),
        "y_min": float(yg.min()),
        "y_max": float(yg.max()),
        "z_min": float(zg.min()),
        "z_max": float(zg.max()),
    }
    return interp, interp_meta


def build_energy_lut_map(input_dir: Path, ctx: dict[str, object]) -> dict[str, dict[str, object]]:
    mesh = ctx["mesh"]
    cs = ctx["cs"]
    opts = pm.mc_default_opts(input_dir)
    opts = pm.resolve_linearization_temperature(mesh, opts)
    materials = pm.resolve_case_materials(cs)
    lut_map: dict[str, dict[str, object]] = {}
    for entry in materials["list"]:
        key = str(entry["key"])
        spec = pm.build_spectral_grid(entry["mat"], opts)
        lut_map[key] = pm.build_E_T_lookup(spec, pm.et_lookup_cfg_from_opts(opts))
    return lut_map


def build_core_sample_table(
    input_dir: Path,
    temp_df: pd.DataFrame,
    core_y_block: int,
    core_y_average: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    ctx = load_case_context(input_dir)
    mesh = ctx["mesh"]
    centers = ctx["centers"]
    boxes = ctx["boxes"]
    reservoir_mask = ctx["reservoir_mask"]
    ix = temp_df["idxcell"].to_numpy(dtype=np.int64)
    iy = temp_df["idycell"].to_numpy(dtype=np.int64)
    iz = temp_df["idzcell"].to_numpy(dtype=np.int64)
    cid = pm.sub2ind(mesh["Nx"], mesh["Ny"], mesh["Nz"], ix, iy, iz).astype(np.int64)
    keep = ~reservoir_mask[cid - 1]
    core = temp_df.loc[keep, ["idxcell", "idycell", "idzcell", "T"]].copy().reset_index(drop=True)
    cid_core = cid[keep]
    ctr = centers[cid_core - 1]
    bx = boxes[cid_core - 1]
    core["cid"] = cid_core
    core["x_m"] = ctr[:, 0]
    core["y_m"] = ctr[:, 1]
    core["z_m"] = ctr[:, 2]
    core["x0_m"] = bx[:, 0]
    core["x1_m"] = bx[:, 1]
    core["y0_m"] = bx[:, 2]
    core["y1_m"] = bx[:, 3]
    core["z0_m"] = bx[:, 4]
    core["z1_m"] = bx[:, 5]
    if "cell_vol" in mesh:
        core["V_m3"] = np.asarray(mesh["cell_vol"], dtype=np.float64)[cid_core - 1]
    else:
        core["V_m3"] = (core["x1_m"] - core["x0_m"]) * (core["y1_m"] - core["y0_m"]) * (core["z1_m"] - core["z0_m"])
    mat_names = np.asarray(mesh.get("cell_material_name", []), dtype=object)
    if mat_names.size >= int(np.max(cid_core)):
        core["material_key"] = [pm.material_key(str(name)) for name in mat_names[cid_core - 1]]
    else:
        core["material_key"] = ""
    if int(core_y_block) <= 1:
        samples = core.sort_values(["idxcell", "idycell", "idzcell"], kind="stable").reset_index(drop=True)
        return samples, ctx
    core = core.sort_values(["idxcell", "idzcell", "idycell"], kind="stable").reset_index(drop=True)
    core["y_block_group"] = core.groupby(["idxcell", "idzcell"], sort=False).cumcount() // int(core_y_block)
    lut_map = build_energy_lut_map(input_dir, ctx) if core_y_average == "energy" else {}
    rows: list[dict[str, object]] = []
    for _, grp in core.groupby(["idxcell", "idzcell", "y_block_group"], sort=False):
        weights = grp["V_m3"].to_numpy(dtype=np.float64)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
            weights = np.ones(len(grp), dtype=np.float64)
        if core_y_average == "energy" and grp["material_key"].nunique() == 1:
            mat_key = str(grp["material_key"].iloc[0])
            lut = lut_map.get(mat_key)
        else:
            lut = None
        if lut is not None:
            tvals = np.clip(grp["T"].to_numpy(dtype=np.float64), float(lut["T"][0]), float(lut["T"][-1]))
            uvals = np.asarray(lut["U_interp"](tvals), dtype=np.float64)
            uavg = float(np.sum(uvals * weights) / np.sum(weights))
            uavg = float(np.clip(uavg, float(lut["U_mono"][0]), float(lut["U_mono"][-1])))
            Tgrp = float(lut["inv_interp"](uavg))
        else:
            Tgrp = float(np.average(grp["T"].to_numpy(dtype=np.float64), weights=weights))
        y0 = float(grp["y0_m"].min())
        y1 = float(grp["y1_m"].max())
        rows.append(
            {
                "idxcell": int(grp["idxcell"].iloc[0]),
                "idycell": int(grp["idycell"].iloc[0]),
                "idzcell": int(grp["idzcell"].iloc[0]),
                "T": Tgrp,
                "cid": int(grp["cid"].iloc[0]),
                "x_m": float(grp["x_m"].iloc[0]),
                "y_m": 0.5 * (y0 + y1),
                "z_m": float(grp["z_m"].iloc[0]),
                "x0_m": float(grp["x0_m"].min()),
                "x1_m": float(grp["x1_m"].max()),
                "y0_m": y0,
                "y1_m": y1,
                "z0_m": float(grp["z0_m"].min()),
                "z1_m": float(grp["z1_m"].max()),
                "V_m3": float(grp["V_m3"].sum()),
                "material_key": str(grp["material_key"].iloc[0]) if grp["material_key"].nunique() == 1 else "MIXED",
                "group_size": int(len(grp)),
                "idycell_first": int(grp["idycell"].min()),
                "idycell_last": int(grp["idycell"].max()),
            }
        )
    samples = pd.DataFrame(rows).sort_values(["idxcell", "idzcell", "y_m"], kind="stable").reset_index(drop=True)
    return samples, ctx


def boundary_mask_from_points(pts_df: pd.DataFrame, meta: dict[str, float]) -> np.ndarray:
    x_um = pts_df["x"].to_numpy(dtype=np.float64)
    y_um = pts_df["y"].to_numpy(dtype=np.float64)
    z_um = pts_df["z"].to_numpy(dtype=np.float64)
    x_min_um = meta["core_x_min"] * 1e6
    x_max_um = meta["core_x_max"] * 1e6
    y_min_um = meta["core_y_min"] * 1e6
    y_max_um = meta["core_y_max"] * 1e6
    z_min_um = meta["core_z_min"] * 1e6
    z_max_um = meta["core_z_max"] * 1e6
    tol_um = 1e-9
    return (
        np.isclose(x_um, x_min_um, atol=tol_um, rtol=0.0)
        | np.isclose(x_um, x_max_um, atol=tol_um, rtol=0.0)
        | np.isclose(y_um, y_min_um, atol=tol_um, rtol=0.0)
        | np.isclose(y_um, y_max_um, atol=tol_um, rtol=0.0)
        | np.isclose(z_um, z_min_um, atol=tol_um, rtol=0.0)
        | np.isclose(z_um, z_max_um, atol=tol_um, rtol=0.0)
    )


def export_points_interpolated(
    input_dir: Path,
    temp_df: pd.DataFrame,
    points_file: Path,
    output_path: Path,
    core_y_block: int = 1,
    core_y_average: str = "energy",
) -> None:
    pts_df = read_point_cloud(points_file)
    core_samples, ctx = build_core_sample_table(input_dir, temp_df, core_y_block, core_y_average)
    interp, meta = build_core_interpolator(core_samples, ctx["meta"])
    q = pts_df[["x", "y", "z"]].to_numpy(dtype=np.float64) * 1e-6
    q[:, 0] = np.clip(q[:, 0], meta["x_min"], meta["x_max"])
    q[:, 1] = np.clip(q[:, 1], meta["y_min"], meta["y_max"])
    q[:, 2] = np.clip(q[:, 2], meta["z_min"], meta["z_max"])
    Tout = np.asarray(interp(q), dtype=np.float64)
    # Preserve the original TCAD temperatures on the six core boundary faces.
    # Only strict interior points are reconstructed from the MC cell-centered field.
    Tin = pts_df["T"].to_numpy(dtype=np.float64)
    boundary_mask = boundary_mask_from_points(pts_df, meta)
    Tout[boundary_mask] = Tin[boundary_mask]
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# X Y Z T_K\n")
        for (x, y, z), T in zip(pts_df[["x", "y", "z"]].to_numpy(dtype=np.float64), Tout):
            f.write(f"{x:.12g} {y:.12g} {z:.12g} {T:.12f}\n")


def export_points_direct(
    input_dir: Path,
    temp_df: pd.DataFrame,
    points_file: Path,
    output_path: Path,
    core_y_block: int = 1,
    core_y_average: str = "energy",
) -> None:
    core_samples, ctx = build_core_sample_table(input_dir, temp_df, core_y_block, core_y_average)
    meta = ctx["meta"]
    pts_df = read_point_cloud(points_file)
    boundary_mask = boundary_mask_from_points(pts_df, meta)
    boundary_df = pts_df.loc[boundary_mask, ["x", "y", "z", "T"]].copy()
    center_df = core_samples.loc[:, ["x_m", "y_m", "z_m", "T"]].copy()
    center_df["x"] = center_df["x_m"] * 1e6
    center_df["y"] = center_df["y_m"] * 1e6
    center_df["z"] = center_df["z_m"] * 1e6
    center_df = center_df.loc[:, ["x", "y", "z", "T"]]
    out_df = pd.concat([boundary_df, center_df], ignore_index=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# X Y Z T_K\n")
        for r in out_df.itertuples(index=False):
            f.write(f"{r.x:.12g} {r.y:.12g} {r.z:.12g} {r.T:.12f}\n")


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if int(args.core_y_block) < 1:
        raise ValueError("--core-y-block must be >= 1")

    steps = resolve_steps(run_dir, args.step)
    temp_df = aggregate_temperature_tables(run_dir, steps)
    points_file = Path(args.points_file).expanduser().resolve() if args.points_file else input_dir / "Temperature_distribution.txt"
    output_path = build_output_path(run_dir, steps, args.output, args.core_y_block)

    if args.mode == "interp":
        export_points_interpolated(
            input_dir,
            temp_df,
            points_file,
            output_path,
            core_y_block=args.core_y_block,
            core_y_average=args.core_y_average,
        )
    else:
        export_points_direct(
            input_dir,
            temp_df,
            points_file,
            output_path,
            core_y_block=args.core_y_block,
            core_y_average=args.core_y_average,
        )

    print(f"WROTE {output_path}")
    print(f"STEPS_USED {','.join(str(step) for step in steps)}")


if __name__ == "__main__":
    main()
