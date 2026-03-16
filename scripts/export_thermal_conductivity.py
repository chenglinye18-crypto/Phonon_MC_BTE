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

from phonon_mc import init_mesh_from_geom, resolve_input_dir, setup_case_from_ldg_lgrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-monitor thermal conductivity from solver output. "
            "k = q / (dT / dL), where dT is the temperature difference between the two "
            "cells adjacent to the monitor plane and dL is the distance between their centers."
        )
    )
    parser.add_argument("--run-dir", required=True, help="Run output directory, e.g. output/run_20260316_154407")
    parser.add_argument("--steps", nargs="+", required=True, help="Step numbers, or a single token 'all'")
    parser.add_argument("--output-csv", default="", help="Output CSV path. Default: <run-dir>/thermal_conductivity_*.csv")
    parser.add_argument("--input-dir", default="", help="Optional input directory containing ldg.txt and lgrid.txt")
    return parser.parse_args()


def resolve_steps(run_dir: Path, step_args: list[str]) -> list[int]:
    available: list[int] = []
    for step_dir in sorted((run_dir / "steps").glob("step_*")):
        match = re.fullmatch(r"step_(\d+)", step_dir.name)
        if match:
            available.append(int(match.group(1)))
    if not available:
        raise FileNotFoundError(f"no step directories found under {run_dir / 'steps'}")
    if len(step_args) == 1 and step_args[0].lower() == "all":
        return available
    requested = sorted({int(step) for step in step_args})
    missing = sorted(set(requested) - set(available))
    if missing:
        raise FileNotFoundError(f"requested steps not found: {missing}")
    return requested


def resolve_output_csv(run_dir: Path, steps: list[int], output_csv: str) -> Path:
    if output_csv:
        return Path(output_csv).expanduser().resolve()
    if len(steps) == 1:
        return run_dir / f"thermal_conductivity_step_{steps[0]:05d}.csv"
    return run_dir / f"thermal_conductivity_steps_{steps[0]:05d}_{steps[-1]:05d}.csv"


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


def load_monitor_length_scale(run_dir: Path) -> float:
    manifest = run_dir / "run_manifest.txt"
    if not manifest.is_file():
        return 1e-6
    table = pd.read_csv(manifest)
    row = table.loc[table["key"] == "monitor_length_scale"]
    if row.empty:
        return 1e-6
    try:
        return float(row["value"].iloc[0])
    except Exception:
        return 1e-6


def load_temperature_vector(mesh: dict, temperature_file: Path) -> np.ndarray:
    temp_tbl = pd.read_csv(temperature_file)
    Tcell = np.zeros(mesh["Nc"], dtype=np.float64)
    idx = temp_tbl["idxcell"].to_numpy(dtype=np.int64)
    idy = temp_tbl["idycell"].to_numpy(dtype=np.int64)
    idz = temp_tbl["idzcell"].to_numpy(dtype=np.int64)
    lin = idx + (idy - 1) * mesh["Nx"] + (idz - 1) * mesh["Nx"] * mesh["Ny"]
    Tcell[lin - 1] = temp_tbl["Temperature"].to_numpy(dtype=np.float64)
    return Tcell


def overlap_mask(a0: np.ndarray, a1: np.ndarray, b0: float, b1: float, tol: float) -> np.ndarray:
    return (a1 > b0 + tol) & ((a0 < b1 - tol) | (np.abs(a0 - b1) <= tol) | (np.abs(a1 - b0) <= tol))


def overlap_length(a0: np.ndarray, a1: np.ndarray, b0: float, b1: float) -> np.ndarray:
    return np.maximum(0.0, np.minimum(a1, b1) - np.maximum(a0, b0))


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return float("nan")
    if weights.size == 0 or np.all(weights <= 0):
        return float(np.mean(values))
    return float(np.sum(values * weights) / np.sum(weights))


def format_cell_ids(mesh: dict, cell_ids: np.ndarray) -> str:
    parts: list[str] = []
    for cid in np.asarray(cell_ids, dtype=np.int64):
        cid0 = cid - 1
        ix = cid0 % mesh["Nx"] + 1
        iy = (cid0 // mesh["Nx"]) % mesh["Ny"] + 1
        iz = cid0 // (mesh["Nx"] * mesh["Ny"]) + 1
        parts.append(f"{ix}:{iy}:{iz}")
    return ";".join(parts)


def monitor_side_cells(mesh: dict, mon: pd.Series, monitor_length_scale: float) -> dict[str, np.ndarray]:
    normal = str(mon["effective_normal"]).upper()
    bounds_input = np.array([mon["x0_in"], mon["x1_in"], mon["y0_in"], mon["y1_in"], mon["z0_in"], mon["z1_in"]], dtype=np.float64)
    bounds = bounds_input * monitor_length_scale
    boxes = np.asarray(mesh["boxes"], dtype=np.float64)
    centers = np.asarray(mesh["centers"], dtype=np.float64)
    tol = 1e-12 * max(1.0, float(np.max(np.abs(np.concatenate((boxes.reshape(-1), bounds.reshape(-1)))))))
    axis = normal[-1]
    if axis == "X":
        coord = 0.5 * (bounds[0] + bounds[1])
        minus_touch = np.abs(boxes[:, 1] - coord) <= tol
        plus_touch = np.abs(boxes[:, 0] - coord) <= tol
        tangential = overlap_mask(boxes[:, 2], boxes[:, 3], bounds[2], bounds[3], tol) & overlap_mask(boxes[:, 4], boxes[:, 5], bounds[4], bounds[5], tol)
        weights = overlap_length(boxes[:, 2], boxes[:, 3], bounds[2], bounds[3]) * overlap_length(boxes[:, 4], boxes[:, 5], bounds[4], bounds[5])
        minus_centers = centers[:, 0]
        plus_centers = centers[:, 0]
    elif axis == "Y":
        coord = 0.5 * (bounds[2] + bounds[3])
        minus_touch = np.abs(boxes[:, 3] - coord) <= tol
        plus_touch = np.abs(boxes[:, 2] - coord) <= tol
        tangential = overlap_mask(boxes[:, 0], boxes[:, 1], bounds[0], bounds[1], tol) & overlap_mask(boxes[:, 4], boxes[:, 5], bounds[4], bounds[5], tol)
        weights = overlap_length(boxes[:, 0], boxes[:, 1], bounds[0], bounds[1]) * overlap_length(boxes[:, 4], boxes[:, 5], bounds[4], bounds[5])
        minus_centers = centers[:, 1]
        plus_centers = centers[:, 1]
    elif axis == "Z":
        coord = 0.5 * (bounds[4] + bounds[5])
        minus_touch = np.abs(boxes[:, 5] - coord) <= tol
        plus_touch = np.abs(boxes[:, 4] - coord) <= tol
        tangential = overlap_mask(boxes[:, 0], boxes[:, 1], bounds[0], bounds[1], tol) & overlap_mask(boxes[:, 2], boxes[:, 3], bounds[2], bounds[3], tol)
        weights = overlap_length(boxes[:, 0], boxes[:, 1], bounds[0], bounds[1]) * overlap_length(boxes[:, 2], boxes[:, 3], bounds[2], bounds[3])
        minus_centers = centers[:, 2]
        plus_centers = centers[:, 2]
    else:
        raise ValueError(f"unsupported monitor normal: {normal}")
    minus_idx = np.flatnonzero(minus_touch & tangential)
    plus_idx = np.flatnonzero(plus_touch & tangential)
    if minus_idx.size == 0 or plus_idx.size == 0:
        raise RuntimeError(f"failed to locate cells adjacent to monitor {mon['label']}")
    return {
        "minus_cells": minus_idx + 1,
        "plus_cells": plus_idx + 1,
        "minus_weights": weights[minus_idx],
        "plus_weights": weights[plus_idx],
        "minus_centers": minus_centers[minus_idx],
        "plus_centers": plus_centers[plus_idx],
    }


def build_rows(run_dir: Path, mesh: dict, monitor_tbl: pd.DataFrame, steps: list[int], monitor_length_scale: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for step in steps:
        step_dir = run_dir / "steps" / f"step_{step:05d}"
        temp_file = step_dir / "temperature.txt"
        heat_file = step_dir / "heat_flux.txt"
        info_file = step_dir / "step_info.txt"
        if not temp_file.is_file() or not heat_file.is_file() or not info_file.is_file():
            raise FileNotFoundError(f"missing required step files under {step_dir}")
        Tcell = load_temperature_vector(mesh, temp_file)
        heat_tbl = pd.read_csv(heat_file)
        info = pd.read_csv(info_file).iloc[0]
        for _, mon in monitor_tbl.iterrows():
            heat_row = heat_tbl.loc[heat_tbl["label"] == mon["label"]]
            if heat_row.empty:
                continue
            heat = heat_row.iloc[0]
            side = monitor_side_cells(mesh, mon, monitor_length_scale)
            minus_cells = side["minus_cells"]
            plus_cells = side["plus_cells"]
            T_minus = weighted_average(Tcell[minus_cells - 1], side["minus_weights"])
            T_plus = weighted_average(Tcell[plus_cells - 1], side["plus_weights"])
            x_minus = weighted_average(side["minus_centers"], side["minus_weights"])
            x_plus = weighted_average(side["plus_centers"], side["plus_weights"])
            deltaL = float(x_plus - x_minus)
            deltaT = float(T_plus - T_minus)
            gradT = deltaT / deltaL if deltaL != 0 else np.nan
            q_interval = float(heat["flux_interval_W_m2"])
            q_cumulative = float(heat["flux_cumulative_W_m2"])
            k_interval = q_interval / gradT if np.isfinite(gradT) and gradT != 0 else np.nan
            k_cumulative = q_cumulative / gradT if np.isfinite(gradT) and gradT != 0 else np.nan
            rows.append(
                {
                    "row_type": "step",
                    "run_dir": str(run_dir.resolve()),
                    "step": step,
                    "step_tag": f"{step:05d}",
                    "steps_used": f"{step:05d}",
                    "label": mon["label"],
                    "requested_direction": mon["requested_direction"],
                    "effective_normal": mon["effective_normal"],
                    "area_m2": float(mon["area_m2"]),
                    "elapsed_time_s": float(info["elapsed_time_s"]),
                    "interval_time_s": float(info["interval_time_s"]),
                    "interval_energy_net_J": float(heat["interval_energy_net_J"]),
                    "cumulative_energy_net_J": float(heat["cumulative_energy_net_J"]),
                    "q_flux_interval_W_m2": q_interval,
                    "q_flux_cumulative_W_m2": q_cumulative,
                    "minus_cell_ids": format_cell_ids(mesh, minus_cells),
                    "plus_cell_ids": format_cell_ids(mesh, plus_cells),
                    "minus_cell_count": int(minus_cells.size),
                    "plus_cell_count": int(plus_cells.size),
                    "minus_center_m": x_minus,
                    "plus_center_m": x_plus,
                    "deltaL_m": deltaL,
                    "T_minus_K": T_minus,
                    "T_plus_K": T_plus,
                    "deltaT_K": deltaT,
                    "gradT_K_per_m": gradT,
                    "k_interval_W_m_K": k_interval,
                    "k_cumulative_W_m_K": k_cumulative,
                }
            )
    out = pd.DataFrame(rows)
    if len(steps) <= 1 or out.empty:
        return out
    summary_rows: list[dict[str, object]] = []
    steps_tag = ";".join(f"{step:05d}" for step in steps)
    for label, grp in out.groupby("label", sort=False):
        base = grp.iloc[0].to_dict()
        base.update(
            {
                "row_type": "mean",
                "step": np.nan,
                "step_tag": f"avg[{steps_tag}]",
                "steps_used": steps_tag,
                "elapsed_time_s": float(grp["elapsed_time_s"].mean()),
                "interval_time_s": float(grp["interval_time_s"].mean()),
                "interval_energy_net_J": float(grp["interval_energy_net_J"].mean()),
                "cumulative_energy_net_J": float(grp["cumulative_energy_net_J"].mean()),
                "q_flux_interval_W_m2": float(grp["q_flux_interval_W_m2"].mean()),
                "q_flux_cumulative_W_m2": float(grp["q_flux_cumulative_W_m2"].mean()),
                "T_minus_K": float(grp["T_minus_K"].mean()),
                "T_plus_K": float(grp["T_plus_K"].mean()),
                "deltaT_K": float(grp["deltaT_K"].mean()),
                "gradT_K_per_m": float(grp["gradT_K_per_m"].mean()),
                "k_interval_W_m_K": float(grp["k_interval_W_m_K"].mean()),
                "k_cumulative_W_m_K": float(grp["k_cumulative_W_m_K"].mean()),
                "k_interval_std_W_m_K": float(grp["k_interval_W_m_K"].std(ddof=0)),
                "k_cumulative_std_W_m_K": float(grp["k_cumulative_W_m_K"].std(ddof=0)),
                "sample_count": int(len(grp)),
            }
        )
        summary_rows.append(base)
    summary = pd.DataFrame(summary_rows)
    out["k_interval_std_W_m_K"] = np.nan
    out["k_cumulative_std_W_m_K"] = np.nan
    out["sample_count"] = 1
    return pd.concat([out, summary], ignore_index=True)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    steps = resolve_steps(run_dir, args.steps)
    output_csv = resolve_output_csv(run_dir, steps, args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ldg_file, lgrid_file = resolve_case_files(run_dir, args.input_dir)
    cs = setup_case_from_ldg_lgrid(ldg_file, lgrid_file, length_scale=1e-6, input_length_unit="um", verbose=False)
    mesh = init_mesh_from_geom(cs)
    monitor_tbl = pd.read_csv(run_dir / "heat_flux_monitors_manifest.txt")
    monitor_length_scale = load_monitor_length_scale(run_dir)
    out = build_rows(run_dir, mesh, monitor_tbl, steps, monitor_length_scale)
    out.to_csv(output_csv, index=False)
    print(f"wrote {len(out)} rows to {output_csv}")


if __name__ == "__main__":
    main()
