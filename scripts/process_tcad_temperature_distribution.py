from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phonon_mc import parse_lgrid


CORE_BOUNDS_UM = {
    "x": (0.01, 0.02),
    "y": (0.01, 0.16),
    "z": (0.01, 0.08),
}


def clip(value: float, bounds: tuple[float, float]) -> float:
    return float(min(max(value, bounds[0]), bounds[1]))


def load_tcad_temperature_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, comment="#", sep=r"\s+", names=["x", "y", "z", "T"])
    x = np.sort(df["x"].unique())
    y = np.sort(df["y"].unique())
    z = np.sort(df["z"].unique())
    grid = (
        df.pivot_table(index="x", columns=["y", "z"], values="T")
        .reindex(index=x)
        .to_numpy(dtype=np.float64)
        .reshape(len(x), len(y), len(z))
    )
    return x, y, z, grid


def classify_cell(cx: float, cy: float, cz: float) -> str:
    if cx < CORE_BOUNDS_UM["x"][0]:
        return "X-"
    if cx > CORE_BOUNDS_UM["x"][1]:
        return "X+"
    if cy < CORE_BOUNDS_UM["y"][0]:
        return "Y-"
    if cy > CORE_BOUNDS_UM["y"][1]:
        return "Y+"
    if cz < CORE_BOUNDS_UM["z"][0]:
        return "Z-"
    if cz > CORE_BOUNDS_UM["z"][1]:
        return "Z+"
    return "CORE"


def build_temperature_field(input_dir: Path, temperature_path: Path) -> pd.DataFrame:
    grid = parse_lgrid(input_dir / "lgrid.txt", 1.0)
    x_edges = np.asarray(grid["x_edges"], dtype=np.float64)
    y_edges = np.asarray(grid["y_edges"], dtype=np.float64)
    z_edges = np.asarray(grid["z_edges"], dtype=np.float64)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    tx, ty, tz, tgrid = load_tcad_temperature_grid(temperature_path)
    interp_3d = RegularGridInterpolator((tx, ty, tz), tgrid, bounds_error=False, fill_value=None)
    interp_xmin = RegularGridInterpolator((ty, tz), tgrid[0, :, :], bounds_error=False, fill_value=None)
    interp_xmax = RegularGridInterpolator((ty, tz), tgrid[-1, :, :], bounds_error=False, fill_value=None)
    interp_ymin = RegularGridInterpolator((tx, tz), tgrid[:, 0, :], bounds_error=False, fill_value=None)
    interp_ymax = RegularGridInterpolator((tx, tz), tgrid[:, -1, :], bounds_error=False, fill_value=None)
    interp_zmin = RegularGridInterpolator((tx, ty), tgrid[:, :, 0], bounds_error=False, fill_value=None)
    interp_zmax = RegularGridInterpolator((tx, ty), tgrid[:, :, -1], bounds_error=False, fill_value=None)

    rows: list[dict[str, float | int | str]] = []
    for iz, cz in enumerate(z_centers, start=1):
        for iy, cy in enumerate(y_centers, start=1):
            for ix, cx in enumerate(x_centers, start=1):
                cell_type = classify_cell(float(cx), float(cy), float(cz))
                if cell_type == "CORE":
                    temp = float(interp_3d([[cx, cy, cz]])[0])
                elif cell_type == "X-":
                    yz = [[clip(cy, CORE_BOUNDS_UM["y"]), clip(cz, CORE_BOUNDS_UM["z"])]]
                    temp = float(interp_xmin(yz)[0])
                elif cell_type == "X+":
                    yz = [[clip(cy, CORE_BOUNDS_UM["y"]), clip(cz, CORE_BOUNDS_UM["z"])]]
                    temp = float(interp_xmax(yz)[0])
                elif cell_type == "Y-":
                    xz = [[clip(cx, CORE_BOUNDS_UM["x"]), clip(cz, CORE_BOUNDS_UM["z"])]]
                    temp = float(interp_ymin(xz)[0])
                elif cell_type == "Y+":
                    xz = [[clip(cx, CORE_BOUNDS_UM["x"]), clip(cz, CORE_BOUNDS_UM["z"])]]
                    temp = float(interp_ymax(xz)[0])
                elif cell_type == "Z-":
                    xy = [[clip(cx, CORE_BOUNDS_UM["x"]), clip(cy, CORE_BOUNDS_UM["y"])]]
                    temp = float(interp_zmin(xy)[0])
                else:
                    xy = [[clip(cx, CORE_BOUNDS_UM["x"]), clip(cy, CORE_BOUNDS_UM["y"])]]
                    temp = float(interp_zmax(xy)[0])
                rows.append(
                    {
                        "idxcell": ix,
                        "idycell": iy,
                        "idzcell": iz,
                        "x_center_um": float(cx),
                        "y_center_um": float(cy),
                        "z_center_um": float(cz),
                        "region_type": cell_type,
                        "temperature_K": temp,
                    }
                )
    return pd.DataFrame(rows)


def add_cell_volumes(field: pd.DataFrame, input_dir: Path) -> pd.DataFrame:
    grid = parse_lgrid(input_dir / "lgrid.txt", 1.0)
    dx = np.diff(np.asarray(grid["x_edges"], dtype=np.float64))
    dy = np.diff(np.asarray(grid["y_edges"], dtype=np.float64))
    dz = np.diff(np.asarray(grid["z_edges"], dtype=np.float64))
    out = field.copy()
    out["cell_volume_um3"] = (
        dx[out["idxcell"].to_numpy(dtype=np.int64) - 1]
        * dy[out["idycell"].to_numpy(dtype=np.int64) - 1]
        * dz[out["idzcell"].to_numpy(dtype=np.int64) - 1]
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TCAD temperature samples to MC cell temperatures.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--temperature-file",
        type=Path,
        default=None,
        help="Defaults to <input-dir>/Temperature_distribution.txt",
    )
    parser.add_argument("--update-reference", action="store_true", help="Also overwrite reference_temperature.txt")
    parser.add_argument(
        "--reference-mode",
        choices=("volume_mean", "midpoint", "copy_field"),
        default="volume_mean",
        help="How to build uniform Tref when --update-reference is used.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    temperature_path = (args.temperature_file or (input_dir / "Temperature_distribution.txt")).resolve()
    field = add_cell_volumes(build_temperature_field(input_dir, temperature_path), input_dir)

    initial_out = field[["idxcell", "idycell", "idzcell"]].copy()
    initial_out["Tinit"] = field["temperature_K"]
    initial_out.to_csv(input_dir / "initial_temperature.csv", index=False)

    tmin = float(field["temperature_K"].min())
    tmax = float(field["temperature_K"].max())
    tref_midpoint = 0.5 * (tmin + tmax)
    tref_volume_mean = float(np.average(field["temperature_K"], weights=field["cell_volume_um3"]))

    if args.update_reference:
        ref_out = field[["idxcell", "idycell", "idzcell"]].copy()
        if args.reference_mode == "copy_field":
            tref_value = None
            ref_out["Tref"] = field["temperature_K"]
        else:
            tref_value = tref_volume_mean if args.reference_mode == "volume_mean" else tref_midpoint
            ref_out["Tref"] = tref_value
        ref_out.to_csv(input_dir / "reference_temperature.txt", index=False)

    field.to_csv(input_dir / "processed_temperature_field.csv", index=False)

    core = field[field["region_type"] == "CORE"]["temperature_K"]
    shell = field[field["region_type"] != "CORE"]["temperature_K"]
    print(f"cells_total={len(field)}")
    print(f"cells_core={len(core)}")
    print(f"cells_shell={len(shell)}")
    print(f"T_core[min,mean,max]=[{core.min():.6f}, {core.mean():.6f}, {core.max():.6f}]")
    print(f"T_shell[min,mean,max]=[{shell.min():.6f}, {shell.mean():.6f}, {shell.max():.6f}]")
    print(f"T_all[min,max]=[{tmin:.6f}, {tmax:.6f}]")
    print(f"Tref_midpoint={tref_midpoint:.6f}")
    print(f"Tref_volume_mean={tref_volume_mean:.6f}")
    print(f"wrote={input_dir / 'initial_temperature.csv'}")
    if args.update_reference:
        print(f"wrote={input_dir / 'reference_temperature.txt'}")
        print(f"reference_mode={args.reference_mode}")
        if args.reference_mode != 'copy_field':
            print(f"Tref_uniform={tref_value:.6f}")
    print(f"wrote={input_dir / 'processed_temperature_field.csv'}")


if __name__ == "__main__":
    main()
