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

from phonon_mc import (
    HBAR,
    K_B,
    REALMIN,
    build_spectral_grid,
    load_material,
    mat_from_phonon_dispersion_file,
    material_key,
    mc_default_opts,
    resolve_case_materials,
    setup_case_from_ldg_lgrid,
)
from scripts.plot_kappa_vs_temperature import paper_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export equilibrium heat capacity versus temperature using the same omega-integral "
            "convention as the solver E-T lookup. The script computes U(T) by integrating "
            "DOS*hbar*omega*n_BE over omega, then reports C(T)=dU/dT."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="input",
        help="Input directory containing ldg.txt, lgrid.txt, and solver_params.toml. Default: input",
    )
    parser.add_argument(
        "--dispersion-file",
        default="",
        help="Optional direct phonon-dispersion file. If provided, the script uses this file instead of case materials.",
    )
    parser.add_argument(
        "--material-name",
        default="TableDriven",
        help="Material label used when --dispersion-file is provided. Default: TableDriven",
    )
    parser.add_argument("--T-min", type=float, required=True, help="Minimum temperature in K.")
    parser.add_argument("--T-max", type=float, required=True, help="Maximum temperature in K.")
    parser.add_argument("--num-points", type=int, default=301, help="Number of temperatures. Default: 301")
    parser.add_argument(
        "--material",
        nargs="*",
        default=[],
        help="Optional material filter, e.g. IGZO SI. By default all materials used in the case are exported.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/heat_capacity_vs_temperature",
        help="Directory for exported CSV/PNG/PDF files.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="PNG DPI. Default: 220")
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(name).strip())
    return cleaned.strip("_") or "material"


def resolve_case_files(input_dir: Path) -> tuple[Path, Path]:
    ldg = input_dir / "ldg.txt"
    lgrid = input_dir / "lgrid.txt"
    if not ldg.is_file() or not lgrid.is_file():
        raise FileNotFoundError(f"missing ldg.txt or lgrid.txt under {input_dir}")
    return ldg, lgrid


def resolve_material_entries(input_dir: Path, material_filters: list[str]) -> list[dict[str, object]]:
    ldg_file, lgrid_file = resolve_case_files(input_dir)
    cs = setup_case_from_ldg_lgrid(ldg_file, lgrid_file, length_scale=1e-6, input_length_unit="um", verbose=False)
    entries = list(resolve_case_materials(cs)["list"])
    if not material_filters:
        return entries
    wanted = {material_key(name) for name in material_filters}
    selected = [entry for entry in entries if entry["key"] in wanted]
    missing = sorted(wanted - {entry["key"] for entry in selected})
    for key in missing:
        selected.append({"name": key, "key": key, "mat": load_material(key, key)})
    return selected


def resolve_direct_material_entry(dispersion_file: Path, material_name: str) -> list[dict[str, object]]:
    mat = mat_from_phonon_dispersion_file(file_path=dispersion_file, material_name=material_name)
    return [{"name": material_name, "key": material_key(material_name), "mat": mat}]


def build_opts(input_dir: Path, grid_temperature: float) -> dict[str, object]:
    opts = mc_default_opts(input_dir)
    opts["T0"] = float(grid_temperature)
    return opts


def compute_equilibrium_tables(spec: dict[str, object], temperatures: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T = np.asarray(temperatures, dtype=np.float64)
    DOS = np.maximum(np.asarray(spec["DOS_w_b"], dtype=np.float64), 0.0)
    w = np.maximum(np.asarray(spec["w_mid"], dtype=np.float64), 0.0)
    dw = np.asarray(spec["dw"], dtype=np.float64)
    if dw.ndim == 1:
        dw = np.tile(dw.reshape(1, -1), (w.shape[0], 1))
    x = (HBAR * w[np.newaxis, :, :]) / (K_B * T[:, np.newaxis, np.newaxis])
    nbe = 1.0 / np.maximum(np.exp(np.minimum(x, 700.0)) - 1.0, REALMIN)
    dU = DOS[np.newaxis, :, :] * (HBAR * w[np.newaxis, :, :]) * nbe * dw[np.newaxis, :, :]
    Ub = dU.sum(axis=2)
    U = Ub.sum(axis=1)
    dndT = (HBAR * w[np.newaxis, :, :]) / (K_B * T[:, np.newaxis, np.newaxis] ** 2) * nbe * (nbe + 1.0)
    dC = DOS[np.newaxis, :, :] * (HBAR * w[np.newaxis, :, :]) * dndT * dw[np.newaxis, :, :]
    Cb_direct = dC.sum(axis=2)
    C_direct = Cb_direct.sum(axis=1)
    return U, Ub, C_direct, Cb_direct


def compute_capacity_table(material_entry: dict[str, object], spec: dict[str, object], temperatures: np.ndarray) -> pd.DataFrame:
    branches = list(spec["branches"])
    T = np.asarray(temperatures, dtype=np.float64)
    U, Ub, C_direct, Cb_direct = compute_equilibrium_tables(spec, T)
    edge_order = 2 if T.size >= 3 else 1
    C_delta = np.gradient(U, T, edge_order=edge_order)
    Cb_delta = np.vstack([np.gradient(Ub[:, ib], T, edge_order=edge_order) for ib in range(Ub.shape[1])]).T

    rows: list[dict[str, object]] = []
    for i, temp_k in enumerate(T):
        for ib, branch_name in enumerate(branches):
            rows.append(
                {
                    "material": str(material_entry["name"]),
                    "temperature_K": float(temp_k),
                    "branch": str(branch_name),
                    "energy_density_J_m3": float(Ub[i, ib]),
                    "heat_capacity_deltaE_deltaT_J_m3K": float(Cb_delta[i, ib]),
                    "heat_capacity_direct_J_m3K": float(Cb_direct[i, ib]),
                }
            )
        rows.append(
            {
                "material": str(material_entry["name"]),
                "temperature_K": float(temp_k),
                "branch": "TOTAL",
                "energy_density_J_m3": float(U[i]),
                "heat_capacity_deltaE_deltaT_J_m3K": float(C_delta[i]),
                "heat_capacity_direct_J_m3K": float(C_direct[i]),
            }
        )
    return pd.DataFrame(rows)


def plot_total_heat_capacity(table: pd.DataFrame, output_png: Path, output_pdf: Path, dpi: int) -> None:
    paper_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    totals = table.loc[table["branch"].astype(str) == "TOTAL"].copy()
    materials = list(dict.fromkeys(totals["material"].astype(str).tolist()))
    colors = ["#0f4c81", "#b45309", "#9f1239", "#047857", "#7c3aed"]
    for i, material in enumerate(materials):
        sub = totals.loc[totals["material"].astype(str) == material].sort_values("temperature_K")
        ax.plot(
            sub["temperature_K"].to_numpy(dtype=np.float64),
            sub["heat_capacity_deltaE_deltaT_J_m3K"].to_numpy(dtype=np.float64),
            color=colors[i % len(colors)],
            linewidth=2.0,
            label=f"{material}  dE/dT",
        )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Heat Capacity, $C$ (J m$^{-3}$ K$^{-1}$)")
    ax.set_title("Equilibrium Heat Capacity vs Temperature")
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = (ROOT / args.input_dir).resolve() if not Path(args.input_dir).is_absolute() else Path(args.input_dir).resolve()
    dispersion_file = Path(args.dispersion_file).expanduser()
    if args.dispersion_file:
        dispersion_file = dispersion_file.resolve() if dispersion_file.is_absolute() else (ROOT / dispersion_file).resolve()
    output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if float(args.T_max) <= float(args.T_min):
        raise ValueError("T-max must be greater than T-min")
    if int(args.num_points) < 2:
        raise ValueError("num-points must be >= 2")

    temperatures = np.linspace(float(args.T_min), float(args.T_max), int(args.num_points), dtype=np.float64)
    grid_temperature = float(0.5 * (temperatures[0] + temperatures[-1]))
    opts = build_opts(input_dir, grid_temperature)
    if args.dispersion_file:
        material_entries = resolve_direct_material_entry(dispersion_file, str(args.material_name))
    else:
        material_entries = resolve_material_entries(input_dir, list(args.material))

    tables: list[pd.DataFrame] = []
    for material_entry in material_entries:
        spec = build_spectral_grid(dict(material_entry["mat"]), opts)
        tables.append(compute_capacity_table(material_entry, spec, temperatures))

    table = pd.concat(tables, ignore_index=True)
    material_tag = "_".join(sanitize_name(entry["name"]) for entry in material_entries)
    tag = f"{material_tag}_T{temperatures[0]:.3f}K_{temperatures[-1]:.3f}K_{int(args.num_points)}pts"
    csv_path = output_dir / f"heat_capacity_vs_temperature_{tag}.csv"
    png_path = output_dir / f"heat_capacity_vs_temperature_{tag}.png"
    pdf_path = output_dir / f"heat_capacity_vs_temperature_{tag}.pdf"
    table.to_csv(csv_path, index=False)
    plot_total_heat_capacity(table, png_path, pdf_path, int(args.dpi))
    print(f"[ok] csv -> {csv_path}")
    print(f"[ok] png -> {png_path}")
    print(f"[ok] pdf -> {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
