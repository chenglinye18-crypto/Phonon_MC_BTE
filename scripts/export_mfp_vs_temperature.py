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
    material_key,
    mc_default_opts,
    resolve_case_materials,
    scattering_rate_table_formula,
    setup_case_from_ldg_lgrid,
)
from scripts.plot_kappa_vs_temperature import paper_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export equilibrium mean-free-path versus temperature. The mode weight uses the "
            "equilibrium occupancy-spectrum convention "
            "DOS*n_BE*domega."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input directory containing ldg.txt, lgrid.txt, and solver_params.toml.",
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
        default="output/mfp_vs_temperature",
        help="Directory for exported CSV/PNG/PDF files.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="PNG DPI. Default: 220")
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(name).strip())
    return cleaned.strip("_") or "material"


def branch_display_names(spec: dict[str, object]) -> list[str]:
    raw = [str(name) for name in spec["branches"]]
    ta_count = 0
    named: list[str] = []
    for name in raw:
        key = name.upper().replace(" ", "")
        if "TA" in key:
            ta_count += 1
            named.append(f"TA{ta_count}")
        elif "LA" in key:
            named.append("LA")
        else:
            named.append(name)
    return named


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


def build_opts(input_dir: Path, grid_temperature: float) -> dict[str, object]:
    opts = mc_default_opts(input_dir)
    opts["T0"] = float(grid_temperature)
    return opts


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    v = np.asarray(values, dtype=np.float64)[order]
    w = np.asarray(weights, dtype=np.float64)[order]
    cdf = np.cumsum(w) / np.sum(w)
    idx = int(np.searchsorted(cdf, 0.5, side="left"))
    idx = min(max(idx, 0), v.size - 1)
    return float(v[idx])


def compute_mfp_table(material_entry: dict[str, object], spec: dict[str, object], temperatures: np.ndarray, opts: dict[str, object]) -> pd.DataFrame:
    branches = branch_display_names(spec)
    DOS = np.maximum(np.asarray(spec["DOS_w_b"], dtype=np.float64), 0.0)
    w = np.maximum(np.asarray(spec["w_mid"], dtype=np.float64), 0.0)
    dw = np.asarray(spec["dw"], dtype=np.float64)
    if dw.ndim == 1:
        dw = np.tile(dw.reshape(1, -1), (w.shape[0], 1))
    vg = np.maximum(np.asarray(spec["vg_w"], dtype=np.float64), 0.0)

    rows: list[dict[str, object]] = []
    for temperature in np.asarray(temperatures, dtype=np.float64):
        T = max(float(temperature), 1e-12)
        nbe = 1.0 / np.maximum(np.exp(np.minimum(HBAR * w / (K_B * T), 700.0)) - 1.0, REALMIN)
        weights = DOS * nbe * dw
        rate_total = np.maximum(scattering_rate_table_formula(spec, T, opts), 0.0)
        valid = (rate_total > 0.0) & (vg > 0.0) & (weights > 0.0) & np.isfinite(rate_total) & np.isfinite(vg)
        if not np.any(valid):
            continue
        mfp_nm = (vg[valid] / rate_total[valid]) * 1e9
        wgt = weights[valid]
        rows.append(
            {
                "material": str(material_entry["name"]),
                "temperature_K": T,
                "branch": "TOTAL",
                "mean_mfp_nm": weighted_mean(mfp_nm, wgt),
                "median_mfp_nm": weighted_median(mfp_nm, wgt),
            }
        )
        for ib, branch_name in enumerate(branches):
            b_valid = valid[ib]
            if not np.any(b_valid):
                continue
            b_mfp_nm = (vg[ib, b_valid] / rate_total[ib, b_valid]) * 1e9
            b_wgt = weights[ib, b_valid]
            rows.append(
                {
                    "material": str(material_entry["name"]),
                    "temperature_K": T,
                    "branch": branch_name,
                    "mean_mfp_nm": weighted_mean(b_mfp_nm, b_wgt),
                    "median_mfp_nm": weighted_median(b_mfp_nm, b_wgt),
                }
            )
    return pd.DataFrame(rows)


def plot_total_mfp(table: pd.DataFrame, output_png: Path, output_pdf: Path, dpi: int) -> None:
    paper_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    totals = table.loc[table["branch"].astype(str) == "TOTAL"].copy()
    materials = list(dict.fromkeys(totals["material"].astype(str).tolist()))
    palette = ["#0f4c81", "#b45309", "#9f1239", "#047857"]
    for i, material in enumerate(materials):
        sub = totals.loc[totals["material"].astype(str) == material].sort_values("temperature_K")
        color = palette[i % len(palette)]
        ax.plot(
            sub["temperature_K"].to_numpy(dtype=np.float64),
            sub["mean_mfp_nm"].to_numpy(dtype=np.float64),
            color=color,
            linewidth=2.2,
            label=f"{material} mean",
        )
        ax.plot(
            sub["temperature_K"].to_numpy(dtype=np.float64),
            sub["median_mfp_nm"].to_numpy(dtype=np.float64),
            color=color,
            linewidth=1.8,
            linestyle="--",
            label=f"{material} median",
        )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("MFP (nm)")
    ax.set_title("Equilibrium Mean Free Path vs Temperature")
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser()
    input_dir = input_dir.resolve() if input_dir.is_absolute() else (ROOT / input_dir).resolve()
    output_dir = Path(args.output_dir).expanduser()
    output_dir = output_dir.resolve() if output_dir.is_absolute() else (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if float(args.T_max) <= float(args.T_min):
        raise ValueError("T-max must be greater than T-min")
    if int(args.num_points) < 2:
        raise ValueError("num-points must be >= 2")

    temperatures = np.linspace(float(args.T_min), float(args.T_max), int(args.num_points), dtype=np.float64)
    opts = build_opts(input_dir, float(0.5 * (temperatures[0] + temperatures[-1])))
    material_entries = resolve_material_entries(input_dir, list(args.material))

    tables: list[pd.DataFrame] = []
    for material_entry in material_entries:
        spec = build_spectral_grid(dict(material_entry["mat"]), opts)
        tables.append(compute_mfp_table(material_entry, spec, temperatures, opts))

    table = pd.concat(tables, ignore_index=True)
    material_tag = "_".join(sanitize_name(entry["name"]) for entry in material_entries)
    tag = f"{material_tag}_T{temperatures[0]:.3f}K_{temperatures[-1]:.3f}K_{int(args.num_points)}pts"
    csv_path = output_dir / f"mfp_vs_temperature_{tag}.csv"
    png_path = output_dir / f"mfp_vs_temperature_{tag}.png"
    pdf_path = output_dir / f"mfp_vs_temperature_{tag}.pdf"
    table.to_csv(csv_path, index=False)
    plot_total_mfp(table, png_path, pdf_path, int(args.dpi))
    print(f"[ok] csv -> {csv_path}")
    print(f"[ok] png -> {png_path}")
    print(f"[ok] pdf -> {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
