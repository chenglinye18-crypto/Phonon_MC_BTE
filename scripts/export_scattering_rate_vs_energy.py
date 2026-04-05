from __future__ import annotations

import argparse
import math
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
    build_spectral_grid,
    load_material,
    material_key,
    mc_default_opts,
    q_vabs_from_w_table,
    resolve_case_materials,
    resolve_input_dir,
    setup_case_from_ldg_lgrid,
)

E_CHARGE = 1.602176634e-19


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export branch-wise scattering-rate vs energy plots from the current input case. "
            "The supplied temperature is used both as T0 for the spectral grid and as the "
            "scattering-rate evaluation temperature. If multiple temperatures are supplied, "
            "the script also exports a multi-temperature total-scattering-rate vs energy plot."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="",
        help="Input directory containing ldg.txt, lgrid.txt, and solver_params.toml. Default: repo input/",
    )
    parser.add_argument(
        "--temperature",
        "--T0",
        dest="temperature",
        type=float,
        nargs="+",
        required=True,
        help=(
            "One or more temperatures in K used for both spectral-grid construction and "
            "scattering-rate evaluation."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for exported PNG/CSV files. Default: output/scattering_rate_vs_energy",
    )
    parser.add_argument(
        "--material",
        nargs="*",
        default=[],
        help="Optional material filter, e.g. IGZO SI. By default all materials used in the case are exported.",
    )
    parser.add_argument(
        "--cos-beta",
        type=float,
        default=1.0,
        help=(
            "Cosine term used in the thin-film boundary scattering factor when PB_Tsi > 0. "
            "Default 1.0, i.e. velocity parallel to the transport direction."
        ),
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI. Default: 180")
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


def resolve_output_dir(output_dir: str) -> Path:
    if output_dir:
        path = Path(output_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = ROOT / "output" / "scattering_rate_vs_energy"
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def build_opts(input_dir: Path, temperature: float) -> dict[str, object]:
    opts = mc_default_opts(input_dir)
    opts["T0"] = float(temperature)
    return opts


def branch_rates(spec: dict[str, object], opts: dict[str, object], temperature: float, cos_beta: float) -> dict[str, np.ndarray]:
    w = np.maximum(np.asarray(spec["w_mid"], dtype=np.float64), 0.0)
    B, Nw = w.shape
    q = np.zeros_like(w)
    vabs = np.zeros_like(w)
    for ib in range(B):
        b = np.full(Nw, ib + 1, dtype=np.int64)
        q[ib], vabs[ib] = q_vabs_from_w_table(spec, w[ib], b)
    T = max(float(temperature), 1e-6)
    x = HBAR * w / (K_B * T)
    sinh_x = np.sinh(np.minimum(x, 10000.0))
    zeros = np.zeros_like(w)

    rate_la = zeros.copy()
    rate_tan = zeros.copy()
    rate_tau = zeros.copy()
    rate_loto = zeros.copy()
    rate_imp = zeros.copy()
    rate_pb = zeros.copy()

    la_mask = np.asarray(spec["branch_is_la"], dtype=bool).reshape(-1, 1)
    ta_mask = np.asarray(spec["branch_is_ta"], dtype=bool).reshape(-1, 1)
    loto_mask = np.asarray(spec["branch_is_loto"], dtype=bool).reshape(-1, 1)

    if np.any(la_mask):
        rate_la = np.where(la_mask, float(opts["BL"]) * w**2 * T**3, 0.0)
    if np.any(ta_mask):
        rate_tan = np.where(ta_mask, float(opts["BTN"]) * w * T**4, 0.0)
        umask = ta_mask & (w > float(spec["omega_cut_ta"]))
        rate_tau[umask] = float(opts["BTU"]) * w[umask] ** 2 / np.maximum(sinh_x[umask], 1e-12)
    if np.any(loto_mask):
        rate_loto = np.where(loto_mask, 1.0 / (float(opts["tau_LTO_ps"]) * 1e-12), 0.0)
    # Script-side impurity model only: tau_PI^-1 = A*omega^4 + B*omega^2 + C.
    rate_imp = float(opts.get("A_imp", 0.0)) * w**4
    rate_imp = rate_imp + float(opts.get("B_imp", 0.0)) * w**2
    rate_imp = rate_imp + float(opts.get("C_imp", 0.0))

    Tsi = float(opts["PB_Tsi"])
    if Tsi > 0.0:
        delta = float(opts["PB_Delta"])
        cos2 = float(cos_beta) ** 2
        p_spec = np.exp(-4.0 * (np.maximum(q, 0.0) * delta) ** 2 * cos2)
        ff = (1.0 - p_spec) / (1.0 + p_spec)
        rate_pb = vabs / max(Tsi, 1e-12) * ff
    else:
        bulk_L = float(opts["PB_bulk_L"])
        bulk_F = float(opts["PB_bulk_F"])
        if bulk_L > 0.0:
            rate_pb = vabs / max(bulk_L * bulk_F, 1e-12)

    rate_total = rate_la + rate_tan + rate_tau + rate_loto + rate_imp + rate_pb
    energy_eV = HBAR * w / E_CHARGE
    return {
        "energy_eV": energy_eV,
        "omega_rad_s": w,
        "q_1_m": q,
        "vabs_m_s": vabs,
        "rate_total_s_inv": rate_total,
        "rate_la_s_inv": rate_la,
        "rate_tan_s_inv": rate_tan,
        "rate_tau_s_inv": rate_tau,
        "rate_loto_s_inv": rate_loto,
        "rate_imp_s_inv": rate_imp,
        "rate_pb_s_inv": rate_pb,
    }


def rates_to_table(material_entry: dict[str, object], spec: dict[str, object], rates: dict[str, np.ndarray], temperature: float, cos_beta: float) -> pd.DataFrame:
    branches = np.asarray(branch_display_names(spec), dtype=object)
    n_branch, n_bin = rates["rate_total_s_inv"].shape
    return pd.DataFrame(
        {
            "material": np.repeat(str(material_entry["name"]), n_branch * n_bin),
            "branch": np.repeat(branches, n_bin),
            "temperature_K": np.full(n_branch * n_bin, float(temperature)),
            "energy_eV": np.asarray(rates["energy_eV"], dtype=np.float64).reshape(-1),
            "scattering_rate_s_inv": np.asarray(rates["rate_total_s_inv"], dtype=np.float64).reshape(-1),
        }
    )


def collapsed_total_rates_table(
    material_entry: dict[str, object],
    spec: dict[str, object],
    rates: dict[str, np.ndarray],
    temperature: float,
) -> pd.DataFrame:
    branches = branch_display_names(spec)
    energy = np.asarray(rates["energy_eV"], dtype=np.float64)
    total = np.asarray(rates["rate_total_s_inv"], dtype=np.float64)
    rows: list[pd.DataFrame] = []
    ta_idx = [i for i, name in enumerate(branches) if name.startswith("TA")]
    if ta_idx:
        rows.append(
            pd.DataFrame(
                {
                    "material": str(material_entry["name"]),
                    "branch": "TA",
                    "temperature_K": float(temperature),
                    "energy_eV": energy[ta_idx[0]],
                    "scattering_rate_s_inv": np.mean(total[ta_idx], axis=0),
                }
            )
        )
    for i, name in enumerate(branches):
        if name == "LA":
            rows.append(
                pd.DataFrame(
                    {
                        "material": str(material_entry["name"]),
                        "branch": "LA",
                        "temperature_K": float(temperature),
                        "energy_eV": energy[i],
                        "scattering_rate_s_inv": total[i],
                    }
                )
            )
    if not rows:
        return pd.DataFrame(columns=["material", "branch", "temperature_K", "energy_eV", "scattering_rate_s_inv"])
    return pd.concat(rows, ignore_index=True)


def compute_thermal_conductivity(material_entry: dict[str, object], spec: dict[str, object], rates: dict[str, np.ndarray], temperature: float) -> pd.DataFrame:
    w = np.maximum(np.asarray(spec["w_mid"], dtype=np.float64), 0.0)
    dos = np.maximum(np.asarray(spec["DOS_w_b"], dtype=np.float64), 0.0)
    v = np.maximum(np.asarray(rates["vabs_m_s"], dtype=np.float64), 0.0)
    dw = np.asarray(spec["dw"], dtype=np.float64)
    if dw.ndim == 1:
        dw = dw.reshape(1, -1)
    T = max(float(temperature), 1e-12)
    x = HBAR * w / (K_B * T)
    ex = np.exp(np.minimum(x, 10000.0))
    nbe = 1.0 / np.maximum(ex - 1.0, np.finfo(np.float64).tiny)
    dndT = (HBAR * w / (K_B * T * T)) * nbe * (nbe + 1.0)
    cv_mode = HBAR * w * dndT
    rate_total = np.asarray(rates["rate_total_s_inv"], dtype=np.float64)
    tau = np.zeros_like(rate_total)
    valid = np.isfinite(rate_total) & (rate_total > 0.0)
    tau[valid] = 1.0 / rate_total[valid]
    kappa_density = (1.0 / 3.0) * dos * cv_mode * (v**2) * tau * dw
    branch_kappa = np.sum(kappa_density, axis=1)
    branches = list(spec["branches"])
    rows = [
        {
            "material": str(material_entry["name"]),
            "branch": str(branch_name),
            "thermal_conductivity_W_mK": float(branch_kappa[i]),
        }
        for i, branch_name in enumerate(branches)
    ]
    rows.append(
        {
            "material": str(material_entry["name"]),
            "branch": "TOTAL",
            "thermal_conductivity_W_mK": float(np.sum(branch_kappa)),
        }
    )
    return pd.DataFrame(rows)


def positive_or_nan(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.where(arr > 0.0, arr, np.nan)


def temperature_tag(temperatures: list[float]) -> str:
    parts = [f"{float(temp):.3f}K" for temp in temperatures]
    return "_".join(parts)


def plot_material_rates(material_entry: dict[str, object], spec: dict[str, object], rates: dict[str, np.ndarray], temperature: float, cos_beta: float, output_png: Path, dpi: int) -> None:
    branches = branch_display_names(spec)
    n_branch = len(branches)
    ncols = 2 if n_branch > 1 else 1
    nrows = int(math.ceil(n_branch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.4 * nrows), squeeze=False)
    colors = {
        "total": "#111111",
        "la": "#0f766e",
        "tan": "#1d4ed8",
        "tau": "#9333ea",
        "loto": "#c2410c",
        "imp": "#b91c1c",
        "pb": "#4b5563",
    }
    for ib, branch_name in enumerate(branches):
        ax = axes[ib // ncols][ib % ncols]
        x = rates["energy_eV"][ib]
        order = np.argsort(x, kind="stable")
        x = x[order]
        ax.plot(x, positive_or_nan(rates["rate_total_s_inv"][ib][order]), color=colors["total"], linewidth=2.0, label="total")
        ax.plot(x, positive_or_nan(rates["rate_la_s_inv"][ib][order]), color=colors["la"], linewidth=1.2, label="LA")
        ax.plot(x, positive_or_nan(rates["rate_tan_s_inv"][ib][order]), color=colors["tan"], linewidth=1.2, label="TA-N")
        ax.plot(x, positive_or_nan(rates["rate_tau_s_inv"][ib][order]), color=colors["tau"], linewidth=1.2, label="TA-U")
        ax.plot(x, positive_or_nan(rates["rate_loto_s_inv"][ib][order]), color=colors["loto"], linewidth=1.2, label="LO/TO")
        ax.plot(x, positive_or_nan(rates["rate_imp_s_inv"][ib][order]), color=colors["imp"], linewidth=1.2, label="impurity")
        ax.plot(x, positive_or_nan(rates["rate_pb_s_inv"][ib][order]), color=colors["pb"], linewidth=1.2, label="boundary")
        ax.set_yscale("log")
        ax.set_xlabel("E (eV)")
        ax.set_ylabel("Scattering Rate (s$^{-1}$)")
        ax.set_title(str(branch_name))
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.legend(fontsize=8, frameon=False)
    for idx in range(n_branch, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle(
        f"{material_entry['name']} | T = {temperature:.3f} K | cos(beta) = {cos_beta:.3f}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_material_total_rates_single_temperature(
    material_entry: dict[str, object],
    spec: dict[str, object],
    rates: dict[str, np.ndarray],
    temperature: float,
    output_png: Path,
    dpi: int,
) -> None:
    table = collapsed_total_rates_table(material_entry, spec, rates, temperature)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = {
        "LA": "#0f766e",
        "TA": "#1d4ed8",
    }
    for branch_name in ["TA", "LA"]:
        df = table[table["branch"] == branch_name]
        if df.empty:
            continue
        x = df["energy_eV"].to_numpy(dtype=np.float64)
        y = df["scattering_rate_s_inv"].to_numpy(dtype=np.float64)
        order = np.argsort(x, kind="stable")
        ax.plot(
            x[order],
            positive_or_nan(y[order]),
            linewidth=2.0,
            color=colors.get(branch_name, None),
            label=branch_name,
        )
    ax.set_yscale("log")
    ax.set_xlabel("E (eV)")
    ax.set_ylabel("Total Scattering Rate (s$^{-1}$)")
    ax.set_title(f"{material_entry['name']} | T = {temperature:.3f} K")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_material_total_rates_multi_temperature(
    material_entry: dict[str, object],
    curves: list[tuple[float, dict[str, object], dict[str, np.ndarray]]],
    output_png: Path,
    dpi: int,
) -> None:
    if not curves:
        return
    branches = ["TA", "LA"]
    n_branch = len(branches)
    ncols = 2 if n_branch > 1 else 1
    nrows = int(math.ceil(n_branch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.4 * nrows), squeeze=False)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0.10, 0.90, len(curves))]
    for ib, branch_name in enumerate(branches):
        ax = axes[ib // ncols][ib % ncols]
        for color, (temperature, _spec, rates) in zip(colors, curves):
            table = collapsed_total_rates_table(material_entry, _spec, rates, temperature)
            df = table[table["branch"] == branch_name]
            if df.empty:
                continue
            x = df["energy_eV"].to_numpy(dtype=np.float64)
            y = df["scattering_rate_s_inv"].to_numpy(dtype=np.float64)
            order = np.argsort(x, kind="stable")
            ax.plot(
                x[order],
                positive_or_nan(y[order]),
                color=color,
                linewidth=1.8,
                label=f"{float(temperature):.3f} K",
            )
        ax.set_yscale("log")
        ax.set_xlabel("E (eV)")
        ax.set_ylabel("Total Scattering Rate (s$^{-1}$)")
        ax.set_title(str(branch_name))
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.legend(fontsize=8, frameon=False)
    for idx in range(n_branch, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    temp_desc = ", ".join(f"{float(temp):.3f} K" for temp, _, _ in curves)
    fig.suptitle(
        f"{material_entry['name']} | Total Scattering Rate vs E | T = {temp_desc}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = resolve_input_dir(args.input_dir if args.input_dir else None)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    output_dir = resolve_output_dir(args.output_dir)
    temperatures = [float(temp) for temp in args.temperature]
    material_entries = resolve_material_entries(input_dir, list(args.material))

    manifest_rows = []
    multi_temp_manifest_rows = []
    for temperature in temperatures:
        kappa_csv_path = output_dir / f"thermal_conductivity_T{temperature:.3f}K.csv"
        if kappa_csv_path.exists():
            kappa_csv_path.unlink()
    for entry in material_entries:
        safe_name = sanitize_name(f"{entry['key']}_{entry['name']}")
        multi_temp_curves: list[tuple[float, dict[str, object], dict[str, np.ndarray]]] = []
        for temperature in temperatures:
            opts = build_opts(input_dir, temperature)
            mat = dict(entry["mat"])
            spec = build_spectral_grid(mat, opts)
            rates = branch_rates(spec, opts, temperature, args.cos_beta)
            csv_path = output_dir / f"scattering_rate_vs_E_{safe_name}_T{temperature:.3f}K.csv"
            png_path = output_dir / f"scattering_rate_vs_E_{safe_name}_T{temperature:.3f}K.png"
            total_csv_path = output_dir / f"total_scattering_rate_vs_E_{safe_name}_T{temperature:.3f}K.csv"
            total_png_path = output_dir / f"total_scattering_rate_vs_E_{safe_name}_T{temperature:.3f}K.png"
            kappa_csv_path = output_dir / f"thermal_conductivity_T{temperature:.3f}K.csv"
            table = rates_to_table(entry, spec, rates, temperature, args.cos_beta)
            table.to_csv(csv_path, index=False)
            collapsed_total_rates_table(entry, spec, rates, temperature).to_csv(total_csv_path, index=False)
            compute_thermal_conductivity(entry, spec, rates, temperature).to_csv(
                kappa_csv_path,
                mode="a" if kappa_csv_path.exists() else "w",
                header=not kappa_csv_path.exists(),
                index=False,
            )
            plot_material_rates(entry, spec, rates, temperature, args.cos_beta, png_path, int(args.dpi))
            plot_material_total_rates_single_temperature(entry, spec, rates, temperature, total_png_path, int(args.dpi))
            manifest_rows.append(
                {
                    "material_key": entry["key"],
                    "material_name": entry["name"],
                    "temperature_K": temperature,
                    "cos_beta": float(args.cos_beta),
                    "csv_path": str(csv_path.resolve()),
                    "png_path": str(png_path.resolve()),
                    "total_csv_path": str(total_csv_path.resolve()),
                    "total_png_path": str(total_png_path.resolve()),
                    "thermal_conductivity_csv_path": str(kappa_csv_path.resolve()),
                }
            )
            multi_temp_curves.append((temperature, spec, rates))
            print(f"[ok] {entry['name']} | T={temperature:.3f} K -> {png_path}")
        if len(multi_temp_curves) > 1:
            multi_tag = temperature_tag(temperatures)
            total_png_path = output_dir / f"total_scattering_rate_vs_E_{safe_name}_T{multi_tag}.png"
            plot_material_total_rates_multi_temperature(entry, multi_temp_curves, total_png_path, int(args.dpi))
            multi_temp_manifest_rows.append(
                {
                    "material_key": entry["key"],
                    "material_name": entry["name"],
                    "temperatures_K": ";".join(f"{temp:.3f}" for temp in temperatures),
                    "png_path": str(total_png_path.resolve()),
                }
            )
            print(f"[ok] {entry['name']} | multi-T total -> {total_png_path}")
    for temperature in temperatures:
        per_temp_rows = [row for row in manifest_rows if float(row["temperature_K"]) == float(temperature)]
        pd.DataFrame(per_temp_rows).to_csv(output_dir / f"manifest_T{temperature:.3f}K.csv", index=False)
    if multi_temp_manifest_rows:
        multi_tag = temperature_tag(temperatures)
        pd.DataFrame(multi_temp_manifest_rows).to_csv(
            output_dir / f"manifest_total_scattering_rate_multiT_{multi_tag}.csv",
            index=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
