from __future__ import annotations

import argparse
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

from phonon_mc import HBAR, K_B, REALMIN, build_spectral_grid, mat_from_phonon_dispersion_file
from scripts.plot_kappa_vs_temperature import paper_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare DOS*nBE and hbar*omega*DOS*nBE weights between two phonon-dispersion files. "
            "The script exports raw and normalized weight curves."
        )
    )
    parser.add_argument(
        "--baseline-dispersion-file",
        required=True,
        help="Reference dispersion file, e.g. current phonon_dispersion_IGZO.txt",
    )
    parser.add_argument(
        "--compare-dispersion-file",
        required=True,
        help="Comparison dispersion file, e.g. phonon_q_omega_vg_really63.txt",
    )
    parser.add_argument("--baseline-label", default="IGZO", help="Legend label for the baseline dispersion.")
    parser.add_argument("--compare-label", default="really63", help="Legend label for the comparison dispersion.")
    parser.add_argument(
        "--temperature",
        type=float,
        nargs="+",
        default=[300.0, 323.0, 373.0],
        help="One or more temperatures in K. Default: 300 323 373",
    )
    parser.add_argument("--n-q", type=int, default=5000, help="Spectral-grid q count. Default: 5000")
    parser.add_argument("--n-w", type=int, default=1000, help="Spectral-grid omega-bin count. Default: 1000")
    parser.add_argument("--dpi", type=int, default=260, help="PNG DPI. Default: 260")
    parser.add_argument(
        "--output-dir",
        default="output/dos_nbe_weight_compare",
        help="Directory for exported CSV/PNG/PDF files.",
    )
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def build_spec_from_dispersion(file_path: Path, label: str, n_q: int, n_w: int, T0: float) -> dict[str, object]:
    mat = mat_from_phonon_dispersion_file(file_path=file_path, material_name=label)
    opts = {
        "T0": float(T0),
        "n_q": int(n_q),
        "n_w": int(n_w),
        "weight_by_Cv_for_Q": True,
    }
    return build_spectral_grid(mat, opts)


def compute_weights(spec: dict[str, object], temperature: float) -> pd.DataFrame:
    w = np.maximum(np.asarray(spec["w_mid"], dtype=np.float64), 0.0)
    dos = np.maximum(np.asarray(spec["DOS_w_b"], dtype=np.float64), 0.0).sum(axis=0)
    w1 = np.asarray(w[0], dtype=np.float64)
    dw = np.asarray(spec["dw"], dtype=np.float64).reshape(-1)
    x = HBAR * w1 / (K_B * max(float(temperature), 1e-12))
    nbe = 1.0 / np.maximum(np.exp(np.minimum(x, 700.0)) - 1.0, REALMIN)
    dos_nbe = dos * nbe
    energy_weight = (HBAR * w1) * dos_nbe
    norm_den_num = float(np.sum(dos_nbe * dw))
    norm_den_energy = float(np.sum(energy_weight * dw))
    dos_nbe_norm = dos_nbe / max(norm_den_num, REALMIN)
    energy_weight_norm = energy_weight / max(norm_den_energy, REALMIN)
    return pd.DataFrame(
        {
            "omega_rad_s": w1,
            "energy_eV": (HBAR * w1) / 1.602176634e-19,
            "dw_rad_s": dw,
            "dos_total": dos,
            "nBE": nbe,
            "dos_nbe": dos_nbe,
            "hbar_omega_dos_nbe": energy_weight,
            "dos_nbe_norm_density": dos_nbe_norm,
            "hbar_omega_dos_nbe_norm_density": energy_weight_norm,
        }
    )


def build_long_table(
    baseline_spec: dict[str, object],
    compare_spec: dict[str, object],
    temperatures: list[float],
    baseline_label: str,
    compare_label: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for label, spec in ((baseline_label, baseline_spec), (compare_label, compare_spec)):
        for temp in temperatures:
            table = compute_weights(spec, float(temp))
            table.insert(0, "source", str(label))
            table.insert(1, "temperature_K", float(temp))
            rows.append(table)
    return pd.concat(rows, ignore_index=True)


def plot_weight_grid(table: pd.DataFrame, source_order: list[str], output_png: Path, output_pdf: Path, dpi: int) -> None:
    paper_style()
    temperatures = sorted(table["temperature_K"].drop_duplicates().astype(float).tolist())
    fig, axes = plt.subplots(len(temperatures), 2, figsize=(11.0, 3.2 * len(temperatures)), constrained_layout=True)
    if len(temperatures) == 1:
        axes = np.asarray([axes], dtype=object)
    colors = {
        source_order[0]: "#0f4c81",
        source_order[1]: "#b45309",
    }
    for row_idx, temp in enumerate(temperatures):
        ax_l = axes[row_idx, 0]
        ax_r = axes[row_idx, 1]
        for source in source_order:
            sub = table.loc[(table["temperature_K"] == temp) & (table["source"] == source)].sort_values("energy_eV")
            x = sub["energy_eV"].to_numpy(dtype=np.float64)
            ax_l.plot(x, sub["dos_nbe_norm_density"].to_numpy(dtype=np.float64), color=colors[source], linewidth=2.0, label=source)
            ax_r.plot(x, sub["hbar_omega_dos_nbe_norm_density"].to_numpy(dtype=np.float64), color=colors[source], linewidth=2.0, label=source)
        ax_l.set_ylabel(f"{temp:.0f} K\nNormalized Density")
        ax_l.set_title(r"$DOS(\omega)\,n_{BE}$")
        ax_r.set_title(r"$\hbar\omega\,DOS(\omega)\,n_{BE}$")
        ax_l.grid(True, linestyle="--", alpha=0.25)
        ax_r.grid(True, linestyle="--", alpha=0.25)
        ax_l.set_yscale("log")
        ax_r.set_yscale("log")
        if row_idx == len(temperatures) - 1:
            ax_l.set_xlabel(r"Mode Energy, $\hbar\omega$ (eV)")
            ax_r.set_xlabel(r"Mode Energy, $\hbar\omega$ (eV)")
        if row_idx == 0:
            ax_l.legend(loc="best", frameon=False)
            ax_r.legend(loc="best", frameon=False)
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_raw_weight_grid(table: pd.DataFrame, source_order: list[str], output_png: Path, output_pdf: Path, dpi: int) -> None:
    paper_style()
    temperatures = sorted(table["temperature_K"].drop_duplicates().astype(float).tolist())
    fig, axes = plt.subplots(len(temperatures), 2, figsize=(11.0, 3.2 * len(temperatures)), constrained_layout=True)
    if len(temperatures) == 1:
        axes = np.asarray([axes], dtype=object)
    colors = {
        source_order[0]: "#0f4c81",
        source_order[1]: "#b45309",
    }
    for row_idx, temp in enumerate(temperatures):
        ax_l = axes[row_idx, 0]
        ax_r = axes[row_idx, 1]
        for source in source_order:
            sub = table.loc[(table["temperature_K"] == temp) & (table["source"] == source)].sort_values("energy_eV")
            x = sub["energy_eV"].to_numpy(dtype=np.float64)
            ax_l.plot(x, sub["dos_nbe"].to_numpy(dtype=np.float64), color=colors[source], linewidth=2.0, label=source)
            ax_r.plot(x, sub["hbar_omega_dos_nbe"].to_numpy(dtype=np.float64), color=colors[source], linewidth=2.0, label=source)
        ax_l.set_ylabel(f"{temp:.0f} K\nRaw Weight")
        ax_l.set_title(r"$DOS(\omega)\,n_{BE}$")
        ax_r.set_title(r"$\hbar\omega\,DOS(\omega)\,n_{BE}$")
        ax_l.grid(True, linestyle="--", alpha=0.25)
        ax_r.grid(True, linestyle="--", alpha=0.25)
        ax_l.set_yscale("log")
        ax_r.set_yscale("log")
        if row_idx == len(temperatures) - 1:
            ax_l.set_xlabel(r"Mode Energy, $\hbar\omega$ (eV)")
            ax_r.set_xlabel(r"Mode Energy, $\hbar\omega$ (eV)")
        if row_idx == 0:
            ax_l.legend(loc="best", frameon=False)
            ax_r.legend(loc="best", frameon=False)
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def build_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (source, temp), sub in table.groupby(["source", "temperature_K"], sort=True):
        omega = sub["omega_rad_s"].to_numpy(dtype=np.float64)
        dw = sub["dw_rad_s"].to_numpy(dtype=np.float64)
        dos_nbe = sub["dos_nbe"].to_numpy(dtype=np.float64)
        energy_weight = sub["hbar_omega_dos_nbe"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "source": str(source),
                "temperature_K": float(temp),
                "int_dos_nbe": float(np.sum(dos_nbe * dw)),
                "int_hbar_omega_dos_nbe": float(np.sum(energy_weight * dw)),
                "energy_peak_dos_nbe_eV": float(sub.loc[sub["dos_nbe"].idxmax(), "energy_eV"]),
                "energy_peak_hbar_omega_dos_nbe_eV": float(sub.loc[sub["hbar_omega_dos_nbe"].idxmax(), "energy_eV"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["temperature_K", "source"], kind="stable").reset_index(drop=True)


def main() -> int:
    args = parse_args()
    baseline_file = resolve_path(args.baseline_dispersion_file)
    compare_file = resolve_path(args.compare_dispersion_file)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temperatures = [float(v) for v in args.temperature]
    T0_ref = float(np.mean(temperatures))

    baseline_spec = build_spec_from_dispersion(baseline_file, str(args.baseline_label), int(args.n_q), int(args.n_w), T0_ref)
    compare_spec = build_spec_from_dispersion(compare_file, str(args.compare_label), int(args.n_q), int(args.n_w), T0_ref)
    table = build_long_table(baseline_spec, compare_spec, temperatures, str(args.baseline_label), str(args.compare_label))
    summary = build_summary(table)

    tag = f"{str(args.baseline_label)}_vs_{str(args.compare_label)}"
    csv_path = output_dir / f"dos_nbe_weight_compare_{tag}.csv"
    summary_path = output_dir / f"dos_nbe_weight_compare_{tag}_summary.csv"
    norm_png = output_dir / f"dos_nbe_weight_compare_{tag}_normalized.png"
    norm_pdf = output_dir / f"dos_nbe_weight_compare_{tag}_normalized.pdf"
    raw_png = output_dir / f"dos_nbe_weight_compare_{tag}_raw.png"
    raw_pdf = output_dir / f"dos_nbe_weight_compare_{tag}_raw.pdf"

    table.to_csv(csv_path, index=False)
    summary.to_csv(summary_path, index=False)
    order = [str(args.baseline_label), str(args.compare_label)]
    plot_weight_grid(table, order, norm_png, norm_pdf, int(args.dpi))
    plot_raw_weight_grid(table, order, raw_png, raw_pdf, int(args.dpi))
    print(f"[ok] csv -> {csv_path}")
    print(f"[ok] summary -> {summary_path}")
    print(f"[ok] raw png -> {raw_png}")
    print(f"[ok] normalized png -> {norm_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
