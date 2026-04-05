from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phonon_mc import HBAR, K_B
from scripts.export_scattering_rate_vs_energy import (
    branch_rates,
    build_opts,
    build_spectral_grid,
    resolve_material_entries,
    sanitize_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Debug thermal-conductivity temperature dependence by separating the omega-integrated "
            "Cv-like factor, v^2 factor, and tau factor."
        )
    )
    parser.add_argument("--input-dir", default="input", help="Input directory containing solver_params.toml. Default: input")
    parser.add_argument("--material", default="", help="Optional material name/key. Default: first material used by the case.")
    parser.add_argument("--T-min", type=float, default=18.0, help="Minimum temperature in K. Default: 18")
    parser.add_argument("--T-max", type=float, default=400.0, help="Maximum temperature in K. Default: 400")
    parser.add_argument("--num-points", type=int, default=383, help="Number of temperatures. Default: 383")
    parser.add_argument("--cos-beta", type=float, default=1.0, help="Same definition as export_scattering_rate_vs_energy.py")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="Thread count. Default: min(8, cpu_count)")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI. Default: 300")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for output figures and CSV. Default: output/kappa_debug_factors",
    )
    return parser.parse_args()


def resolve_output_dir(output_dir: str) -> Path:
    if output_dir:
        path = Path(output_dir).expanduser().resolve()
    else:
        path = ROOT / "output" / "kappa_debug_factors"
    path.mkdir(parents=True, exist_ok=True)
    return path


def choose_material(input_dir: Path, requested_name: str) -> dict[str, object]:
    entries = resolve_material_entries(input_dir, [requested_name] if requested_name else [])
    if not entries:
        raise ValueError(f"no material resolved from {input_dir}")
    return entries[0]


def disambiguate_branch_labels(branches: list[str]) -> list[str]:
    total_count = Counter(branches)
    seen: defaultdict[str, int] = defaultdict(int)
    labels: list[str] = []
    for name in branches:
        seen[name] += 1
        if total_count[name] == 1:
            labels.append(name)
        else:
            labels.append(f"{name}#{seen[name]}")
    return labels


def ensure_dw_2d(spec: dict[str, object]) -> np.ndarray:
    dw = np.asarray(spec["dw"], dtype=np.float64)
    if dw.ndim == 1:
        return dw.reshape(1, -1)
    return dw


def compute_debug_metrics_at_temperature(material_entry: dict[str, object], input_dir: Path, temperature: float, cos_beta: float) -> tuple[pd.DataFrame, list[str]]:
    opts = build_opts(input_dir, float(temperature))
    spec = build_spectral_grid(dict(material_entry["mat"]), opts)
    rates = branch_rates(spec, opts, float(temperature), float(cos_beta))

    w = np.maximum(np.asarray(spec["w_mid"], dtype=np.float64), 0.0)
    dos = np.maximum(np.asarray(spec["DOS_w_b"], dtype=np.float64), 0.0)
    v2 = np.square(np.maximum(np.asarray(rates["vabs_m_s"], dtype=np.float64), 0.0))
    rate_total = np.asarray(rates["rate_total_s_inv"], dtype=np.float64)
    tau = np.zeros_like(rate_total)
    valid = np.isfinite(rate_total) & (rate_total > 0.0)
    tau[valid] = 1.0 / rate_total[valid]
    dw = ensure_dw_2d(spec)

    T = max(float(temperature), 1e-12)
    x = HBAR * w / (K_B * T)
    ex = np.exp(np.minimum(x, 700.0))
    nbe = 1.0 / np.maximum(ex - 1.0, np.finfo(np.float64).tiny)
    dndT = (HBAR * w / (K_B * T * T)) * nbe * (nbe + 1.0)
    cv_like = HBAR * w * dndT

    raw_cv = np.sum(cv_like * dw, axis=1)
    raw_v2 = np.sum(v2 * dw, axis=1)
    raw_tau = np.sum(tau * dw, axis=1)

    dos_int = np.sum(dos * dw, axis=1)
    dos_cv = np.sum(dos * cv_like * dw, axis=1)
    dos_v2 = np.sum(dos * v2 * dw, axis=1)
    dos_tau = np.sum(dos * tau * dw, axis=1)
    kappa_branch = np.sum((1.0 / 3.0) * dos * cv_like * v2 * tau * dw, axis=1)

    branches = list(spec["branches"])
    labels = disambiguate_branch_labels(branches)
    rows: list[dict[str, object]] = []
    branch_metrics = {
        "raw_hw_dndT_int": raw_cv,
        "raw_v2_int": raw_v2,
        "raw_tau_int": raw_tau,
        "dos_int": dos_int,
        "dos_hw_dndT_int": dos_cv,
        "dos_v2_int": dos_v2,
        "dos_tau_int": dos_tau,
        "kappa": kappa_branch,
    }
    for metric_name, values in branch_metrics.items():
        for label, value in zip(labels, values, strict=True):
            rows.append(
                {
                    "temperature_K": float(temperature),
                    "metric": metric_name,
                    "branch": label,
                    "value": float(value),
                }
            )
        rows.append(
            {
                "temperature_K": float(temperature),
                "metric": metric_name,
                "branch": "TOTAL",
                "value": float(np.sum(values)),
            }
        )
    return pd.DataFrame(rows), labels


def compute_metric_table(material_entry: dict[str, object], input_dir: Path, temperatures: np.ndarray, cos_beta: float, jobs: int) -> tuple[pd.DataFrame, list[str]]:
    workers = max(1, int(jobs))
    if workers == 1:
        frames = [compute_debug_metrics_at_temperature(material_entry, input_dir, float(T), cos_beta) for T in temperatures]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            frames = list(executor.map(lambda T: compute_debug_metrics_at_temperature(material_entry, input_dir, float(T), cos_beta), temperatures))
    labels = frames[0][1] if frames else []
    table = pd.concat([frame for frame, _ in frames], ignore_index=True) if frames else pd.DataFrame(columns=["temperature_K", "metric", "branch", "value"])
    return table, labels


def set_debug_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "stix",
            "font.size": 11.5,
            "axes.labelsize": 12.5,
            "axes.titlesize": 12.5,
            "axes.linewidth": 1.1,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "legend.fontsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def metric_title(metric: str) -> str:
    titles = {
        "raw_hw_dndT_int": r"$\int \hbar\omega\,\partial n_{\mathrm{BE}}/\partial T \, d\omega$",
        "raw_v2_int": r"$\int v_g^2 \, d\omega$",
        "raw_tau_int": r"$\int \tau \, d\omega$",
        "dos_int": r"$\int D(\omega) \, d\omega$",
        "dos_hw_dndT_int": r"$\int D(\omega)\,\hbar\omega\,\partial n_{\mathrm{BE}}/\partial T \, d\omega$",
        "dos_v2_int": r"$\int D(\omega)\,v_g^2 \, d\omega$",
        "dos_tau_int": r"$\int D(\omega)\,\tau \, d\omega$",
        "kappa": r"$\kappa(T)$ from full integrand",
    }
    return titles[metric]


def plot_single_metric(
    table: pd.DataFrame,
    metric: str,
    branch_labels: list[str],
    material_name: str,
    output_png: Path,
    output_pdf: Path,
    dpi: int,
) -> None:
    set_debug_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    palette = plt.get_cmap("tab10")
    branch_colors = {label: palette(i % 10) for i, label in enumerate(branch_labels)}
    metric_df = table.loc[table["metric"] == metric].copy()
    total_df = metric_df.loc[metric_df["branch"] == "TOTAL"].sort_values("temperature_K", kind="stable")
    ax.plot(
        total_df["temperature_K"].to_numpy(dtype=np.float64),
        total_df["value"].to_numpy(dtype=np.float64),
        color="#111111",
        linewidth=2.7,
        label="TOTAL",
        zorder=3,
    )
    for label in branch_labels:
        branch_df = metric_df.loc[metric_df["branch"] == label].sort_values("temperature_K", kind="stable")
        ax.plot(
            branch_df["temperature_K"].to_numpy(dtype=np.float64),
            branch_df["value"].to_numpy(dtype=np.float64),
            color=branch_colors[label],
            linewidth=1.7,
            alpha=0.95,
            label=label,
        )
    ax.set_title(f"{material_name} | {metric_title(metric)}")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Integrated Value")
    ax.grid(True, which="major", linestyle="--", linewidth=0.65, alpha=0.30)
    ax.minorticks_on()
    ax.legend(loc="best", frameon=False)
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_metric_family(
    table: pd.DataFrame,
    metrics: list[str],
    branch_labels: list[str],
    material_name: str,
    output_png: Path,
    output_pdf: Path,
    dpi: int,
    subtitle: str,
) -> None:
    set_debug_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), constrained_layout=True)
    axes = axes.ravel()
    palette = plt.get_cmap("tab10")
    branch_colors = {label: palette(i % 10) for i, label in enumerate(branch_labels)}

    for ax, metric in zip(axes, metrics, strict=True):
        metric_df = table.loc[table["metric"] == metric].copy()
        total_df = metric_df.loc[metric_df["branch"] == "TOTAL"].sort_values("temperature_K", kind="stable")
        ax.plot(
            total_df["temperature_K"].to_numpy(dtype=np.float64),
            total_df["value"].to_numpy(dtype=np.float64),
            color="#111111",
            linewidth=2.6,
            label="TOTAL",
            zorder=3,
        )
        for label in branch_labels:
            branch_df = metric_df.loc[metric_df["branch"] == label].sort_values("temperature_K", kind="stable")
            ax.plot(
                branch_df["temperature_K"].to_numpy(dtype=np.float64),
                branch_df["value"].to_numpy(dtype=np.float64),
                color=branch_colors[label],
                linewidth=1.6,
                alpha=0.95,
                label=label,
            )
        ax.set_title(metric_title(metric))
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Integrated Value")
        ax.grid(True, which="major", linestyle="--", linewidth=0.65, alpha=0.30)
        ax.minorticks_on()
        if metric.endswith("tau_int"):
            y = total_df["value"].to_numpy(dtype=np.float64)
            if np.all(np.isfinite(y)) and np.all(y > 0.0):
                ax.set_yscale("log")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=min(4, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"{material_name} Factor Debug vs Temperature\n{subtitle}", y=1.04, fontsize=14)
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    output_dir = resolve_output_dir(args.output_dir)
    material_entry = choose_material(input_dir, str(args.material))
    temperatures = np.linspace(float(args.T_min), float(args.T_max), int(args.num_points), dtype=np.float64)

    table, branch_labels = compute_metric_table(material_entry, input_dir, temperatures, float(args.cos_beta), int(args.jobs))
    safe_name = sanitize_name(str(material_entry["name"]))
    stem = f"kappa_factor_debug_{safe_name}_{temperatures[0]:.0f}K_{temperatures[-1]:.0f}K"

    csv_path = output_dir / f"{stem}.csv"
    raw_png = output_dir / f"{stem}__raw.png"
    raw_pdf = output_dir / f"{stem}__raw.pdf"
    dos_png = output_dir / f"{stem}__dos_weighted.png"
    dos_pdf = output_dir / f"{stem}__dos_weighted.pdf"
    dos_only_png = output_dir / f"{stem}__dos_integral.png"
    dos_only_pdf = output_dir / f"{stem}__dos_integral.pdf"

    table.to_csv(csv_path, index=False)
    plot_metric_family(
        table,
        metrics=["raw_hw_dndT_int", "raw_v2_int", "raw_tau_int", "kappa"],
        branch_labels=branch_labels,
        material_name=str(material_entry["name"]),
        output_png=raw_png,
        output_pdf=raw_pdf,
        dpi=int(args.dpi),
        subtitle="Raw omega integrals plus full conductivity",
    )
    plot_metric_family(
        table,
        metrics=["dos_hw_dndT_int", "dos_v2_int", "dos_tau_int", "kappa"],
        branch_labels=branch_labels,
        material_name=str(material_entry["name"]),
        output_png=dos_png,
        output_pdf=dos_pdf,
        dpi=int(args.dpi),
        subtitle="DOS-weighted omega integrals plus full conductivity",
    )
    plot_single_metric(
        table,
        metric="dos_int",
        branch_labels=branch_labels,
        material_name=str(material_entry["name"]),
        output_png=dos_only_png,
        output_pdf=dos_only_pdf,
        dpi=int(args.dpi),
    )

    print(f"[ok] csv -> {csv_path}")
    print(f"[ok] raw png -> {raw_png}")
    print(f"[ok] raw pdf -> {raw_pdf}")
    print(f"[ok] dos-weighted png -> {dos_png}")
    print(f"[ok] dos-weighted pdf -> {dos_pdf}")
    print(f"[ok] dos-integral png -> {dos_only_png}")
    print(f"[ok] dos-integral pdf -> {dos_only_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
