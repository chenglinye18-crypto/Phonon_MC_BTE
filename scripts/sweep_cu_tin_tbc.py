#!/usr/bin/env python3
"""Cu/TiN interface TBC parameter sweep using the analytical Li et al. 2015 model.

**IMPORTANT**: This script uses Debye-model placeholder material parameters.
These are first-pass estimates — replace by literature, Materials Project,
or Phonopy data for quantitative studies.

Usage::

    python scripts/sweep_cu_tin_tbc.py --temperature 300 --output-dir output_cu_tin_tbc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from interface_tbc_models import (
    debye_spectrum,
    dmm_phonon_conductance,
    dmm_transmission_from_spectra,
    metal_nonmetal_tbc,
    sweep_metal_interface_tbc,
)

# ======================================================================
# Placeholder material parameters
# ======================================================================
# These are first-pass estimates.  Replace by literature data for
# quantitative studies.
#
# Cu (metal):
#   v_l ≈ 4700 m/s, v_t ≈ 2300 m/s  (Debye-model sound speeds for Cu)
#   omega_max ≈ 5.0e13 rad/s         (approximate Debye cut-off)
#
# TiN (ceramic/metal-nitride):
#   v_l ≈ 9000 m/s, v_t ≈ 5500 m/s  (hard ceramic, higher sound speeds)
#   omega_max ≈ 8.0e13 rad/s
#
# Metal-side thermal parameters (Cu):
#   kappa_e ≈ 350 W/(m K)            (electronic, near room T)
#   kappa_p ≈  20 W/(m K)            (lattice, small in Cu)
#   G_ep_bulk ≈ 10^16–10^19 W/(m^3 K)  (to be swept)
#   G_ep_int  ≈ 10^7–10^10 W/(m^2 K)   (to be swept)

CU_DEBYE_PARAMS = dict(v_l=4700.0, v_t=2300.0, omega_max=5.0e13)
TIN_DEBYE_PARAMS = dict(v_l=9000.0, v_t=5500.0, omega_max=8.0e13)

DEFAULT_METAL_PARAMS = {
    "kappa_e": 350.0,       # W/(m K)
    "kappa_p": 20.0,        # W/(m K)
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cu/TiN interface TBC parameter sweep (analytical Li et al. 2015 model)"
    )
    p.add_argument("--temperature", type=float, default=300.0,
                   help="Temperature in K. Default: 300")
    p.add_argument("--output-dir", type=str, default="output_cu_tin_tbc",
                   help="Output directory. Default: output_cu_tin_tbc")
    p.add_argument("--n-omega", type=int, default=1000,
                   help="Number of frequency grid points. Default: 1000")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip plot generation, only write CSV/summary.")
    p.add_argument("--dpi", type=int, default=200, help="Figure DPI. Default: 200")
    return p.parse_args()


def write_summary(
    output_dir: Path,
    g_pp: float,
    metal_params: dict,
    df_bulk: pd.DataFrame,
    df_int: pd.DataFrame,
    dmm_result: dict,
) -> None:
    """Write a human-readable summary."""
    lines = [
        "Cu/TiN Interface TBC — Analytical Estimate",
        "=" * 60,
        "",
        "IMPORTANT: Placeholder Debye-model parameters.  Replace by",
        "literature / Materials Project / Phonopy data for quantitative studies.",
        "",
        f"Temperature:             {metal_params.get('temperature_K', 300):.1f} K",
        "",
        "--- Cu Debye spectrum ---",
        f"  v_l              = {CU_DEBYE_PARAMS['v_l']} m/s",
        f"  v_t              = {CU_DEBYE_PARAMS['v_t']} m/s",
        f"  omega_max        = {CU_DEBYE_PARAMS['omega_max']:.2e} rad/s",
        "",
        "--- TiN Debye spectrum ---",
        f"  v_l              = {TIN_DEBYE_PARAMS['v_l']} m/s",
        f"  v_t              = {TIN_DEBYE_PARAMS['v_t']} m/s",
        f"  omega_max        = {TIN_DEBYE_PARAMS['omega_max']:.2e} rad/s",
        "",
        "--- DMM phonon-phonon conductance ---",
        f"  G_pp             = {g_pp:.4e} W/(m^2 K)",
        f"  G_pp             = {g_pp*1e-9:.4f} GW/(m^2 K)",
        "",
        "--- Metal-side parameters (Cu) ---",
        f"  kappa_e          = {metal_params['kappa_e']} W/(m K)",
        f"  kappa_p          = {metal_params['kappa_p']} W/(m K)",
        "",
        "--- G_ep_bulk sweep (G_ep_int fixed at 1e9) ---",
        f"  G_ep_bulk range  = {df_bulk['G_ep_bulk_W_m3K'].min():.1e} – "
        f"{df_bulk['G_ep_bulk_W_m3K'].max():.1e} W/(m^3 K)",
        f"  G_total range    = {df_bulk['G_total_W_m2K'].min():.2e} – "
        f"{df_bulk['G_total_W_m2K'].max():.2e} W/(m^2 K)",
        "",
        "--- G_ep_int sweep (G_ep_bulk fixed at 1e17) ---",
        f"  G_ep_int range   = {df_int['G_ep_int_W_m2K'].min():.1e} – "
        f"{df_int['G_ep_int_W_m2K'].max():.1e} W/(m^2 K)",
        f"  G_total range    = {df_int['G_total_W_m2K'].min():.2e} – "
        f"{df_int['G_total_W_m2K'].max():.2e} W/(m^2 K)",
        "",
        "--- Interpretation ---",
        "G_total is dominated by the parallel combination of an electron-",
        "mediated channel and a phonon-mediated channel.  Increasing",
        "G_ep_bulk reduces the coupling length l_ep, increasing G_e_channel.",
        "Increasing G_ep_int directly increases G_e_channel.",
        "The phonon channel (G_pp) provides a floor when the electron",
        "channels are weak.",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_sweeps(
    output_dir: Path,
    df_bulk: pd.DataFrame,
    df_int: pd.DataFrame,
    dmm_result: dict,
    dpi: int,
) -> None:
    """Generate all diagnostic plots."""
    style = {
        "font.family": "DejaVu Serif", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 13,
    }
    plt.rcParams.update(style)

    # --- G_total vs G_ep_bulk ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    x = df_bulk["G_ep_bulk_W_m3K"].to_numpy()
    y = df_bulk["G_total_W_m2K"].to_numpy()
    ax.loglog(x, y * 1e-9, "o-", color="#0f4c81", markersize=4, linewidth=1.5)
    ax.set_xlabel(r"$G_{ep,bulk}$ (W m$^{-3}$ K$^{-1}$)")
    ax.set_ylabel(r"$G_{total}$ (GW m$^{-2}$ K$^{-1}$)")
    ax.set_title("Cu/TiN: Total TBC vs bulk e-ph coupling")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(output_dir / "G_total_vs_G_ep_bulk.png", dpi=dpi)
    plt.close(fig)

    # --- G_total vs G_ep_int ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    x = df_int["G_ep_int_W_m2K"].to_numpy()
    y = df_int["G_total_W_m2K"].to_numpy()
    ax.loglog(x, y * 1e-9, "s-", color="#bf360c", markersize=4, linewidth=1.5)
    ax.set_xlabel(r"$G_{ep,int}$ (W m$^{-2}$ K$^{-1}$)")
    ax.set_ylabel(r"$G_{total}$ (GW m$^{-2}$ K$^{-1}$)")
    ax.set_title("Cu/TiN: Total TBC vs interface e-ph conductance")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(output_dir / "G_total_vs_G_ep_int.png", dpi=dpi)
    plt.close(fig)

    # --- Channel fractions vs G_ep_bulk ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    x = df_bulk["G_ep_bulk_W_m3K"].to_numpy()
    ax.semilogx(x, df_bulk["electron_channel_fraction"].to_numpy(),
                "o-", color="#0f4c81", markersize=4, linewidth=1.5, label="electron")
    ax.semilogx(x, df_bulk["phonon_channel_fraction"].to_numpy(),
                "s-", color="#bf360c", markersize=4, linewidth=1.5, label="phonon")
    ax.set_xlabel(r"$G_{ep,bulk}$ (W m$^{-3}$ K$^{-1}$)")
    ax.set_ylabel("Channel fraction")
    ax.set_title("Cu/TiN: Channel fraction vs bulk e-ph coupling")
    ax.legend(frameon=False)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(output_dir / "channel_fraction_vs_G_ep_bulk.png", dpi=dpi)
    plt.close(fig)

    # --- G_total vs G_pp ---
    base = {
        "G_pp": 1e8, "G_ep_int": 1e9, "G_ep_bulk": 1e17,
        "kappa_e": 350.0, "kappa_p": 20.0,
    }
    df_pp = sweep_metal_interface_tbc(
        base, {"G_pp": np.logspace(6, 10, 41)}
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    x = df_pp["G_pp_W_m2K"].to_numpy()
    y = df_pp["G_total_W_m2K"].to_numpy()
    ax.loglog(x, y * 1e-9, "D-", color="#2e7d32", markersize=4, linewidth=1.5)
    ax.set_xlabel(r"$G_{pp}$ (W m$^{-2}$ K$^{-1}$)")
    ax.set_ylabel(r"$G_{total}$ (GW m$^{-2}$ K$^{-1}$)")
    ax.set_title("Cu/TiN: Total TBC vs phonon-phonon conductance")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(output_dir / "G_total_vs_G_pp.png", dpi=dpi)
    plt.close(fig)

    # --- DMM transmission spectrum ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    omega = dmm_result["omega"]
    ax.plot(omega, dmm_result["T_i_to_j"], color="#0f4c81", linewidth=1.5,
            label=r"$T_{\mathrm{Cu}\to\mathrm{TiN}}$")
    ax.plot(omega, dmm_result["T_j_to_i"], color="#bf360c", linewidth=1.5,
            linestyle="--", label=r"$T_{\mathrm{TiN}\to\mathrm{Cu}}$")
    ax.set_xlabel(r"$\omega$ (rad/s)")
    ax.set_ylabel("DMM transmission probability")
    ax.set_title("Cu/TiN: DMM transmission spectrum")
    ax.legend(frameon=False)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "dmm_transmission_spectrum.png", dpi=dpi)
    plt.close(fig)

    # Also save the G_pp sweep CSV.
    df_pp.to_csv(output_dir / "cu_tin_tbc_sweep_G_pp.csv", index=False)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    T = float(args.temperature)

    # ------------------------------------------------------------------
    # 1. Build Debye spectra.
    # ------------------------------------------------------------------
    print("Building Debye spectra ...")
    spec_cu = debye_spectrum(**CU_DEBYE_PARAMS, n_omega=args.n_omega)
    spec_tin = debye_spectrum(**TIN_DEBYE_PARAMS, n_omega=args.n_omega)
    print(f"  Cu:  {spec_cu['notes']}")
    print(f"  TiN: {spec_tin['notes']}")

    # Save spectrum data.
    spec_df = pd.DataFrame({
        "omega_rad_s": spec_cu["omega"],
        "DOS_Cu_LA": spec_cu["DOS_w_b"][0],
        "DOS_Cu_TA1": spec_cu["DOS_w_b"][1],
        "DOS_Cu_TA2": spec_cu["DOS_w_b"][2],
        "DOS_TiN_LA": spec_tin["DOS_w_b"][0],
        "DOS_TiN_TA1": spec_tin["DOS_w_b"][1],
        "DOS_TiN_TA2": spec_tin["DOS_w_b"][2],
    })
    spec_df.to_csv(output_dir / "cu_tin_dmm_spectrum.csv", index=False)

    # ------------------------------------------------------------------
    # 2. DMM phonon-phonon conductance.
    # ------------------------------------------------------------------
    print("Computing DMM phonon-phonon conductance ...")
    g_pp_result = dmm_phonon_conductance(spec_cu, spec_tin, T=T)
    G_pp = g_pp_result["G_pp_W_m2K"]
    print(f"  G_pp = {G_pp:.4e} W/(m^2 K) = {G_pp*1e-9:.4f} GW/(m^2 K)")

    # Save DMM transmission data.
    dmm_data = dmm_transmission_from_spectra(spec_cu, spec_tin)
    dmm_df = pd.DataFrame({
        "omega_rad_s": dmm_data["omega"],
        "M_Cu": dmm_data["M_i"],
        "M_TiN": dmm_data["M_j"],
        "T_Cu_to_TiN": dmm_data["T_i_to_j"],
        "T_TiN_to_Cu": dmm_data["T_j_to_i"],
    })
    dmm_df.to_csv(output_dir / "cu_tin_dmm_transmission.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Metal/nonmetal TBC parameter sweeps.
    # ------------------------------------------------------------------
    metal_params = dict(DEFAULT_METAL_PARAMS)
    metal_params["temperature_K"] = T

    base = {
        "G_pp": G_pp,
        "G_ep_int": 1e9,
        "G_ep_bulk": 1e17,
        **metal_params,
    }

    # Sweep G_ep_bulk.
    print("Sweeping G_ep_bulk ...")
    df_bulk = sweep_metal_interface_tbc(
        base, {"G_ep_bulk": np.logspace(16, 19, 31)}
    )
    df_bulk.to_csv(output_dir / "cu_tin_tbc_sweep_G_ep_bulk.csv", index=False)
    print(f"  G_total range: {df_bulk['G_total_W_m2K'].min():.2e} – "
          f"{df_bulk['G_total_W_m2K'].max():.2e} W/(m^2 K)")

    # Sweep G_ep_int.
    print("Sweeping G_ep_int ...")
    df_int = sweep_metal_interface_tbc(
        base, {"G_ep_int": np.logspace(7, 10, 31)}
    )
    df_int.to_csv(output_dir / "cu_tin_tbc_sweep_G_ep_int.csv", index=False)
    print(f"  G_total range: {df_int['G_total_W_m2K'].min():.2e} – "
          f"{df_int['G_total_W_m2K'].max():.2e} W/(m^2 K)")

    # ------------------------------------------------------------------
    # 4. Output.
    # ------------------------------------------------------------------
    write_summary(output_dir, G_pp, metal_params, df_bulk, df_int, dmm_data)

    if not args.no_plot:
        print("Generating plots ...")
        plot_sweeps(output_dir, df_bulk, df_int, dmm_data, args.dpi)

    print(f"\nOutput written to: {output_dir}")
    print(f"  {output_dir / 'summary.txt'}")
    print(f"  {output_dir / 'cu_tin_dmm_spectrum.csv'}")
    print(f"  {output_dir / 'cu_tin_dmm_transmission.csv'}")
    print(f"  {output_dir / 'cu_tin_tbc_sweep_G_ep_bulk.csv'}")
    print(f"  {output_dir / 'cu_tin_tbc_sweep_G_ep_int.csv'}")
    if not args.no_plot:
        for fname in [
            "G_total_vs_G_ep_bulk.png", "G_total_vs_G_ep_int.png",
            "G_total_vs_G_pp.png", "channel_fraction_vs_G_ep_bulk.png",
            "dmm_transmission_spectrum.png",
        ]:
            print(f"  {output_dir / fname}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
