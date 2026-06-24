#!/usr/bin/env python3
"""Sweep bulk κ(T) sensitivity to scattering parameters (L_eff, A_0, A_U, A_I).

Falls back to Debye spectrum if no processed MP data is available.

Usage::

    python scripts/sweep_bulk_kappa_sensitivity.py
"""

from __future__ import annotations

import sys, tomllib
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bulk_kappa_models import kappa_phonon_rta_from_spectrum
from interface_tbc_models import debye_spectrum

MATERIALS_DATA = REPO / "materials_data"
PLOTS = MATERIALS_DATA / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
REPORTS = MATERIALS_DATA / "reports"


def load_spec(label: str) -> dict[str, Any]:
    debye_defaults = {
        "Cu": (4700, 2300, 5e13), "TiN": (9000, 5500, 8e13),
        "SiO2": (5800, 3800, 8e13), "Si3N4": (8000, 5000, 1e14),
        "HfO2": (5000, 3000, 6e13),
    }
    vl, vt, wmax = debye_defaults.get(label, (5000, 3000, 5e13))
    print(f"  {label}: Debye fallback (v_l={vl}, v_t={vt}, ω_max={wmax:.1e})")
    return debye_spectrum(v_l=vl, v_t=vt, omega_max=wmax, n_omega=400)


def sweep_one(label: str, spec: dict, base_params: dict) -> pd.DataFrame:
    """Sweep L_eff, A_0, A_U, A_I for one material."""
    rows = []
    T_vals = [200, 300, 400, 500]
    base = dict(base_params)

    # Sweep L_eff
    for L in np.logspace(-9, -5, 21):
        sp = {**base, "L_eff": L}
        for T in T_vals:
            r = kappa_phonon_rta_from_spectrum(spec, T, sp)
            rows.append({"label": label, "sweep_param": "L_eff", "param_value": L,
                         "T_K": T, "kappa_p_W_mK": r["kappa_p_W_mK"]})

    # Sweep A_0
    for A0 in np.logspace(8, 14, 21):
        sp = {**base, "A_0": A0}
        for T in T_vals:
            r = kappa_phonon_rta_from_spectrum(spec, T, sp)
            rows.append({"label": label, "sweep_param": "A_0", "param_value": A0,
                         "T_K": T, "kappa_p_W_mK": r["kappa_p_W_mK"]})

    # Sweep A_U
    for AU in np.logspace(-47, -42, 21):
        sp = {**base, "A_U": AU}
        for T in T_vals:
            r = kappa_phonon_rta_from_spectrum(spec, T, sp)
            rows.append({"label": label, "sweep_param": "A_U", "param_value": AU,
                         "T_K": T, "kappa_p_W_mK": r["kappa_p_W_mK"]})

    return pd.DataFrame(rows)


def plot_sweeps(label: str, df: pd.DataFrame):
    """Generate sensitivity plots."""
    for param in ["L_eff", "A_0", "A_U"]:
        sub = df[df["sweep_param"] == param]
        if len(sub) == 0:
            continue
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        for T in sorted(sub["T_K"].unique()):
            st = sub[sub["T_K"] == T].sort_values("param_value")
            ax.loglog(st["param_value"], st["kappa_p_W_mK"], "o-", markersize=3,
                      linewidth=1.2, label=f"{int(T)} K")
        ax.set_xlabel(param)
        ax.set_ylabel("κp (W/(m K))")
        ax.set_title(f"{label}: κp vs {param}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fname = f"{label}_kappa_vs_{param}.png"
        fig.savefig(PLOTS / fname, dpi=120)
        plt.close(fig)

    # T-dependence plot
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    base_sub = df[(df["sweep_param"] == "A_0") & (df["param_value"].between(1e10, 1e11))]
    for pv, grp in base_sub.groupby("param_value"):
        grp_sorted = grp.sort_values("T_K")
        ax.plot(grp_sorted["T_K"], grp_sorted["kappa_p_W_mK"], "o-", markersize=3,
                linewidth=1.2, label=f"A0={pv:.1e}")
    ax.set_xlabel("T (K)")
    ax.set_ylabel("κp (W/(m K))")
    ax.set_title(f"{label}: κp(T)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(PLOTS / f"{label}_kappa_vs_T_sensitivity.png", dpi=120)
    plt.close(fig)


def main() -> int:
    params_path = MATERIALS_DATA / "kappa_targets" / "kappa_fit_initial_params.toml"
    if not params_path.is_file():
        print(f"ERROR: {params_path} not found")
        return 1
    with open(params_path, "rb") as f:
        cfg = tomllib.load(f)

    all_dfs = []
    for label in ["Cu", "TiN", "SiO2", "Si3N4", "HfO2"]:
        init = cfg.get(label, {})
        if not init:
            continue
        print(f"\nSweeping: {label}")
        spec = load_spec(label)
        base = {"A_U": float(init.get("A_U", 1e-45)),
                "A_I": float(init.get("A_I", 1e-42)),
                "A_0": float(init.get("A_0", 1e10)),
                "L_eff": float(init.get("L_eff", 1e-7)),
                "theta_U": float(init.get("theta_U", 300)),
                "b_U": float(init.get("b_U", 3))}
        df = sweep_one(label, spec, base)
        all_dfs.append(df)
        plot_sweeps(label, df)

    combined = pd.concat(all_dfs, ignore_index=True)
    csv_path = REPORTS / "bulk_kappa_sensitivity_summary.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\nCSV: {csv_path}")
    print(f"Plots: {PLOTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
