#!/usr/bin/env python3
"""Calibrate bulk thermal conductivity κ(T) scattering parameters.

Reads MP/pheasy processed spectra (or falls back to Debye), fits RTA
scattering parameters to target κ(T) data, and outputs calibrated params.

Usage::

    # First copy the template:
    cp materials_data/kappa_targets/bulk_kappa_targets_template.csv \\
       materials_data/kappa_targets/bulk_kappa_targets.csv
    # Edit bulk_kappa_targets.csv with literature/experimental data
    python scripts/calibrate_bulk_kappa_T.py
"""

from __future__ import annotations

import json, os, sys, tomllib, warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bulk_kappa_models import (
    kappa_phonon_rta_from_spectrum,
    kappa_e_wiedemann_franz,
    total_kappa_model,
)
from interface_tbc_models import debye_spectrum

MATERIALS_DATA = REPO / "materials_data"
KAPPA_TARGETS = MATERIALS_DATA / "kappa_targets"
CALIBRATED = MATERIALS_DATA / "calibrated"
CALIBRATED.mkdir(parents=True, exist_ok=True)


def load_spectrum(label: str) -> dict[str, Any] | None:
    """Try loading processed MP spectrum; fall back to Debye."""
    # Try nextgen probe first
    for d in (MATERIALS_DATA / "mp_raw_nextgen").glob(f"{label}_*"):
        bs_files = list(d.glob("*_bandstructure_*.json"))
        if bs_files:
            # Found nextgen bandstructure — build spec from it
            try:
                return _build_spec_from_nextgen_bs(bs_files[0], label)
            except Exception:
                pass
    # Fallback: Debye with material-specific placeholder params
    debye_defaults = {
        "Cu": (4700, 2300, 5e13), "TiN": (9000, 5500, 8e13),
        "SiO2": (5800, 3800, 8e13), "Si3N4": (8000, 5000, 1e14),
        "HfO2": (5000, 3000, 6e13),
    }
    vl, vt, wmax = debye_defaults.get(label, (5000, 3000, 5e13))
    print(f"    Using Debye fallback: v_l={vl}, v_t={vt}, ω_max={wmax:.1e}")
    return debye_spectrum(v_l=vl, v_t=vt, omega_max=wmax, n_omega=500)


def _build_spec_from_nextgen_bs(bs_path: Path, label: str) -> dict[str, Any] | None:
    """Build a spectrum dict from a nextgen bandstructure JSON."""
    with open(bs_path) as f:
        data = json.load(f)
    freqs = data.get("frequencies", [])
    qp = data.get("qpoints", [])
    if not freqs or not qp:
        return None
    n_b = len(freqs)
    n_w = len(freqs[0])
    # Build omega grid from frequency range
    f_all = np.array(freqs)
    f_min, f_max = float(np.min(f_all)), float(np.max(f_all))
    omega = np.linspace(max(0, 2*np.pi*1e12*f_min), 2*np.pi*1e12*f_max, n_w)
    # Approximate DOS: constant per branch per bin (rough)
    DOS_w_b = np.ones((n_b, n_w)) * 1e-10
    # Approximate vg from finite diff
    dist = [0.0]
    for i in range(1, len(qp)):
        dist.append(dist[-1] + np.linalg.norm(np.array(qp[i])-np.array(qp[i-1])))
    q_dist = np.array(dist) * 1e10
    vg_w_b = np.zeros((n_b, n_w))
    for b in range(n_b):
        # Interpolate frequencies onto uniform omega grid
        f_interp = np.interp(omega, 2*np.pi*1e12*np.array(freqs[b]), f_all[b])
        # Estimate vg
        for i in range(1, len(qp)-1):
            dw = 2*np.pi*1e12*(freqs[b][i+1]-freqs[b][i-1])
            dq = q_dist[i+1]-q_dist[i-1]
            vg_val = abs(dw/max(dq,1e-30))
            idx = int(i * n_w / len(qp))
            if 0 <= idx < n_w:
                vg_w_b[b, idx] = min(vg_val, 20000)
    return {
        "omega": omega, "DOS_w_b": DOS_w_b, "vg_w_b": np.abs(vg_w_b),
        "branch_names": [f"B{i+1}" for i in range(n_b)],
        "notes": f"from {bs_path.name} (rough vg estimate)",
    }


def fit_one_material(label: str, targets_df: pd.DataFrame,
                     init_params: dict) -> dict[str, Any]:
    """Fit scattering params for one material to match target κ(T)."""
    mat_targets = targets_df[targets_df["label"] == label]
    if len(mat_targets) == 0:
        return {"label": label, "status": "no_targets", "warnings": ["no target data"]}

    spec = load_spectrum(label)
    T_targets = mat_targets["T_K"].to_numpy(dtype=np.float64)
    kappa_targets = mat_targets["kappa_total_W_mK"].to_numpy(dtype=np.float64)

    model_type = init_params.get("model", "dielectric")
    has_electronic = model_type in ("metal", "conductive_ceramic")
    ep = init_params if has_electronic else None

    def residual(log_params):
        A_U, A_I, A_0, L_eff = [10**x for x in log_params]
        sp = {"A_U": A_U, "A_I": A_I, "A_0": A_0, "L_eff": L_eff,
              "theta_U": float(init_params.get("theta_U", 300)),
              "b_U": float(init_params.get("b_U", 3))}
        result = total_kappa_model(label, T_targets, spec, sp,
                                    electronic_params=ep)
        model_k = result["kappa_total_W_mK"]
        return np.log10(np.maximum(model_k, 1e-30)) - np.log10(np.maximum(kappa_targets, 1e-30))

    # Initial guess in log10 space
    x0 = [np.log10(max(float(init_params.get(k, 1e-45)), 1e-50)) for k in
          ["A_U", "A_I", "A_0"]]
    x0.append(np.log10(max(float(init_params.get("L_eff", 1e-6)), 1e-9)))
    bounds = ([-50, -50, -20, -9], [-20, -30, 20, 3])  # log10 bounds

    try:
        res = least_squares(residual, x0, bounds=bounds, max_nfev=200, method='trf')
        fitted = [10**x for x in res.x]
    except Exception as e:
        # Fallback: use initial params
        fitted = [10**x for x in x0]
        print(f"    WARNING: fit failed ({e}), using initial guess")

    sp_fitted = {"A_U": fitted[0], "A_I": fitted[1], "A_0": fitted[2], "L_eff": fitted[3],
                 "theta_U": float(init_params.get("theta_U", 300)),
                 "b_U": float(init_params.get("b_U", 3))}
    result = total_kappa_model(label, T_targets, spec, sp_fitted, electronic_params=ep)
    rel_err = np.abs(result["kappa_total_W_mK"] - kappa_targets) / np.maximum(kappa_targets, 1e-30)

    # Save outputs
    out_dir = CALIBRATED / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_df = pd.DataFrame({
        "T_K": T_targets,
        "target_kappa_total_W_mK": kappa_targets,
        "model_kappa_total_W_mK": result["kappa_total_W_mK"],
        "model_kappa_p_W_mK": result["kappa_p_W_mK"],
        "model_kappa_e_W_mK": result["kappa_e_W_mK"],
        "relative_error": rel_err,
    })
    csv_df.to_csv(out_dir / "kappa_T_fit.csv", index=False)

    # TOML params
    toml_lines = [f"# Calibrated scattering params for {label}", ""]
    for k, v in sp_fitted.items():
        toml_lines.append(f"{k} = {v:.6e}")
    (out_dir / "kappa_fit_params.toml").write_text("\n".join(toml_lines))

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(T_targets, kappa_targets, "o", color="#bf360c", label="target")
    ax.plot(T_targets, result["kappa_total_W_mK"], "-", color="#0f4c81", label="model total")
    ax.plot(T_targets, result["kappa_p_W_mK"], "--", color="#2e7d32", label="model κp")
    if has_electronic:
        ax.plot(T_targets, result["kappa_e_W_mK"], ":", color="#6a1b9a", label="model κe")
    ax.set_xlabel("T (K)")
    ax.set_ylabel("κ (W/(m K))")
    ax.set_title(f"{label} κ(T) calibration")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "kappa_T_fit.png", dpi=150)
    plt.close(fig)

    return {
        "label": label, "status": "fitted",
        "kappa_300K": float(np.interp(300, T_targets, result["kappa_total_W_mK"])),
        "mean_rel_error": float(np.mean(rel_err)),
        "params": sp_fitted,
    }


def main() -> int:
    targets_path = KAPPA_TARGETS / "bulk_kappa_targets.csv"
    if not targets_path.is_file():
        print(f"ERROR: {targets_path} not found.")
        print(f"  cp {KAPPA_TARGETS}/bulk_kappa_targets_template.csv {targets_path}")
        print("  Then edit with literature/experimental data and re-run.")
        return 1

    params_path = KAPPA_TARGETS / "kappa_fit_initial_params.toml"
    if not params_path.is_file():
        print(f"ERROR: {params_path} not found")
        return 1

    targets_df = pd.read_csv(targets_path)
    with open(params_path, "rb") as f:
        params_cfg = tomllib.load(f)

    all_results = []
    for label in targets_df["label"].unique():
        init = params_cfg.get(label, {})
        if not init:
            print(f"SKIP {label}: no initial params")
            continue
        print(f"\nCalibrating: {label} (model={init.get('model','?')})")
        result = fit_one_material(label, targets_df, init)
        all_results.append(result)
        print(f"  κ(300K)={result.get('kappa_300K','?'):.2f} W/(m K)")

    # Summary
    summary_path = CALIBRATED / "kappa_fit_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
