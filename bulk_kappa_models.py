#!/usr/bin/env python3
"""Bulk thermal conductivity models for phonon RTA and electronic WF.

Provides:
- bose_mode_heat_capacity()
- scattering_rate_rta()          — Umklapp + impurity + boundary + constant
- kappa_phonon_rta_from_spectrum() — integrates κp from a phonon spectrum
- kappa_e_wiedemann_franz()      — electronic thermal conductivity
- total_kappa_model()            — combines κp + κe per material type

All units are SI unless noted otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

HBAR = 1.054571817e-34
K_B = 1.380649e-23


# ======================================================================
# 1. Bose heat capacity
# ======================================================================

def bose_mode_heat_capacity(omega: np.ndarray, T: float) -> np.ndarray:
    """C_mode = k_B * x^2 * exp(x) / (exp(x)-1)^2   [J/K per mode]

    x = ħω / k_B T.  Safe against overflow.
    """
    T = max(float(T), 1e-12)
    w = np.asarray(omega, dtype=np.float64).ravel()
    x = np.minimum(HBAR * w / (K_B * T), 700.0)
    # Handle ω → 0:  C → k_B (classical limit), not 0/0.
    small = w < 1e-6
    C = np.zeros_like(w)
    # For ω > 0, use the full quantum formula.
    mask = ~small
    if np.any(mask):
        ex = np.exp(x[mask])
        denom = np.maximum(ex - 1.0, 1e-300)
        C[mask] = K_B * x[mask] * x[mask] * ex / (denom * denom)
    # For ω ≈ 0,  C(ω→0) = k_B  (each mode carries k_B in classical limit).
    C[small] = K_B
    return C


# ======================================================================
# 2. RTA scattering rate
# ======================================================================

def scattering_rate_rta(
    omega: np.ndarray,
    vg: np.ndarray,
    T: float,
    A_U: float = 0.0,
    theta_U: float = 300.0,
    b_U: float = 3.0,
    A_I: float = 0.0,
    L_eff: float = np.inf,
    A_0: float = 0.0,
) -> np.ndarray:
    """Compute phonon scattering rate τ⁻¹ (1/s) in RTA.

    τ⁻¹ = A_U * ω² * T * exp[-θ_U / (b_U * T)]    (Umklapp)
        + A_I * ω⁴                                   (impurity / isotope)
        + vg / L_eff                                  (boundary / grain)
        + A_0                                         (constant background)

    All terms are clamped ≥ 0.  Minimum rate = 1e-30.
    """
    # Force 1D arrays and ensure matching sizes.
    w = np.asarray(omega, dtype=np.float64).reshape(-1)
    v = np.asarray(vg, dtype=np.float64).reshape(-1)
    n = max(w.size, v.size)
    if w.size == 1:
        w = np.full(n, w[0])
    if v.size == 1:
        v = np.full(n, v[0])
    if w.size != v.size:
        raise ValueError(f"omega and vg size mismatch after broadcast: {w.size} vs {v.size}")
    T_safe = max(float(T), 1e-12)

    rate = np.zeros(n, dtype=np.float64)

    # Umklapp
    if A_U > 0:
        exp_term = np.exp(-theta_U / max(b_U * T_safe, 1e-12))
        rate += A_U * w * w * T_safe * exp_term

    # Impurity
    if A_I > 0:
        rate += A_I * w ** 4

    # Boundary / grain
    if np.isfinite(L_eff) and L_eff > 0:
        rate += np.abs(v) / L_eff

    # Constant background
    if A_0 > 0:
        rate += A_0

    return np.maximum(rate, 1e-30)


# ======================================================================
# 3. κp from spectrum via RTA
# ======================================================================

def kappa_phonon_rta_from_spectrum(
    spec: dict[str, Any],
    T: float,
    scattering_params: dict[str, float] | None = None,
    geometry_factor: float = 1.0 / 3.0,
) -> dict[str, Any]:
    """Compute phonon thermal conductivity from a spectrum dict.

    κp = geom_factor * Σ_b ∫ C(ω,T) * vg_b²(ω) * τ_b(ω,T) * DOS_b(ω) dω

    Parameters
    ----------
    spec : dict
        Must contain ``omega``, ``DOS_w_b``, ``vg_w_b``, ``branch_names``.
        Compatible with ``debye_spectrum()`` and MP-processed spectra.
    T : float
        Temperature in K.
    scattering_params : dict, optional
        Keys: ``A_U``, ``theta_U``, ``b_U``, ``A_I``, ``L_eff``, ``A_0``.
    geometry_factor : float
        Default 1/3 for isotropic 3D.

    Returns
    -------
    dict with keys: kappa_p_W_mK, branch_kappa_W_mK, temperature_K, ...
    """
    sp = scattering_params or {}
    # Support both debye_spectrum (1D omega) and build_spectral_grid (2D omega/w_mid).
    if "omega" in spec:
        w_arr = np.asarray(spec["omega"], dtype=np.float64)
        omega = w_arr[0] if w_arr.ndim == 2 else w_arr.ravel()
    elif "w_mid" in spec:
        wm = np.asarray(spec["w_mid"], dtype=np.float64)
        omega = wm[0] if wm.ndim == 2 else wm.ravel()
    else:
        raise KeyError("spec must contain 'omega' or 'w_mid'")
    DOS_b = np.asarray(spec["DOS_w_b"], dtype=np.float64)
    vg_raw = np.asarray(spec.get("vg_w_b", spec.get("vg_w", np.zeros_like(DOS_b))),
                        dtype=np.float64)
    # High-symmetry path vg is sparse and NOT a transport vg.
    # Use MP DOS (good) but replace sparse vg with representative sound speeds.
    # Extract per-branch average vg from non-zero values, or use material defaults.
    vg_fraction = np.count_nonzero(np.abs(vg_raw) > 1.0) / max(vg_raw.size, 1)
    vg_max_val = max(float(np.max(np.abs(vg_raw))), 1000.0)
    vg_per_branch = np.zeros(DOS_b.shape[0], dtype=np.float64)
    for ib in range(DOS_b.shape[0]):
        nz = np.abs(vg_raw[ib]) > 1.0
        vg_per_branch[ib] = float(np.mean(np.abs(vg_raw[ib][nz]))) if np.any(nz) else vg_max_val * 0.3
    vg_per_branch = np.maximum(vg_per_branch, 500.0)  # floor 500 m/s
    # Build constant-vg array matching DOS shape.
    vg_b = np.tile(vg_per_branch[:, None], (1, DOS_b.shape[1]))
    if vg_fraction < 0.1:
        spec.setdefault("warnings", []).append(
            f"vg data sparse ({vg_fraction:.1%} non-zero); using per-branch avg vg={vg_per_branch} m/s. "
            "This is a first-pass transport approximation."
        )
    B = DOS_b.shape[0]

    A_U = float(sp.get("A_U", 0.0))
    A_I = float(sp.get("A_I", 0.0))
    A_0 = float(sp.get("A_0", 0.0))
    L_eff = float(sp.get("L_eff", np.inf))
    theta_U = float(sp.get("theta_U", 300.0))
    b_U = float(sp.get("b_U", 3.0))

    dw = np.gradient(omega)
    branch_kappa = np.zeros(B, dtype=np.float64)

    for ib in range(B):
        vg_ib = np.abs(vg_b[ib]).reshape(-1)  # ensure 1D
        C = bose_mode_heat_capacity(omega, T)
        vg2 = vg_ib ** 2
        tau_inv = scattering_rate_rta(omega, vg_ib, T,
                                      A_U=A_U, theta_U=theta_U, b_U=b_U,
                                      A_I=A_I, L_eff=L_eff, A_0=A_0)
        tau = 1.0 / np.maximum(tau_inv, 1e-30)
        integrand = C * vg2 * tau * np.abs(DOS_b[ib]).ravel()
        branch_kappa[ib] = float(np.trapz(integrand, omega))

    kappa_total = float(np.sum(branch_kappa)) * geometry_factor
    branch_kappa = branch_kappa * geometry_factor

    return {
        "kappa_p_W_mK": kappa_total,
        "branch_kappa_W_mK": branch_kappa,
        "temperature_K": T,
        "scattering_params": scattering_params,
        "geometry_factor": geometry_factor,
        "warnings": [],
    }


# ======================================================================
# 4. Electronic thermal conductivity (Wiedemann-Franz)
# ======================================================================

def kappa_e_wiedemann_franz(
    T: float,
    rho0: float,
    alpha: float | None = None,
    T0: float = 300.0,
    lorenz: float = 2.44e-8,
) -> float:
    """Electronic thermal conductivity via Wiedemann-Franz law.

    ρ_e(T) = ρ₀ × [1 + α × (T - T₀)]   (linear TCR model)
    κ_e(T) = L₀ × T / ρ_e(T)

    Parameters
    ----------
    T : float — temperature in K
    rho0 : float — resistivity at T₀ in Ω·m
    alpha : float or None — temperature coefficient of resistivity (1/K)
    T0 : float — reference temperature for rho0 (default 300 K)
    lorenz : float — Lorenz number in W·Ω/K² (default 2.44e-8)
    """
    T = max(float(T), 1e-12)
    if alpha is not None and alpha != 0:
        rho_T = rho0 * (1.0 + alpha * (T - T0))
    else:
        rho_T = rho0
    rho_T = max(rho_T, 1e-20)
    return lorenz * T / rho_T


# ======================================================================
# 5. Total kappa model
# ======================================================================

def total_kappa_model(
    material_label: str,
    T_grid: np.ndarray,
    spec: dict[str, Any],
    phonon_scattering_params: dict[str, float] | None = None,
    electronic_params: dict[str, float] | None = None,
    kappa_e_fraction: float | None = None,
) -> dict[str, Any]:
    """Compute κp(T), κe(T), κtotal(T) for a material.

    Parameters
    ----------
    material_label : str — "Cu", "TiN", "SiO2", etc.
    T_grid : (N,) array — temperature points in K.
    spec : dict — phonon spectrum (Debye or processed MP).
    phonon_scattering_params : dict — for kappa_phonon_rta_from_spectrum.
    electronic_params : dict — ``rho0``, ``alpha``, ``T0``, ``lorenz``.
    kappa_e_fraction : float — fixed κe/κtotal ratio (alternative to WF).

    Returns
    -------
    dict with keys:
        T_K, kappa_p_W_mK, kappa_e_W_mK, kappa_total_W_mK, warnings
    """
    T_arr = np.asarray(T_grid, dtype=np.float64).ravel()
    nT = len(T_arr)
    kp = np.zeros(nT, dtype=np.float64)
    ke = np.zeros(nT, dtype=np.float64)
    warnings: list[str] = []

    # Phonon part
    for i, Ti in enumerate(T_arr):
        result = kappa_phonon_rta_from_spectrum(spec, float(Ti), phonon_scattering_params)
        kp[i] = result["kappa_p_W_mK"]
        warnings.extend(result.get("warnings", []))

    # Electronic part
    if electronic_params is not None and electronic_params.get("rho0", 0) > 0:
        rho0 = float(electronic_params["rho0"])
        alpha = electronic_params.get("alpha", None)
        T0 = float(electronic_params.get("T0", 300.0))
        lorenz = float(electronic_params.get("lorenz", 2.44e-8))
        for i, Ti in enumerate(T_arr):
            ke[i] = kappa_e_wiedemann_franz(float(Ti), rho0, alpha=alpha, T0=T0, lorenz=lorenz)
    elif kappa_e_fraction is not None and 0 < kappa_e_fraction < 1:
        # Fixed fraction of total
        for i in range(nT):
            ke[i] = kp[i] * kappa_e_fraction / max(1.0 - kappa_e_fraction, 1e-30)
    else:
        # No electronic contribution
        ke[:] = 0.0

    ktotal = kp + ke

    # Warnings for metals
    metal_labels = {"Cu", "TiN", "metal"}
    if material_label in metal_labels and (electronic_params is None and kappa_e_fraction is None):
        warnings.append(
            f"{material_label}: total κ is electron-dominated. "
            "Provide electronic_params (rho0) or kappa_e_fraction. "
            "Do NOT fit total κ using phonons only."
        )

    return {
        "T_K": T_arr,
        "kappa_p_W_mK": kp,
        "kappa_e_W_mK": ke,
        "kappa_total_W_mK": ktotal,
        "material_label": material_label,
        "warnings": warnings,
    }
