#!/usr/bin/env python3
"""Analytical metal/nonmetal interface thermal boundary conductance (TBC) models.

This module provides first-pass analytical estimates for:

1. Debye-model phonon spectra (placeholder for MP/Phonopy data).
2. DMM (Diffuse Mismatch Model) phonon-phonon interface conductance.
3. Li et al. 2015 series-parallel resistor network for metal/nonmetal TBC.

All units are SI.  Functions are independent of the phonon MC solver.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# --- physical constants (SI) ---
HBAR = 1.054571817e-34
K_B = 1.380649e-23


# ======================================================================
# 1. Debye phonon spectrum generator
# ======================================================================

def debye_spectrum(
    v_l: float,
    v_t: float,
    n_atoms: float | None = None,
    omega_D: float | None = None,
    omega_max: float | None = None,
    n_omega: int = 1000,
    n_t_branches: int = 2,
) -> dict[str, Any]:
    """Build a branch-resolved Debye-model phonon spectrum.

    Parameters
    ----------
    v_l : float
        Longitudinal sound speed (m/s).
    v_t : float
        Transverse sound speed (m/s).
    n_atoms : float, optional
        Atomic number density (atoms/m^3).  Used to compute the Debye
        frequency when *omega_D* is not given.
    omega_D : float, optional
        Debye frequency (rad/s).  Overrides *n_atoms*.
    omega_max : float, optional
        Hard cut-off frequency (rad/s).  Used as a fallback when neither
        *n_atoms* nor *omega_D* are supplied.  This is a **first-pass
        placeholder** — replace with literature or Phonopy data for
        quantitative studies.
    n_omega : int
        Number of frequency grid points.
    n_t_branches : int
        Number of degenerate transverse branches (default 2).

    Returns
    -------
    dict
        Keys: ``omega``, ``DOS_w_b``, ``vg_w_b``, ``branch_names``,
        ``v_l``, ``v_t``, ``omega_max``, ``notes``.

    Notes
    -----
    The Debye DOS per branch is::

        DOS_b(omega) = omega^2 / (2 * pi^2 * v_b^3)   [s / (m^3 rad)]

    with v_LA = v_l, v_TA = v_t.  This is a spherical-Brillouin-zone
    approximation valid for first-pass physical trend analysis only.
    """
    # Determine frequency cut-off.
    if omega_D is not None and omega_D > 0:
        w_max = float(omega_D)
        note = f"Debye frequency from input: omega_D = {w_max:.3e} rad/s"
    elif n_atoms is not None and n_atoms > 0:
        # Debye frequency: omega_D = v_D * (6 * pi^2 * n)^(1/3)
        # where 3/v_D^3 = 1/v_l^3 + 2/v_t^3
        v_D_inv3 = (1.0 / v_l**3 + float(n_t_branches) / v_t**3) / 3.0
        v_D = v_D_inv3 ** (-1.0 / 3.0)
        w_max = v_D * (6.0 * np.pi**2 * float(n_atoms)) ** (1.0 / 3.0)
        note = f"Debye frequency from n_atoms={n_atoms:.3e}: omega_D = {w_max:.3e} rad/s"
    elif omega_max is not None and omega_max > 0:
        w_max = float(omega_max)
        note = (
            f"PLACEHOLDER omega_max = {w_max:.3e} rad/s — "
            "replace by literature/MP/Phonopy data for quantitative studies"
        )
    else:
        raise ValueError(
            "At least one of n_atoms, omega_D, or omega_max must be provided "
            "and positive."
        )

    # Frequency grid.
    omega = np.linspace(0.0, w_max, n_omega + 1, dtype=np.float64)
    w_mid = 0.5 * (omega[:-1] + omega[1:])
    dw = np.diff(omega)

    # Branch velocities and count.
    v_vals = [v_l] + [v_t] * max(1, int(n_t_branches))
    branch_names = ["LA"] + [f"TA{i+1}" for i in range(max(1, int(n_t_branches)))]
    B = len(v_vals)

    # Debye DOS per branch:  DOS_b(omega) = omega^2 / (2 * pi^2 * v_b^3)
    DOS_w_b = np.zeros((B, n_omega), dtype=np.float64)
    vg_w_b = np.zeros((B, n_omega), dtype=np.float64)
    for ib, vb in enumerate(v_vals):
        # DOS within our frequency grid.
        DOS_w_b[ib, :] = w_mid**2 / (2.0 * np.pi**2 * vb**3)
        # For the valid frequency range, DOS is non-zero; beyond omega_max it is 0.
        # Group velocity magnitude = vb (Debye approximation: vg is constant).
        vg_w_b[ib, :] = vb

    return {
        "omega": w_mid,
        "omega_edges": omega,
        "dw": dw,
        "DOS_w_b": DOS_w_b,
        "vg_w_b": vg_w_b,
        "branch_names": branch_names,
        "B": B,
        "n_omega": n_omega,
        "v_l": v_l,
        "v_t": v_t,
        "omega_max": w_max,
        "notes": note,
    }


# ======================================================================
# 2. DMM phonon-phonon analysis
# ======================================================================

def dmm_transmission_from_spectra(
    spec_i: dict[str, Any],
    spec_j: dict[str, Any],
) -> dict[str, Any]:
    """Compute DMM transmission probability between two phonon spectra.

    Parameters
    ----------
    spec_i, spec_j : dict
        Spectrum dicts as returned by :func:`debye_spectrum` (or from the
        phonon MC framework).  Must contain ``omega``, ``DOS_w_b`` and
        ``vg_w_b``.

    Returns
    -------
    dict
        Keys: ``omega``, ``M_i``, ``M_j``, ``T_i_to_j``, ``T_j_to_i``.
        All arrays have length ``len(spec_i["omega"])``.
    """
    # Use spec_i frequency grid; interpolate spec_j onto it if needed.
    w_i = np.asarray(spec_i["omega"], dtype=np.float64)
    w_j = np.asarray(spec_j["omega"], dtype=np.float64)
    # Interpolate if grids differ in length or values.
    same_grid = (w_i.size == w_j.size) and bool(np.allclose(w_i, w_j, rtol=1e-10))
    if not same_grid:
        DOS_j_interp = _interp_onto_grid(
            w_j, np.asarray(spec_j["DOS_w_b"], dtype=np.float64), w_i
        )
        vg_j_interp = _interp_onto_grid(
            w_j, np.asarray(spec_j["vg_w_b"], dtype=np.float64), w_i
        )
    else:
        DOS_j_interp = np.asarray(spec_j["DOS_w_b"], dtype=np.float64)
        vg_j_interp = np.asarray(spec_j["vg_w_b"], dtype=np.float64)

    DOS_i = np.asarray(spec_i["DOS_w_b"], dtype=np.float64)
    vg_i = np.asarray(spec_i["vg_w_b"], dtype=np.float64)

    # Branch-resolved M(omega) = sum_b DOS_b * |vg_b|
    M_i = np.maximum((DOS_i * np.abs(vg_i)).sum(axis=0), 0.0)
    M_j = np.maximum((DOS_j_interp * np.abs(vg_j_interp)).sum(axis=0), 0.0)

    denom = np.maximum(M_i + M_j, 1e-30)
    T_i_to_j = np.clip(M_j / denom, 0.0, 1.0)
    T_j_to_i = np.clip(M_i / denom, 0.0, 1.0)

    # Zero-DOS bins: default to 0.5 (equal probability).
    zero_mask = (M_i == 0.0) & (M_j == 0.0)
    T_i_to_j[zero_mask] = 0.5
    T_j_to_i[zero_mask] = 0.5

    return {
        "omega": w_i,
        "M_i": M_i,
        "M_j": M_j,
        "T_i_to_j": T_i_to_j,
        "T_j_to_i": T_j_to_i,
    }


def _interp_onto_grid(
    w_src: np.ndarray, arr_2d: np.ndarray, w_dst: np.ndarray
) -> np.ndarray:
    """Linearly interpolate a (B, Nw_src) array onto a new frequency grid."""
    B = arr_2d.shape[0]
    result = np.zeros((B, len(w_dst)), dtype=np.float64)
    for ib in range(B):
        result[ib] = np.interp(w_dst, w_src, arr_2d[ib], left=0.0, right=0.0)
    return result


def _bose_dndT(omega: np.ndarray, T: float) -> np.ndarray:
    """Compute dn_BE/dT without numerical overflow.

    dn/dT = (hbar*omega / (kB*T^2)) * n_BE * (n_BE + 1)
    """
    T = max(float(T), 1e-12)
    x = np.minimum(HBAR * omega / (K_B * T), 700.0)
    ex = np.exp(x)
    n_be = 1.0 / np.maximum(ex - 1.0, np.finfo(np.float64).tiny)
    return (HBAR * omega / (K_B * T * T)) * n_be * (n_be + 1.0)


def dmm_phonon_conductance(
    spec_i: dict[str, Any],
    spec_j: dict[str, Any],
    T: float = 300.0,
    area_factor: float = 0.25,
) -> dict[str, Any]:
    """Compute phonon-phonon DMM interface conductance.

    Uses the linear-response Landauer-like form::

        G_pp = area_factor * integral[
            hbar * omega * dn_BE/dT * M_i(omega) * T_i_to_j(omega) d omega
        ]

    Parameters
    ----------
    spec_i, spec_j : dict
        Phonon spectra for the two materials.
    T : float
        Temperature in K.
    area_factor : float
        Geometric factor for isotropic flux.  Default 0.25 (= 1/4) is the
        standard kinetic-theory value for a hemispherical integration.

    Returns
    -------
    dict
        Keys: ``G_pp_W_m2K``, ``R_pp_m2K_W``, ``omega``, ``integrand``,
        ``T_i_to_j``, ``M_i``.
    """
    dmm = dmm_transmission_from_spectra(spec_i, spec_j)
    omega = dmm["omega"]
    M_i = dmm["M_i"]
    T_ij = dmm["T_i_to_j"]

    dw = np.gradient(omega)
    integrand = HBAR * omega * _bose_dndT(omega, T) * M_i * T_ij
    G_pp = float(np.trapz(integrand, omega)) * float(area_factor)

    if G_pp <= 0:
        G_pp = 1e-30  # safe fallback

    return {
        "G_pp_W_m2K": G_pp,
        "R_pp_m2K_W": 1.0 / G_pp,
        "omega": omega,
        "integrand": integrand,
        "T_i_to_j": T_ij,
        "M_i": M_i,
        "temperature_K": T,
        "area_factor": area_factor,
    }


# ======================================================================
# 3. Li et al. 2015 metal/nonmetal TBC model
# ======================================================================

def metal_nonmetal_tbc(
    G_pp: float,
    G_ep_int: float,
    G_ep_bulk: float,
    kappa_e: float,
    kappa_p: float,
) -> dict[str, Any]:
    """Compute metal/nonmetal interface TBC using the Li et al. 2015 model.

    The total conductance is a parallel combination of an electron-mediated
    channel and a phonon-mediated channel::

        G_total = 1/(R_e_m + R_ep) + 1/(R_p_m + R_pp)

    where the electron-phonon coupling length *l_ep* is::

        l_ep = (G_ep_bulk / kappa_e + G_ep_bulk / kappa_p)^(-1/2)

    and the individual resistances are::

        R_e_m = l_ep / kappa_e    (electron transport in metal)
        R_p_m = l_ep / kappa_p    (phonon transport in metal)
        R_ep  = 1 / G_ep_int      (interface electron-phonon coupling)
        R_pp  = 1 / G_pp          (interface phonon-phonon coupling)

    Parameters
    ----------
    G_pp : float
        Phonon-phonon interface conductance (W/(m^2 K)).
    G_ep_int : float
        Interface electron-phonon conductance (W/(m^2 K)).
    G_ep_bulk : float
        Bulk electron-phonon coupling constant (W/(m^3 K)).
    kappa_e : float
        Electronic thermal conductivity of the metal (W/(m K)).
    kappa_p : float
        Phonon (lattice) thermal conductivity of the metal (W/(m K)).

    Returns
    -------
    dict
        All conductances (W/(m^2 K)), resistances (m^2 K/W), l_ep (m),
        and channel fractions.

    Notes
    -----
    This is a first-pass analytical model.  For Cu/TiN the user should note
    that TiN is a conductive ceramic — it has both electron and phonon
    contributions, but in this model TiN is treated as the phonon-accepting
    side.  Quantitative predictions require literature-calibrated parameters.
    """
    # Validate inputs.
    for name, val in [
        ("G_pp", G_pp), ("G_ep_int", G_ep_int), ("G_ep_bulk", G_ep_bulk),
        ("kappa_e", kappa_e), ("kappa_p", kappa_p),
    ]:
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {val}")

    # Coupling length.
    l_ep = 1.0 / np.sqrt(G_ep_bulk / kappa_e + G_ep_bulk / kappa_p)

    # Resistances.
    R_e_m = l_ep / kappa_e
    R_p_m = l_ep / kappa_p
    R_ep = 1.0 / G_ep_int
    R_pp = 1.0 / G_pp

    # Channel conductances.
    G_e_ch = 1.0 / (R_e_m + R_ep)
    G_p_ch = 1.0 / (R_p_m + R_pp)
    G_total = G_e_ch + G_p_ch

    e_frac = G_e_ch / G_total if G_total > 0 else 0.0
    p_frac = G_p_ch / G_total if G_total > 0 else 0.0

    return {
        "G_total_W_m2K": G_total,
        "G_e_channel_W_m2K": G_e_ch,
        "G_p_channel_W_m2K": G_p_ch,
        "electron_channel_fraction": e_frac,
        "phonon_channel_fraction": p_frac,
        "l_ep_m": l_ep,
        "R_e_m_m2K_W": R_e_m,
        "R_p_m_m2K_W": R_p_m,
        "R_ep_m2K_W": R_ep,
        "R_pp_m2K_W": R_pp,
    }


# ======================================================================
# 4. Parameter sweep
# ======================================================================

def sweep_metal_interface_tbc(
    base_params: dict[str, float],
    sweep_params: dict[str, np.ndarray | list[float]],
) -> pd.DataFrame:
    """Sweep metal/nonmetal TBC parameters and return a DataFrame.

    Parameters
    ----------
    base_params : dict
        Default values for all TBC parameters.  Must contain:
        ``G_pp``, ``G_ep_int``, ``G_ep_bulk``, ``kappa_e``, ``kappa_p``.
    sweep_params : dict
        Dict mapping parameter names to arrays of values to sweep.
        Each key must be one of the five base parameters, or ``kappa_ratio``
        (which sets kappa_e = kappa_p * ratio while keeping kappa_p fixed).

    Returns
    -------
    pd.DataFrame
        One row per parameter combination, with columns for all inputs and
        outputs of :func:`metal_nonmetal_tbc`.
    """
    required = {"G_pp", "G_ep_int", "G_ep_bulk", "kappa_e", "kappa_p"}
    missing = required - set(base_params.keys())
    if missing:
        raise ValueError(f"base_params missing keys: {missing}")

    rows: list[dict[str, Any]] = []

    # Determine which parameter(s) to sweep.
    sweep_keys = list(sweep_params.keys())
    if not sweep_keys:
        # Single evaluation.
        result = metal_nonmetal_tbc(**base_params)
        row = {**base_params, **result}
        rows.append(row)
        return pd.DataFrame(rows)

    # Recursively build the Cartesian product.
    _sweep_recursive(base_params, sweep_params, sweep_keys, 0, rows)

    df = pd.DataFrame(rows)
    return df


def _sweep_recursive(
    base: dict[str, float],
    sweep: dict[str, np.ndarray | list[float]],
    keys: list[str],
    idx: int,
    rows: list[dict[str, Any]],
) -> None:
    if idx >= len(keys):
        # Evaluate at leaf.
        params = dict(base)
        # Handle kappa_ratio → kappa_e = kappa_p * ratio
        if "kappa_ratio" in base:
            params["kappa_e"] = base["kappa_p"] * base["kappa_ratio"]
        try:
            result = metal_nonmetal_tbc(
                G_pp=params["G_pp"],
                G_ep_int=params["G_ep_int"],
                G_ep_bulk=params["G_ep_bulk"],
                kappa_e=params["kappa_e"],
                kappa_p=params["kappa_p"],
            )
        except ValueError:
            return  # skip invalid parameter combinations
        row: dict[str, Any] = {
            "G_pp_W_m2K": params["G_pp"],
            "G_ep_int_W_m2K": params["G_ep_int"],
            "G_ep_bulk_W_m3K": params["G_ep_bulk"],
            "kappa_e_W_mK": params["kappa_e"],
            "kappa_p_W_mK": params["kappa_p"],
            **result,
        }
        if "kappa_ratio" in base:
            row["kappa_ratio"] = base["kappa_ratio"]
        rows.append(row)
        return

    key = keys[idx]
    values = np.asarray(sweep[key], dtype=np.float64).ravel()
    for v in values:
        new_base = dict(base)
        if key == "kappa_ratio":
            new_base["kappa_e"] = float(new_base["kappa_p"]) * float(v)
            new_base["kappa_ratio"] = float(v)
        else:
            new_base[key] = float(v)
        _sweep_recursive(new_base, sweep, keys, idx + 1, rows)
