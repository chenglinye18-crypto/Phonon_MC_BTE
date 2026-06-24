#!/usr/bin/env python3
"""Unit tests for interface_tbc_models.py — PASS/FAIL output."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from interface_tbc_models import (
    debye_spectrum,
    dmm_phonon_conductance,
    dmm_transmission_from_spectra,
    metal_nonmetal_tbc,
    sweep_metal_interface_tbc,
)

FAILS = 0


def check(condition: bool, msg: str) -> None:
    global FAILS
    if condition:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        FAILS += 1


def test_debye_spectrum() -> None:
    print("--- Debye spectrum ---")
    # Placeholder Cu parameters.
    spec = debye_spectrum(v_l=4700.0, v_t=2300.0, omega_max=5.0e13, n_omega=200)
    check(len(spec["omega"]) == 200, "omega grid length")
    check(spec["DOS_w_b"].shape == (3, 200), "DOS shape (LA+2TA x n_omega)")
    check(np.all(spec["DOS_w_b"] >= 0), "DOS non-negative")
    check(np.all(spec["vg_w_b"] >= 0), "vg non-negative")
    check(spec["branch_names"] == ["LA", "TA1", "TA2"], "branch names")
    check("PLACEHOLDER" in spec["notes"], "notes mention placeholder")

    # Test with n_atoms.
    spec2 = debye_spectrum(v_l=4700.0, v_t=2300.0, n_atoms=8.5e28, n_omega=100)
    check(spec2["omega_max"] > 0, "omega_max from n_atoms")
    check("debye" in spec2["notes"].lower(), "notes indicate Debye frequency")

    # Test error on no input.
    try:
        debye_spectrum(v_l=4700, v_t=2300, n_omega=50)
        check(False, "should have raised ValueError with no cut-off")
    except ValueError:
        check(True, "raises ValueError with no cut-off")


def test_dmm_transmission() -> None:
    print("--- DMM transmission ---")
    spec_cu = debye_spectrum(v_l=4700, v_t=2300, omega_max=5.0e13, n_omega=200)
    spec_tin = debye_spectrum(v_l=9000, v_t=5500, omega_max=8.0e13, n_omega=250)

    # Spectra on different grids -> interpolation test.
    dmm = dmm_transmission_from_spectra(spec_cu, spec_tin)
    check(len(dmm["omega"]) == 200, "output omega on first spec grid")
    check(np.all(dmm["T_i_to_j"] >= 0) and np.all(dmm["T_i_to_j"] <= 1),
          "T_i_to_j in [0,1]")
    check(np.all(dmm["T_j_to_i"] >= 0) and np.all(dmm["T_j_to_i"] <= 1),
          "T_j_to_i in [0,1]")
    # T_i_to_j + T_j_to_i should be ~1 where both M > 0.
    mask = (dmm["M_i"] > 1e-30) & (dmm["M_j"] > 1e-30)
    if np.any(mask):
        sum_T = dmm["T_i_to_j"][mask] + dmm["T_j_to_i"][mask]
        check(np.allclose(sum_T, 1.0, atol=1e-12),
              f"T_i_to_j + T_j_to_i = 1 (max diff {np.max(np.abs(sum_T-1)):.2e})")

    # Self-transmission: spec_i = spec_j → T ≈ 0.5 for all bins.
    dmm_self = dmm_transmission_from_spectra(spec_cu, spec_cu)
    T_self = dmm_self["T_i_to_j"]
    mask_self = dmm_self["M_i"] > 0
    if np.any(mask_self):
        check(np.allclose(T_self[mask_self], 0.5, atol=1e-12),
              f"T_i_to_j ≈ 0.5 for same material (max diff {np.max(np.abs(T_self[mask_self]-0.5)):.2e})")


def test_dmm_conductance() -> None:
    print("--- DMM conductance ---")
    spec_cu = debye_spectrum(v_l=4700, v_t=2300, omega_max=5.0e13, n_omega=300)
    spec_tin = debye_spectrum(v_l=9000, v_t=5500, omega_max=8.0e13, n_omega=300)
    g = dmm_phonon_conductance(spec_cu, spec_tin, T=300.0)
    check(g["G_pp_W_m2K"] > 0, f"G_pp > 0: {g['G_pp_W_m2K']:.4e} W/(m^2 K)")
    check(np.isfinite(g["R_pp_m2K_W"]), "R_pp finite")


def test_metal_nonmetal_tbc() -> None:
    print("--- Metal/nonmetal TBC model ---")
    # Placeholder Cu/TiN parameters.
    params = {
        "G_pp": 1e8,
        "G_ep_int": 1e9,
        "G_ep_bulk": 1e17,
        "kappa_e": 350.0,
        "kappa_p": 20.0,
    }
    result = metal_nonmetal_tbc(**params)
    check(result["G_total_W_m2K"] > 0,
          f"G_total > 0: {result['G_total_W_m2K']:.4e}")
    check(0.0 <= result["electron_channel_fraction"] <= 1.0,
          f"e-fraction in [0,1]: {result['electron_channel_fraction']:.4f}")
    check(np.isfinite(result["l_ep_m"]) and result["l_ep_m"] > 0,
          f"l_ep > 0: {result['l_ep_m']:.4e} m")

    # Test input validation.
    try:
        metal_nonmetal_tbc(G_pp=-1, G_ep_int=1e9, G_ep_bulk=1e17,
                           kappa_e=350, kappa_p=20)
        check(False, "should have raised ValueError for negative G_pp")
    except ValueError:
        check(True, "raises ValueError for negative input")

    # Monotonicity: increasing G_ep_int should NOT decrease G_total.
    r1 = metal_nonmetal_tbc(G_pp=1e8, G_ep_int=1e8, G_ep_bulk=1e17,
                            kappa_e=350, kappa_p=20)
    r2 = metal_nonmetal_tbc(G_pp=1e8, G_ep_int=1e10, G_ep_bulk=1e17,
                            kappa_e=350, kappa_p=20)
    check(r2["G_total_W_m2K"] >= r1["G_total_W_m2K"],
          "G_total monotonic with G_ep_int")

    # Monotonicity: increasing G_pp should NOT decrease G_total.
    r3 = metal_nonmetal_tbc(G_pp=1e9, G_ep_int=1e9, G_ep_bulk=1e17,
                            kappa_e=350, kappa_p=20)
    check(r3["G_total_W_m2K"] >= r1["G_total_W_m2K"],
          "G_total monotonic with G_pp")


def test_sweep() -> None:
    print("--- Parameter sweep ---")
    base = {
        "G_pp": 1e8,
        "G_ep_int": 1e9,
        "G_ep_bulk": 1e17,
        "kappa_e": 350.0,
        "kappa_p": 20.0,
    }
    sweep = {"G_ep_bulk": [1e16, 1e17, 1e18]}
    df = sweep_metal_interface_tbc(base, sweep)
    check(len(df) == 3, f"sweep length: {len(df)}")
    check("G_total_W_m2K" in df.columns, "G_total in columns")
    check(df["G_total_W_m2K"].iloc[-1] >= df["G_total_W_m2K"].iloc[0],
          "G_total monotonic with G_ep_bulk")


def main() -> int:
    test_debye_spectrum()
    test_dmm_transmission()
    test_dmm_conductance()
    test_metal_nonmetal_tbc()
    test_sweep()
    print()
    if FAILS == 0:
        print("ALL TESTS PASSED")
        return 0
    else:
        print(f"{FAILS} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
