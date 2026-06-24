#!/usr/bin/env python3
"""Unit tests for bulk_kappa_models.py."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bulk_kappa_models import (
    bose_mode_heat_capacity,
    kappa_e_wiedemann_franz,
    kappa_phonon_rta_from_spectrum,
    scattering_rate_rta,
    total_kappa_model,
)
from interface_tbc_models import debye_spectrum

FAILS = 0

def check(cond, msg):
    global FAILS
    if cond: print(f"  PASS: {msg}")
    else: print(f"  FAIL: {msg}"); FAILS += 1

# ------------------------------------------------------------------
print("--- Bose heat capacity ---")
omega = np.logspace(12, 14, 100)
C = bose_mode_heat_capacity(omega, 300.0)
check(np.all(C >= 0), "C >= 0")
check(np.all(np.isfinite(C)), "C finite")
check(C.max() > 0, "C peaks at some omega")

# ------------------------------------------------------------------
print("--- Scattering rate ---")
vg = np.ones(100) * 5000.0
rate = scattering_rate_rta(omega, vg, 300.0, A_U=1e-45, A_I=1e-42, L_eff=1e-8, A_0=1e10)
check(np.all(rate > 0), "rate > 0")
check(np.all(np.isfinite(rate)), "rate finite")
tau = 1.0 / rate
check(np.all(np.isfinite(tau)), "tau finite")

# Boundary term dominates for small L_eff
rate_bdry = scattering_rate_rta(omega, vg, 300.0, L_eff=1e-9)
check(np.all(rate_bdry > 0), "boundary rate > 0")

# L_eff=inf → no boundary term
rate_no_bdry = scattering_rate_rta(omega, vg, 300.0, L_eff=np.inf)
check(np.all(np.isfinite(rate_no_bdry)), "L_eff=inf finite")

# ------------------------------------------------------------------
print("--- kappa_phonon_rta ---")
spec = debye_spectrum(v_l=4700, v_t=2300, omega_max=5e13, n_omega=300)
kp_result = kappa_phonon_rta_from_spectrum(spec, 300.0, {"A_U": 1e-45, "A_I": 1e-42, "L_eff": 1e-8, "A_0": 1e10})
check(kp_result["kappa_p_W_mK"] > 0, f"κp > 0: {kp_result['kappa_p_W_mK']:.2f} W/(m K)")

# Monotonicity: stronger scattering → lower κp
kp_strong = kappa_phonon_rta_from_spectrum(spec, 300.0, {"A_0": 1e12})
kp_weak = kappa_phonon_rta_from_spectrum(spec, 300.0, {"A_0": 1e8})
check(kp_strong["kappa_p_W_mK"] <= kp_weak["kappa_p_W_mK"],
      f"A_0 stronger → κ lower: {kp_strong['kappa_p_W_mK']:.2e} < {kp_weak['kappa_p_W_mK']:.2e}")

# Shorter L_eff → lower κp
kp_long = kappa_phonon_rta_from_spectrum(spec, 300.0, {"L_eff": 1e-3})
kp_short = kappa_phonon_rta_from_spectrum(spec, 300.0, {"L_eff": 1e-9})
check(kp_short["kappa_p_W_mK"] <= kp_long["kappa_p_W_mK"],
      f"L_eff shorter → κ lower: {kp_short['kappa_p_W_mK']:.2e} < {kp_long['kappa_p_W_mK']:.2e}")

# Higher T → Umklapp stronger → κp should not increase dramatically
kp_lowT = kappa_phonon_rta_from_spectrum(spec, 200.0, {"A_U": 1e-45})
kp_highT = kappa_phonon_rta_from_spectrum(spec, 500.0, {"A_U": 1e-45})
check(kp_highT["kappa_p_W_mK"] < kp_lowT["kappa_p_W_mK"] * 3,
      f"higher T U-scattering limits κ: {kp_highT['kappa_p_W_mK']:.2e} < 3x {kp_lowT['kappa_p_W_mK']:.2e}")

# ------------------------------------------------------------------
print("--- Wiedemann-Franz ---")
ke = kappa_e_wiedemann_franz(300.0, 1.7e-8, alpha=0.0039)
check(ke > 0, f"κe > 0: {ke:.1f} W/(m K)")
check(300 < ke < 500, f"Cu κe ~ 400 W/(m K) at 300K: got {ke:.1f}")

# WF: κe = L0*T/ρ(T).  With TCR, ρ(200) < ρ(300) so κe can increase at lower T.
ke_low = kappa_e_wiedemann_franz(200.0, 1.7e-8, alpha=0.0039)
check(ke_low > 0, f"κe(200K)={ke_low:.0f} > 0 (WF law)")

# ------------------------------------------------------------------
print("--- total_kappa_model ---")
T_grid = np.array([200, 300, 400])
# Metal: Cu without electronic params should warn
cu_result = total_kappa_model("Cu", T_grid, spec, {"A_0": 1e10})
check(len(cu_result["warnings"]) > 0, f"Cu warns about electronic dominance: {cu_result['warnings'][0][:60]}...")

# With electronic params and calibrated phonon scattering
cu_params_cal = {"A_U": 1e-45, "A_I": 1e-42, "A_0": 5e10, "L_eff": 1e-7}
cu_result_elec = total_kappa_model("Cu", T_grid, spec, cu_params_cal,
                                    electronic_params={"rho0": 1.7e-8, "alpha": 0.0039})
check(np.all(cu_result_elec["kappa_e_W_mK"] > 0), "Cu κe > 0 with WF params")
check(cu_result_elec["kappa_total_W_mK"][1] > 0,
      f"Cu κtotal > 0: {cu_result_elec['kappa_total_W_mK'][1]:.1f} W/(m K)")

# Dielectric: no electronic
sio2_spec = debye_spectrum(v_l=5800, v_t=3800, omega_max=8e13, n_omega=200)
sio2_result = total_kappa_model("SiO2", T_grid, sio2_spec,
                                 {"A_U": 1e-43, "A_I": 1e-41, "L_eff": 1e-8, "A_0": 1e12})
check(np.all(sio2_result["kappa_e_W_mK"] == 0), "SiO2 κe = 0 (dielectric)")
check(np.all(sio2_result["kappa_p_W_mK"] > 0), "SiO2 κp > 0")

print()
if FAILS == 0:
    print("ALL TESTS PASSED")
    sys.exit(0)
else:
    print(f"{FAILS} TEST(S) FAILED")
    sys.exit(1)
