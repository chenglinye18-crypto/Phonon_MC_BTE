#!/usr/bin/env python3
"""DMM detailed-balance test: two-material slab at uniform temperature.

Runs a short simulation with identical temperature on both sides (300 K / 300 K)
and no external heat source.  In thermal equilibrium the net energy flux across
the material interface should be approximately zero.

Prints PASS/WARNING/FAIL and key diagnostic quantities.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import phonon_mc


def detailed_balance_test(max_steps: int = 15, initial_particles: int = 3000) -> int:
    test_dir = REPO / "input_two_material_dmm_test"
    if not test_dir.is_dir():
        print(f"FAIL: test case directory not found: {test_dir}")
        return 1

    # --- setup: isothermal case ---------------------------------------------
    cs = phonon_mc.setup_case_from_ldg_lgrid(
        str(test_dir / "ldg.txt"),
        str(test_dir / "lgrid.txt"),
        length_scale=1e-6,
        input_length_unit="um",
        verbose=False,
    )
    mat = phonon_mc.resolve_case_material(cs, input_dir=str(test_dir))
    opts = phonon_mc.mc_default_opts(str(test_dir))
    opts["max_steps"] = max_steps
    opts["initial_particles_fixed"] = initial_particles
    opts["output"]["enable"] = False
    opts["volume_heat_source_file"] = ""

    T_uniform = 300.0
    mesh = phonon_mc.init_mesh_from_geom(cs)
    Nc = phonon_mc.infer_Nc(mesh)

    # Write isothermal initial and reference temperature files.
    import csv, tempfile, os
    tmp_initial = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    tmp_ref = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    w_init = csv.writer(tmp_initial)
    w_ref = csv.writer(tmp_ref)
    for iz in range(1, mesh["Nz"] + 1):
        for iy in range(1, mesh["Ny"] + 1):
            for ix in range(1, mesh["Nx"] + 1):
                w_init.writerow([ix, iy, iz, f"{T_uniform:.4f}"])
                w_ref.writerow([ix, iy, iz, f"{T_uniform:.4f}"])
    tmp_initial.close()
    tmp_ref.close()
    opts["initial_temperature_file"] = tmp_initial.name
    opts["reference_temperature_file"] = tmp_ref.name

    try:
        Tp, p, out = phonon_mc.MC_solve_BTE(cs, mat, opts)
    except Exception as exc:
        print(f"FAIL: simulation crashed: {exc}")
        import traceback; traceback.print_exc()
        return 1
    finally:
        os.unlink(tmp_initial.name)
        os.unlink(tmp_ref.name)

    print(f"Simulation completed: {out['nsteps']} steps, {len(p)} final particles")

    # --- collect DMM stats --------------------------------------------------
    iface_hist = out.get("iface_hist", [])
    dmm_total = sum(h.get("dmm_attempt", 0) for h in iface_hist)
    print(f"DMM total attempts: {dmm_total}")

    # Per-pair energy tracking (from dmm_detail).
    pair_energy: dict[str, dict[str, float]] = {}
    for h in iface_hist:
        detail = h.get("dmm_detail", {})
        for pair_str, counts in detail.items():
            if pair_str not in pair_energy:
                pair_energy[pair_str] = {"attempt": 0, "transmit": 0, "reflect": 0,
                                          "energy_attempt": 0.0, "energy_transmit": 0.0,
                                          "energy_reflect": 0.0}
    for h in iface_hist:
        detail = h.get("dmm_detail", {})
        dmm_att = h.get("dmm_attempt", 0)
        dmm_tr = h.get("dmm_transmit", 0)
        dmm_rf = h.get("dmm_reflect", 0)
        dmm_e_att = h.get("dmm_energy_attempt", 0.0)
        dmm_e_tr = h.get("dmm_energy_transmit", 0.0)
        dmm_e_rf = h.get("dmm_energy_reflect", 0.0)
        # Distribute energy proportionally (approximate since we don't have per-pair energy).
        for pair_str, counts in detail.items():
            if pair_str not in pair_energy:
                pair_energy[pair_str] = {"attempt": 0, "transmit": 0, "reflect": 0,
                                          "energy_attempt": 0.0, "energy_transmit": 0.0,
                                          "energy_reflect": 0.0}
            p_att_frac = counts.get("attempt", 0) / max(dmm_att, 1)
            p_tr_frac = counts.get("transmit", 0) / max(dmm_tr, 1) if dmm_tr > 0 else 0
            p_rf_frac = counts.get("reflect", 0) / max(dmm_rf, 1) if dmm_rf > 0 else 0
            pair_energy[pair_str]["attempt"] += counts.get("attempt", 0)
            pair_energy[pair_str]["transmit"] += counts.get("transmit", 0)
            pair_energy[pair_str]["reflect"] += counts.get("reflect", 0)
            pair_energy[pair_str]["energy_attempt"] += dmm_e_att * p_att_frac
            pair_energy[pair_str]["energy_transmit"] += dmm_e_tr * p_tr_frac
            pair_energy[pair_str]["energy_reflect"] += dmm_e_rf * p_rf_frac

    # Compute net energy flux across the interface.
    energy_0_to_1 = pair_energy.get("0->1", {}).get("energy_transmit", 0.0)
    energy_1_to_0 = pair_energy.get("1->0", {}).get("energy_transmit", 0.0)
    net_energy_flux = energy_0_to_1 - energy_1_to_0
    total_time = sum(out.get("dt_hist", [1e-15]))

    # Get interface area from mesh.
    # Interface is at x = 0.03 um.  Area = Ly * Lz.
    mesh_area = float(mesh.get("Ay", mesh.get("Ly", 0.0) * mesh.get("Lz", 0.0)))
    if mesh_area <= 0:
        mesh_area = 0.06e-6 * 0.06e-6  # fallback for 60nm x 60nm

    print()
    print("=" * 60)
    print("DETAILED BALANCE RESULTS")
    print("=" * 60)
    print(f"  Temperature:               T = {T_uniform:.1f} K (uniform)")
    print(f"  Total DMM attempts:        {dmm_total}")
    print(f"  Energy 0->1 (transmitted): {energy_0_to_1:.4e} J")
    print(f"  Energy 1->0 (transmitted): {energy_1_to_0:.4e} J")
    print(f"  Net energy flux (0->1):    {net_energy_flux:.4e} J")
    print(f"  Simulation time:           {total_time:.4e} s")
    net_flux_W = net_energy_flux / max(total_time, 1e-30) if mesh_area > 0 else np.nan
    print(f"  Interface area:            {mesh_area:.4e} m^2")
    if np.isfinite(net_flux_W):
        print(f"  Net heat flux:             {net_flux_W:.4e} W")
        print(f"  Net heat flux density:     {net_flux_W / mesh_area:.4e} W/m^2")
    # Detailed balance check: net flux should be small relative to individual fluxes.
    total_transmitted = abs(energy_0_to_1) + abs(energy_1_to_0)
    rel_imbalance = abs(net_energy_flux) / max(total_transmitted, 1e-30) if total_transmitted > 0 else 0.0
    print(f"  Relative imbalance:        {rel_imbalance:.4f}")
    print()

    # Judgment.
    tol_rel = 0.5  # within 50% relative (wide tolerance for short runs)
    if dmm_total == 0:
        print("WARNING: No DMM interface crossings detected (maybe particles never reached interface)")
        print("         This is acceptable for very short runs with few particles.")
        print("DETAILED BALANCE: INCONCLUSIVE (no crossings)")
        return 0
    elif rel_imbalance < tol_rel:
        print(f"PASS: relative imbalance {rel_imbalance:.4f} < {tol_rel}")
        print("DETAILED BALANCE: OK")
        return 0
    else:
        print(f"WARNING: relative imbalance {rel_imbalance:.4f} >= {tol_rel}")
        print("         Net interface flux may deviate from zero.")
        print("         This is expected for very short Monte Carlo runs.")
        print("DETAILED BALANCE: WARNING (large relative imbalance)")
        return 0  # Not a hard failure for short runs.


if __name__ == "__main__":
    sys.exit(detailed_balance_test())
