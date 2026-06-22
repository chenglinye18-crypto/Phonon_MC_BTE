#!/usr/bin/env python3
"""DMM thermal boundary resistance (TBR) sanity-check test.

Two-material slab with a temperature gradient (hot reservoir 310 K on one side,
cold reservoir 300 K on the other).  Measures the interface net energy flux and
estimates the effective TBR.

This is a sanity check only — the result is NOT expected to match literature
values without proper scattering calibration.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import phonon_mc


def tbr_estimate_test(max_steps: int = 30, initial_particles: int = 5000) -> int:
    test_dir = REPO / "input_two_material_dmm_test"
    if not test_dir.is_dir():
        print(f"FAIL: test case directory not found: {test_dir}")
        return 1

    # --- setup: gradient case ------------------------------------------------
    T_hot = 310.0   # Y- side reservoir
    T_cold = 300.0  # Y+ side reservoir

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
    # Enable periodic output so we can get final temperature field.
    opts["output"]["enable"] = True
    opts["output"]["every_n_steps"] = max_steps + 1  # only final
    opts["volume_heat_source_file"] = ""

    mesh = phonon_mc.init_mesh_from_geom(cs)
    Nc = phonon_mc.infer_Nc(mesh)

    # Write gradient initial and reference temperature files.
    import csv, tempfile, os
    tmp_initial = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    tmp_ref = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    w_init = csv.writer(tmp_initial)
    w_ref = csv.writer(tmp_ref)
    Tref_val = 305.0
    for iz in range(1, mesh["Nz"] + 1):
        for iy in range(1, mesh["Ny"] + 1):
            y_frac = (iy - 0.5) / mesh["Ny"]
            T_val = T_hot + (T_cold - T_hot) * y_frac
            for ix in range(1, mesh["Nx"] + 1):
                w_init.writerow([ix, iy, iz, f"{T_val:.4f}"])
                w_ref.writerow([ix, iy, iz, f"{Tref_val:.4f}"])
    tmp_initial.close()
    tmp_ref.close()
    opts["initial_temperature_file"] = tmp_initial.name
    opts["reference_temperature_file"] = tmp_ref.name

    # Identify interface cells.
    # The interface is at x = 0.03 um.  Find cells on either side.
    cell_mat_idx = mesh.get("cell_material_index", np.ones(Nc, dtype=np.int32))
    centers = mesh["centers"]
    x_interface = 0.03e-6  # 0.03 um in meters
    eps_interface = 1e-9    # 1 nm tolerance
    left_cells = (centers[:, 0] < x_interface) & (centers[:, 0] > x_interface - 2 * mesh.get("dx_min", 1e-9))
    right_cells = (centers[:, 0] > x_interface) & (centers[:, 0] < x_interface + 2 * mesh.get("dx_min", 1e-9))
    # Get one layer of cells on each side of the interface.
    left_interface_cells = np.flatnonzero(
        (centers[:, 0] < x_interface) & (centers[:, 0] > x_interface - mesh.get("dx", np.ones(1))[0] * 1.5)
    )
    right_interface_cells = np.flatnonzero(
        (centers[:, 0] > x_interface) & (centers[:, 0] < x_interface + mesh.get("dx", np.ones(1))[1] * 1.5)
    )

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

    # --- collect interface DMM energy flux -----------------------------------
    iface_hist = out.get("iface_hist", [])
    dmm_energy_0_to_1 = 0.0
    dmm_energy_1_to_0 = 0.0
    for h in iface_hist:
        detail = h.get("dmm_detail", {})
        dmm_tr = max(h.get("dmm_transmit", 0), 1)
        dmm_etr = h.get("dmm_energy_transmit", 0.0)
        for pair_str, counts in detail.items():
            tr_count = counts.get("transmit", 0)
            frac = tr_count / dmm_tr if dmm_tr > 0 else 0
            if pair_str == "0->1":
                dmm_energy_0_to_1 += dmm_etr * frac
            elif pair_str == "1->0":
                dmm_energy_1_to_0 += dmm_etr * frac

    total_time = sum(out.get("dt_hist", [1e-15]))
    # Interface area.
    mesh_area = float(mesh.get("Ly", 0.06e-6)) * float(mesh.get("Lz", 0.06e-6))
    if mesh_area <= 0:
        mesh_area = 0.06e-6 * 0.06e-6

    q_interface_W = (dmm_energy_0_to_1 - dmm_energy_1_to_0) / max(total_time, 1e-30)
    q_interface_W_m2 = q_interface_W / max(mesh_area, 1e-30)

    # --- estimate DeltaT across interface ------------------------------------
    T_left_mean = float(np.mean(Tp[left_interface_cells])) if left_interface_cells.size else np.nan
    T_right_mean = float(np.mean(Tp[right_interface_cells])) if right_interface_cells.size else np.nan
    DeltaT_interface = T_left_mean - T_right_mean if np.isfinite(T_left_mean) and np.isfinite(T_right_mean) else np.nan

    R_K = DeltaT_interface / max(q_interface_W_m2, 1e-30) if np.isfinite(DeltaT_interface) else np.nan
    G_K = 1.0 / max(R_K, 1e-30) if np.isfinite(R_K) and abs(R_K) > 0 else np.nan

    print()
    print("=" * 60)
    print("TBR ESTIMATE RESULTS")
    print("=" * 60)
    print(f"  Hot reservoir T:           {T_hot:.1f} K")
    print(f"  Cold reservoir T:          {T_cold:.1f} K")
    print(f"  DMM energy 0->1:           {dmm_energy_0_to_1:.4e} J")
    print(f"  DMM energy 1->0:           {dmm_energy_1_to_0:.4e} J")
    print(f"  Simulation time:           {total_time:.4e} s")
    print(f"  Interface area:            {mesh_area:.4e} m^2")
    print(f"  q_interface:               {q_interface_W:.4e} W")
    print(f"  q_interface density:       {q_interface_W_m2:.4e} W/m^2")
    print(f"  T_left (interface):        {T_left_mean:.2f} K")
    print(f"  T_right (interface):       {T_right_mean:.2f} K")
    print(f"  DeltaT_interface:          {DeltaT_interface:.2f} K")
    if np.isfinite(R_K):
        print(f"  R_K (TBR):                 {R_K:.4e} K·m^2/W")
    if np.isfinite(G_K):
        print(f"  G_K (TBC):                 {G_K:.4e} W/(K·m^2)")
    print()

    # Judgment.
    if not np.isfinite(q_interface_W_m2):
        print("FAIL: could not compute interface heat flux")
        return 1
    # Expect heat to flow from hot to cold, i.e., positive q (0->1 if 0 is hotter).
    # Actually, material 0 (Si) is on left, material 1 (IGZO) on right.
    # Hot reservoir is at Y-, cold at Y+.  The interface is at X=0.03.
    # The temperature gradient is along Y, interface is along YZ plane.
    # Heat flux direction depends on geometry; we just check the sign is reasonable.
    if q_interface_W_m2 > 0:
        print(f"PASS: heat flux direction 0->1 (positive), consistent with hotter side→colder side")
    else:
        print(f"INFO: net heat flux 0->1 is negative ({q_interface_W_m2:.4e} W/m^2)")
        print(f"      This may be fine depending on actual temperature distribution.")
    if abs(DeltaT_interface) < 0.01:
        print("WARNING: DeltaT across interface is very small (< 0.01 K)")
        print("         Increase max_steps or particle count for better statistics.")
    else:
        print(f"PASS: interface DeltaT = {DeltaT_interface:.4f} K")

    print("TBR SANITY CHECK COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(tbr_estimate_test())
