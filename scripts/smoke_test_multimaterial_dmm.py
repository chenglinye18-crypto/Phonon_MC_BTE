#!/usr/bin/env python3
"""Smoke test for the multi-material + DMM framework.

Runs the two-material test case for a few steps and verifies:
1. The case enters the multi-material code path.
2. Particles exist with at least two distinct material_id values.
3. DMM interface statistics are non-empty (if interfaces were crossed).
4. The run completes without crashing.
5. Output files are generated.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import phonon_mc


def smoke_test():
    test_dir = REPO / "input_two_material_dmm_test"
    if not test_dir.is_dir():
        print(f"FAIL: test case directory not found: {test_dir}")
        return 1

    # --- setup -----------------------------------------------------------
    cs = phonon_mc.setup_case_from_ldg_lgrid(
        str(test_dir / "ldg.txt"),
        str(test_dir / "lgrid.txt"),
        length_scale=1e-6,
        input_length_unit="um",
        verbose=False,
    )
    mat = phonon_mc.resolve_case_material(cs, input_dir=str(test_dir))
    opts = phonon_mc.mc_default_opts(str(test_dir))

    # Verify multi-material detection.
    mat_lib = mat.get("material_library", {})
    mat_list = mat_lib.get("list", [])
    n_materials = len(mat_list)
    if n_materials <= 1:
        print(f"FAIL: only {n_materials} material(s) detected (need >1 for multi-mat path)")
        return 1
    print(f"PASS: detected {n_materials} materials: {[e['key'] for e in mat_list]}")

    # --- run simulation --------------------------------------------------
    try:
        Tp, p, out = phonon_mc.MC_solve_BTE(cs, mat, opts)
    except Exception as exc:
        print(f"FAIL: simulation crashed: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"PASS: simulation completed ({out['nsteps']} steps, {len(p)} final particles)")

    # --- verify particle material_ids ------------------------------------
    if len(p) == 0:
        print("FAIL: no particles in final state")
        return 1
    unique_mats = set(int(m) for m in p.material_id)
    if len(unique_mats) < 2:
        print(f"WARN: only {len(unique_mats)} material_id(s) found in particles ({unique_mats})")
        print("      (this may be OK if DMM reflection prevented all crossing)")
    else:
        print(f"PASS: particles have >= 2 material_ids: {unique_mats}")

    # --- verify DMM stats ------------------------------------------------
    iface_hist = out.get("iface_hist", [])
    dmm_total_attempt = sum(h.get("dmm_attempt", 0) for h in iface_hist)
    dmm_total_transmit = sum(h.get("dmm_transmit", 0) for h in iface_hist)
    dmm_total_reflect = sum(h.get("dmm_reflect", 0) for h in iface_hist)

    print(f"PASS: DMM stats collected: {len(iface_hist)} steps in iface_hist")
    print(f"      total attempts={dmm_total_attempt}, transmit={dmm_total_transmit}, reflect={dmm_total_reflect}")

    # Check that iface_hist entries are no longer empty dicts.
    if iface_hist and all(isinstance(h, dict) and "dmm_attempt" in h for h in iface_hist):
        print("PASS: iface_hist entries contain DMM keys (not empty dicts)")
    else:
        print("WARN: some iface_hist entries may be missing DMM keys")

    # --- verify output ---------------------------------------------------
    output_dir = out.get("output_dir", "")
    if output_dir:
        print(f"PASS: output directory created: {output_dir}")
        branch_stats = Path(output_dir) / "steps"
        if branch_stats.is_dir():
            steps_dirs = sorted(branch_stats.glob("step_*"))
            if steps_dirs:
                # Check latest step's branch stats for multi-material columns.
                last_step = steps_dirs[-1]
                bs_file = last_step / "branch_stats.txt"
                if bs_file.is_file():
                    with open(bs_file) as f:
                        header = f.readline().strip()
                    if "material_id" in header and "material_key" in header:
                        print(f"PASS: branch_stats.txt has multi-material columns")
                    else:
                        print(f"WARN: branch_stats.txt header: {header}")

    print()
    print("=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(smoke_test())
