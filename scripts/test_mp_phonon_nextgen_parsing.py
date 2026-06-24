#!/usr/bin/env python3
"""Test next-gen MP phonon JSON parsing using already-downloaded data.

Does NOT require network.  Uses Cu mp-30 nextgen JSON files.

Usage::

    python scripts/test_mp_phonon_nextgen_parsing.py
"""

import json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
NEXTGEN_DIR = REPO / "materials_data" / "mp_raw_nextgen"

FAILS = 0

def check(cond, msg):
    global FAILS
    if cond: print(f"  PASS: {msg}")
    else: print(f"  FAIL: {msg}"); FAILS += 1

def main():
    cu_dir = NEXTGEN_DIR / "Cu_mp-30"
    bs_file = cu_dir / "Cu_mp-30_phonon_bandstructure_pheasy_latimer_munro.json"
    dos_file = cu_dir / "Cu_mp-30_phonon_dos_pheasy.json"

    if not bs_file.is_file():
        print(f"ERROR: {bs_file} not found.  Run mp_probe_phonon_nextgen.py first.")
        print("  python scripts/mp_probe_phonon_nextgen.py --material-id mp-30 --label Cu")
        return 1

    print("=== Test 1: Read bandstructure JSON ===")
    with open(bs_file) as f:
        bs = json.load(f)
    check("frequencies" in bs, f"frequencies key present (keys: {list(bs.keys())[:8]})")

    freqs = bs.get("frequencies", [])
    check(len(freqs) > 0, f"n_branches = {len(freqs)}")
    check(len(freqs[0]) > 0, f"n_qpoints = {len(freqs[0])}")

    qp = bs.get("qpoints", [])
    check(len(qp) == len(freqs[0]), f"qpoints count matches: {len(qp)}")

    labels = bs.get("labels_dict", {})
    check(len(labels) > 0, f"labels_dict has {len(labels)} entries: {list(labels.keys())[:5]}")

    print("\n=== Test 2: Generate q-path from qpoints ===")
    dist = [0.0]
    for i in range(1, len(qp)):
        dq = np.linalg.norm(np.array(qp[i]) - np.array(qp[i-1]))
        dist.append(dist[-1] + dq)
    q_dist = np.array(dist) * 1e10  # → 1/m
    check(np.all(np.diff(q_dist) >= 0), f"q-path monotonic, range [{q_dist[0]:.2e}, {q_dist[-1]:.2e}] 1/m")

    print("\n=== Test 3: Extract frequencies ===")
    freq_all = np.array(freqs)
    check(np.all(np.isfinite(freq_all)), "all frequencies finite")
    check(np.min(freq_all) > -0.5, f"min freq = {np.min(freq_all):.4f} THz (> -0.5)")
    check(np.max(freq_all) < 20, f"max freq = {np.max(freq_all):.4f} THz (< 20)")

    print("\n=== Test 4: Estimate vg ===")
    vg = np.zeros_like(freq_all)
    for b in range(len(freqs)):
        for i in range(1, len(qp)-1):
            dw = 2*np.pi*1e12*(freq_all[b,i+1]-freq_all[b,i-1])
            dq_step = q_dist[i+1]-q_dist[i-1]
            vg[b,i] = dw/max(dq_step,1e-30)
    vg = np.nan_to_num(vg, nan=0, posinf=0, neginf=0)
    check(np.all(np.isfinite(vg)), "all vg finite")
    check(np.max(np.abs(vg)) < 20000, f"max |vg| = {np.max(np.abs(vg)):.0f} m/s (< 20000)")

    print("\n=== Test 5: Generate project-format dispersion ===")
    out_path = REPO / "materials_data" / "processed" / "Cu" / "phonon_dispersion_Cu_nextgen_test.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# test dispersion for Cu mp-30\n")
        f.write(f"# branch_names={','.join(['B'+str(i+1) for i in range(len(freqs))])}\n")
        f.write("# column: branch_id, q_path, frequency_THz, vg_m_per_s\n")
        for b in range(len(freqs)):
            for i in range(len(qp)):
                f.write(f"{b+1}\t{q_dist[i]:.10e}\t{freq_all[b,i]:.10f}\t{vg[b,i]:.6f}\n")
    check(out_path.is_file(), f"dispersion file generated: {out_path}")
    check(out_path.stat().st_size > 100, f"file size = {out_path.stat().st_size} bytes (> 100)")

    print("\n=== Test 6: DOS ===")
    if dos_file.is_file():
        with open(dos_file) as f:
            dos = json.load(f)
        dos_freqs = dos.get("frequencies", [])
        check(len(dos_freqs) > 0, f"DOS: {len(dos_freqs)} frequency points")
    else:
        print(f"  SKIP: {dos_file} not found")

    print()
    if FAILS == 0:
        print("ALL TESTS PASSED")
        return 0
    print(f"{FAILS} TEST(S) FAILED")
    return 1

if __name__ == "__main__":
    sys.exit(main())
