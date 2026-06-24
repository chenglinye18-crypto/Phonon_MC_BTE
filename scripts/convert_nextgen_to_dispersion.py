#!/usr/bin/env python3
"""Convert all nextgen MP bandstructure JSONs to project dispersion format.

Reads materials_data/mp_raw_nextgen/{label}_mp-*/bandstructure JSON,
computes q-path distance and finite-difference vg, writes
materials_data/processed/{label}/phonon_dispersion_{label}.txt

Usage::

    python scripts/convert_nextgen_to_dispersion.py
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
NEXTGEN = REPO / "materials_data" / "mp_raw_nextgen"
PROCESSED = REPO / "materials_data" / "processed"


def convert_one(label: str, bs_path: Path) -> Path | None:
    """Convert one nextgen BS JSON to project dispersion txt."""
    with open(bs_path) as f:
        data = json.load(f)

    freqs = data.get("frequencies", [])
    qpoints = data.get("qpoints", [])
    if not freqs or not qpoints:
        print(f"  {label}: empty frequencies or qpoints → skip")
        return None

    n_b = len(freqs)
    n_q = len(freqs[0])
    if n_q != len(qpoints):
        print(f"  {label}: freq/qpoint count mismatch ({n_q} vs {len(qpoints)})")

    # q-path distance in reciprocal coords → 1/m
    dist = [0.0]
    for i in range(1, len(qpoints)):
        dq = np.linalg.norm(np.array(qpoints[i]) - np.array(qpoints[i-1]))
        dist.append(dist[-1] + dq)
    q_dist = np.array(dist, dtype=np.float64) * 1e10  # → 1/m

    # Frequencies
    f_all = np.array(freqs, dtype=np.float64)  # (B, Nq) in THz
    f_min = float(np.min(f_all))
    f_max = float(np.max(f_all))

    # Clip small imaginary modes
    neg_count = int(np.sum(f_all < -0.1))
    f_all = np.maximum(f_all, 0.0)

    # Finite-difference vg
    vg_all = np.zeros_like(f_all)
    for b in range(n_b):
        for i in range(1, n_q - 1):
            dw = 2.0 * np.pi * 1e12 * (f_all[b, i+1] - f_all[b, i-1])
            dq_step = q_dist[i+1] - q_dist[i-1]
            if dq_step > 1e-30:
                vg_all[b, i] = dw / dq_step
        # Endpoints: one-sided
        if n_q >= 2:
            vg_all[b, 0] = vg_all[b, 1]
            vg_all[b, -1] = vg_all[b, -2]
    vg_all = np.clip(np.nan_to_num(vg_all, nan=0.0, posinf=0.0, neginf=0.0), 0, 20000)

    # Write project dispersion file
    out_dir = PROCESSED / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phonon_dispersion_{label}.txt"

    header = [
        f"# MP nextgen pheasy phonon bandstructure for {label}",
        f"# branch_names={','.join(chr(65+i) for i in range(n_b))}",
        f"# degeneracy={','.join(['1']*n_b)}",
        "# WARNING: vg is high-symmetry-path finite difference, NOT full-BZ transport vg.",
        "# WARNING: q_path in 1/m converted from reciprocal coordinates.",
        "# column: branch_id  q_path(1/m)  frequency_THz  vg_m_per_s",
    ]
    if neg_count > 0:
        header.append(f"# WARNING: {neg_count} imaginary frequencies clipped to 0 THz.")

    with open(out_path, "w") as f:
        f.write("\n".join(header) + "\n")
        for b in range(n_b):
            for i in range(n_q):
                f.write(f"{b+1}\t{q_dist[i]:.10e}\t{f_all[b,i]:.10f}\t{vg_all[b,i]:.6f}\n")

    vg_nonzero = vg_all[vg_all > 0]
    print(f"  {label}: {n_b} br × {n_q} q, "
          f"f=[{f_min:.2f}, {f_max:.2f}] THz, "
          f"vg=[{vg_all.min():.1f}, {vg_all.max():.1f}] m/s, "
          f"nz_vg={vg_nonzero.size}/{vg_all.size}"
          f"{' IMAG!' if neg_count else ''}")
    return out_path


def main() -> int:
    # Map: find nextgen BS files per material
    material_dirs = sorted(NEXTGEN.glob("*_mp-*"))
    if not material_dirs:
        print("No nextgen data found. Run mp_probe_phonon_nextgen.py first.")
        return 1

    converted = []
    for d in material_dirs:
        bs_files = sorted(d.glob("*_bandstructure_*.json"))
        if not bs_files:
            print(f"SKIP {d.name}: no bandstructure JSON")
            continue
        # Extract label from dir name: Cu_mp-30 → Cu
        label = d.name.split("_mp-")[0]
        print(f"Converting: {label} ({d.name})")
        for bs in bs_files:
            out = convert_one(label, bs)
            if out:
                converted.append((label, out))

    print(f"\nConverted {len(converted)} dispersion files:")
    for label, path in converted:
        print(f"  {label}: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
