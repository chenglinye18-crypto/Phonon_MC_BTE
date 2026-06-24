#!/usr/bin/env python3
"""Convert MP phonon data to project dispersion format.

Reads downloaded MP raw data, extracts phonon bandstructure and DOS,
and writes ``phonon_dispersion_{label}.txt`` files in the project format.

Materials without MP phonon data are marked as ``needs_phonopy``.

Usage::

    python scripts/mp_convert_phonons_to_project_format.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MATERIALS_DATA = REPO / "materials_data"
MP_RAW = MATERIALS_DATA / "mp_raw"
PROCESSED = MATERIALS_DATA / "processed"


def convert_bandstructure(label: str) -> dict[str, Any]:
    """Try to convert MP phonon bandstructure to project dispersion txt."""
    bs_path = MP_RAW / label / f"{label}_phonon_bandstructure.json"
    meta: dict[str, Any] = {
        "label": label,
        "has_bandstructure": False,
        "has_dos": False,
        "generated_dispersion_txt": False,
        "n_branches": 0,
        "n_qpoints": 0,
        "frequency_min_THz": None,
        "frequency_max_THz": None,
        "vg_min_m_s": None,
        "vg_max_m_s": None,
        "warnings": [],
        "cannot_generate_project_dispersion": False,
        "needs_phonopy_for_vg": False,
    }

    if not bs_path.is_file():
        meta["warnings"].append("no phonon bandstructure JSON")
        meta["cannot_generate_project_dispersion"] = True
        meta["needs_phonopy_for_vg"] = True
        return meta

    try:
        with open(bs_path) as f:
            data = json.load(f)
    except Exception as exc:
        meta["warnings"].append(f"failed to read bandstructure JSON: {exc}")
        meta["cannot_generate_project_dispersion"] = True
        return meta

    meta["has_bandstructure"] = True

    # Extract branches and qpoints.
    branches = data.get("branches", [])
    qpoints = data.get("qpoints", [])
    if not branches:
        meta["warnings"].append("empty branches in bandstructure")
        meta["cannot_generate_project_dispersion"] = True
        return meta

    n_branches = len(branches)
    n_q = len(qpoints)
    meta["n_branches"] = n_branches
    meta["n_qpoints"] = n_q

    # Compute q-path distances.
    dist = [0.0]
    for i in range(1, n_q):
        dq = np.linalg.norm(np.array(qpoints[i]) - np.array(qpoints[i-1]))
        dist.append(dist[-1] + dq)
    q_dist = np.array(dist, dtype=np.float64)

    # Frequencies and vg estimates.
    freq_all = np.array(branches, dtype=np.float64)  # (n_branches, n_q) in THz
    meta["frequency_min_THz"] = float(np.min(freq_all))
    meta["frequency_max_THz"] = float(np.max(freq_all))

    # Group velocity: finite difference along path.
    vg_all = np.zeros_like(freq_all)
    for b in range(n_branches):
        for i in range(1, n_q - 1):
            dq_step = max(q_dist[i] - q_dist[i-1], 1e-30)
            dw = 2 * np.pi * 1e12 * (freq_all[b, i+1] - freq_all[b, i-1])
            dq = q_dist[i+1] - q_dist[i-1]
            vg_all[b, i] = dw / max(dq, 1e-30) if dq > 0 else 0.0
        # Endpoints: one-sided.
        if n_q >= 2:
            dw0 = 2 * np.pi * 1e12 * (freq_all[b, 1] - freq_all[b, 0])
            dq0 = max(q_dist[1] - q_dist[0], 1e-30)
            vg_all[b, 0] = dw0 / dq0
            dwn = 2 * np.pi * 1e12 * (freq_all[b, -1] - freq_all[b, -2])
            dqn = max(q_dist[-1] - q_dist[-2], 1e-30)
            vg_all[b, -1] = dwn / dqn

    vg_all = np.nan_to_num(vg_all, nan=0.0, posinf=0.0, neginf=0.0)
    meta["vg_min_m_s"] = float(np.min(vg_all))
    meta["vg_max_m_s"] = float(np.max(vg_all))

    # Write project-format dispersion file.
    out_dir = PROCESSED / label
    out_dir.mkdir(parents=True, exist_ok=True)
    disp_path = out_dir / f"phonon_dispersion_{label}.txt"

    header_lines = [
        f"# MP phonon bandstructure for {label}",
        f"# branch_names={','.join(['B'+str(i+1) for i in range(n_branches)])}",
        f"# degeneracy={','.join(['1']*n_branches)}",
        "# WARNING: vg estimated from high-symmetry path finite differences.",
        "# WARNING: MP high-symmetry path distance may not provide full SI q units.",
        "# WARNING: vg is a first-pass approximation, not full-BZ group velocity.",
        "# column: branch_id, q_path (1/m), frequency_THz, vg_m_per_s",
    ]
    with open(disp_path, "w") as f:
        f.write("\n".join(header_lines) + "\n")
        for b in range(n_branches):
            for i in range(n_q):
                f.write(f"{b+1}\t{q_dist[i]:.10e}\t{freq_all[b,i]:.10f}\t{vg_all[b,i]:.6f}\n")

    meta["generated_dispersion_txt"] = True
    meta["warnings"].append(
        "vg is estimated from high-symmetry path finite differences "
        "and is only a first-pass approximation, not full-BZ group velocity."
    )
    meta["warnings"].append(
        "MP high-symmetry path distance may not provide full SI q units; "
        "group velocity is approximate."
    )
    print(f"    Generated: {disp_path} ({n_branches} branches, {n_q} q-points)")
    return meta


def convert_dos(label: str, meta: dict[str, Any]) -> dict[str, Any]:
    """Extract phonon DOS to CSV."""
    dos_path = MP_RAW / label / f"{label}_phonon_dos.json"
    if not dos_path.is_file():
        meta["warnings"].append("no phonon DOS JSON")
        return meta

    try:
        with open(dos_path) as f:
            data = json.load(f)
    except Exception as exc:
        meta["warnings"].append(f"failed to read DOS JSON: {exc}")
        return meta

    meta["has_dos"] = True
    freq = np.array(data.get("frequencies", []), dtype=np.float64)
    dos_vals = np.array(data.get("densities", []), dtype=np.float64)

    out_dir = PROCESSED / label
    out_dir.mkdir(parents=True, exist_ok=True)
    dos_out = out_dir / f"phonon_dos_{label}.csv"
    with open(dos_out, "w") as f:
        f.write("frequency_THz,dos\n")
        for fi, di in zip(freq, dos_vals):
            f.write(f"{fi:.10f},{di:.10e}\n")
    print(f"    Generated DOS: {dos_out}")
    return meta


def main() -> int:
    # Find all downloaded labels.
    labels = sorted([
        d.name for d in MP_RAW.iterdir()
        if d.is_dir() and (d / f"{d.name}_metadata.json").is_file()
    ])
    if not labels:
        print("No downloaded data found.  Run mp_download_target_phonons.py first.")
        return 0

    all_meta = {}
    for label in labels:
        print(f"\nConverting: {label}")
        meta = convert_bandstructure(label)
        meta = convert_dos(label, meta)

        # Save processing metadata.
        out_dir = PROCESSED / label
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / f"{label}_phonon_processing_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        all_meta[label] = meta

    # Summary.
    print(f"\n{'='*50}")
    print("Conversion summary:")
    for label, meta in all_meta.items():
        disp = "disp" if meta.get("generated_dispersion_txt") else "no-disp"
        dos = "DOS" if meta.get("has_dos") else "no-DOS"
        needs = "NEEDS_PHONOPY" if meta.get("needs_phonopy_for_vg") else "OK"
        print(f"  {label:10s}  {disp:7s}  {dos:6s}  {needs}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
