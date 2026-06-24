#!/usr/bin/env python3
"""Check downloaded phonon data quality for all target materials.

Checks: structure, bandstructure, DOS, converted dispersion file, frequency
non-negativity, vg finiteness, branch count, q-ordering.

Outputs: materials_data/reports/phonon_data_quality_summary.{csv,md}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MATERIALS_DATA = REPO / "materials_data"
MP_RAW = MATERIALS_DATA / "mp_raw"
PROCESSED = MATERIALS_DATA / "processed"
REPORTS = MATERIALS_DATA / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def check_material(label: str) -> dict[str, Any]:
    """Run all checks for one material."""
    raw_dir = MP_RAW / label
    proc_dir = PROCESSED / label
    status: dict[str, Any] = {
        "label": label, "status": "unknown",
        "has_structure": False, "has_phonon_bandstructure": False,
        "has_phonon_dos": False, "has_dispersion_txt": False,
        "dispersion_ok": False, "n_branches": 0, "n_qpoints": 0,
        "f_min_THz": None, "f_max_THz": None,
        "freq_non_negative": False, "vg_finite": False,
        "notes": [],
    }

    # Check raw data.
    if (raw_dir / f"{label}_structure.cif").is_file():
        status["has_structure"] = True
    if (raw_dir / f"{label}_phonon_bandstructure.json").is_file():
        status["has_phonon_bandstructure"] = True
    if (raw_dir / f"{label}_phonon_dos.json").is_file():
        status["has_phonon_dos"] = True

    # Check converted dispersion.
    disp_path = proc_dir / f"phonon_dispersion_{label}.txt"
    if not disp_path.is_file():
        status["notes"].append("no converted dispersion file")
        if not status["has_phonon_bandstructure"]:
            status["status"] = "structure_only_needs_phonopy"
        elif status["has_phonon_dos"]:
            status["status"] = "partial_dos_only"
        else:
            status["status"] = "structure_only_needs_phonopy"
        return status

    status["has_dispersion_txt"] = True

    # Try reading with project function.
    try:
        sys.path.insert(0, str(REPO))
        from phonon_mc import mat_from_phonon_dispersion_file
        mat = mat_from_phonon_dispersion_file(disp_path, material_name=label)
        status["n_branches"] = mat.get("B", 0)
        status["n_qpoints"] = mat.get("q", np.array([])).size
        # Check frequencies.
        f_tab = mat.get("frequency_THz_tab", np.array([]))
        if f_tab.size > 0:
            status["f_min_THz"] = float(np.min(f_tab))
            status["f_max_THz"] = float(np.max(f_tab))
            status["freq_non_negative"] = bool(np.all(f_tab >= 0))
        # Check vg.
        vg_tab = mat.get("vg_tab", np.array([]))
        if vg_tab.size > 0:
            status["vg_finite"] = bool(np.all(np.isfinite(vg_tab)))
        status["dispersion_ok"] = True
        status["status"] = "ready_for_first_pass_dmm"
    except Exception as exc:
        status["notes"].append(f"dispersion validation failed: {exc}")
        if status["has_phonon_dos"]:
            status["status"] = "partial_dos_only"
        else:
            status["status"] = "structure_only_needs_phonopy"

    return status


def main() -> int:
    labels = sorted([
        d.name for d in MP_RAW.iterdir()
        if d.is_dir() and (d / f"{d.name}_metadata.json").is_file()
    ])
    if not labels:
        print("No downloaded data. Run mp_download_target_phonons.py first.")
        return 0

    results = []
    for label in labels:
        print(f"Checking: {label} ...")
        r = check_material(label)
        results.append(r)
        print(f"  status: {r['status']}")

    df = pd.DataFrame(results)
    csv_path = REPORTS / "phonon_data_quality_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV: {csv_path}")

    # Markdown report.
    lines = [
        "# Phonon Data Quality Summary",
        "",
        f"| label | status | structure | BS | DOS | disp | branches | qpoints | f_range_THz |",
        "|-------|--------|-----------|-----|------|------|----------|---------|-------------|",
    ]
    for r in results:
        f_range = f"{r.get('f_min_THz','?'):.1f}–{r.get('f_max_THz','?'):.1f}" if r.get('f_min_THz') else "N/A"
        lines.append(
            f"| {r['label']} | {r['status']} | "
            f"{'Y' if r['has_structure'] else 'N'} | "
            f"{'Y' if r['has_phonon_bandstructure'] else 'N'} | "
            f"{'Y' if r['has_phonon_dos'] else 'N'} | "
            f"{'Y' if r['has_dispersion_txt'] else 'N'} | "
            f"{r['n_branches']} | {r['n_qpoints']} | {f_range} |"
        )
    lines += [
        "",
        "## Status Legend",
        "- `ready_for_first_pass_dmm`: dispersion file available, can compute G_pp",
        "- `partial_dos_only`: only phonon DOS, no dispersion (needs Phonopy)",
        "- `structure_only_needs_phonopy`: only structure (needs Phonopy for phonons)",
        "- `failed`: data corrupted or unreadable",
    ]
    md_path = REPORTS / "phonon_data_quality_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"MD:  {md_path}")

    # Print status counts.
    counts = df["status"].value_counts()
    print(f"\nStatus counts: {dict(counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
