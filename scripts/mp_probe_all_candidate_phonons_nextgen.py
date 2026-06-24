#!/usr/bin/env python3
"""Batch-scan all MP candidates for phonon availability via next-gen API.

Reads materials_data/mp_candidates/{label}_candidates.csv, probes each
material_id using mpr.materials.phonon, and writes a new selection file.

Usage::

    export MP_API_KEY="your_key"
    python scripts/mp_probe_all_candidate_phonons_nextgen.py
"""

from __future__ import annotations

import json, os, sys, csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CANDIDATES_DIR = REPO / "materials_data" / "mp_candidates"
REPORTS = REPO / "materials_data" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def probe_availability(mpr, material_id: str) -> dict[str, Any]:
    """Lightweight probe: check which methods/path_types have data."""
    info = {
        "material_id": material_id,
        "has_pheasy_doc": False, "has_dfpt_doc": False,
        "has_bandstructure": False, "has_dos": False, "has_forceconstants": False,
        "available_methods": [], "available_path_types": [],
        "recommended_method": "", "recommended_path_type": "",
        "n_branches": 0, "n_qpoints": 0,
        "frequency_min": None, "frequency_max": None,
        "error_summary": "",
    }
    errors = []

    for method in ["pheasy", "dfpt", None]:
        method_str = method or "default"
        try:
            kwargs = {"material_ids": [material_id]}
            if method is not None:
                kwargs["phonon_method"] = method
            docs = mpr.materials.phonon.search(**kwargs)
            if docs:
                if method == "pheasy":
                    info["has_pheasy_doc"] = True
                elif method == "dfpt":
                    info["has_dfpt_doc"] = True
                info["available_methods"].append(method_str)
                if not info["recommended_method"]:
                    info["recommended_method"] = method_str
        except Exception:
            pass

    if not info["available_methods"]:
        info["error_summary"] = "no phonon data"
        return info

    # Try bandstructure with preferred method
    method = info["recommended_method"]
    if method == "default":
        method = None
    for pt in ["latimer_munro", "setyawan_curtarolo"]:
        try:
            bs = mpr.materials.phonon.get_bandstructure_from_material_id(
                material_id=material_id, phonon_method=method, path_type=pt
            )
            if bs is not None:
                info["has_bandstructure"] = True
                info["available_path_types"].append(pt)
                if not info["recommended_path_type"]:
                    info["recommended_path_type"] = pt
                # Extract branch info
                try:
                    d = bs.as_dict()
                    freq_data = d.get("frequencies", d.get("branches", []))
                    info["n_branches"] = len(freq_data)
                    info["n_qpoints"] = len(d.get("qpoints", []))
                    if freq_data:
                        all_f = [f for b in freq_data for f in (b if isinstance(b, list) else [])]
                        if all_f:
                            info["frequency_min"] = float(np.min(all_f))
                            info["frequency_max"] = float(np.max(all_f))
                except Exception:
                    pass
                break
        except Exception as e:
            errors.append(f"BS({pt}): {e}")

    # Try DOS
    try:
        dos = mpr.materials.phonon.get_dos_from_material_id(
            material_id=material_id, phonon_method=method
        )
        if dos is not None:
            info["has_dos"] = True
    except Exception:
        pass

    # Try force constants (pheasy only)
    if info["has_pheasy_doc"]:
        try:
            fc = mpr.materials.phonon.get_forceconstants_from_material_id(
                material_id=material_id, phonon_method="pheasy"
            )
            if fc is not None:
                info["has_forceconstants"] = True
        except Exception:
            pass

    info["error_summary"] = "; ".join(errors[:3]) if errors else ""
    return info


def main() -> int:
    api_key = os.environ.get("MP_API_KEY", "")
    if not api_key:
        print("ERROR: MP_API_KEY not set.  export MP_API_KEY='your_key'")
        return 1
    try:
        from mp_api.client import MPRester
    except ImportError:
        print("ERROR: pip install -U mp_api pymatgen monty")
        return 1

    mpr = MPRester(api_key=api_key)

    # Read candidate CSVs
    candidate_files = sorted(CANDIDATES_DIR.glob("*_candidates.csv"))
    if not candidate_files:
        print("No candidate CSVs found. Run mp_search_target_materials.py first.")
        return 1

    all_rows = []
    for cf in candidate_files:
        label = cf.stem.replace("_candidates", "")
        print(f"\nProbing: {label}")
        df = pd.read_csv(cf)
        if "material_id" not in df.columns:
            print(f"  SKIP: no material_id column")
            continue

        mids = df["material_id"].dropna().unique()
        # Limit to top candidates per formula (sorted by energy_above_hull if available)
        if "energy_above_hull" in df.columns:
            df_sorted = df.sort_values("energy_above_hull")
            mids = df_sorted["material_id"].dropna().unique()
        mids = mids[:min(len(mids), 3)]  # Only top 3 per formula for speed
        for mid in mids:
            mid_str = str(mid)
            print(f"  {mid_str} ...", end=" ")
            info = probe_availability(mpr, mid_str)
            has = "BS" if info["has_bandstructure"] else ("DOS" if info["has_dos"] else "struct")
            print(f"{has} methods={info['available_methods']} paths={info['available_path_types']}")
            row = {
                "label": label, **info,
                **{c: df[df["material_id"] == mid].iloc[0][c]
                   for c in ["formula_pretty","energy_above_hull","is_stable"]
                   if c in df.columns},
            }
            all_rows.append(row)
            # Incremental save every 3 materials
            if len(all_rows) % 3 == 0:
                pd.DataFrame(all_rows).to_csv(
                    REPORTS / "phonon_availability_nextgen_summary.csv", index=False
                )

    if not all_rows:
        print("No candidates probed.")
        return 1

    result_df = pd.DataFrame(all_rows)

    # Save reports
    csv_path = REPORTS / "phonon_availability_nextgen_summary.csv"
    result_df.to_csv(csv_path, index=False)

    # Markdown
    lines = ["# Phonon Availability (Next-Gen API)", "",
             f"| label | material_id | formula | e_above_hull | stable | BS | DOS | FC | methods | paths | branches | qpoints | f_range_THz |",
             "|-------|-------------|---------|-------------|--------|-----|------|-----|---------|-------|----------|---------|-------------|"]
    for _, r in result_df.iterrows():
        fr = f"{r.get('frequency_min','?'):.1f}–{r.get('frequency_max','?'):.1f}" if r.get('frequency_min') else "N/A"
        lines.append(
            f"| {r['label']} | {r['material_id']} | {r.get('formula_pretty','')} | "
            f"{r.get('energy_above_hull','')} | {r.get('is_stable','')} | "
            f"{'Y' if r['has_bandstructure'] else 'N'} | "
            f"{'Y' if r['has_dos'] else 'N'} | "
            f"{'Y' if r.get('has_forceconstants') else 'N'} | "
            f"{','.join(r.get('available_methods',[]))} | "
            f"{','.join(r.get('available_path_types',[]))} | "
            f"{r['n_branches']} | {r['n_qpoints']} | {fr} |"
        )
    md_path = REPORTS / "phonon_availability_nextgen_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    # Write selection file
    sel_lines = [
        "# AUTO-GENERATED material selection with phonon availability (next-gen API).",
        "# Review before downloading.  Prefer has_bandstructure + has_dos.",
        f"# Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    for label in result_df["label"].unique():
        sub = result_df[result_df["label"] == label]
        best = sub[sub["has_bandstructure"] & sub["has_dos"]]
        if len(best) == 0:
            best = sub[sub["has_bandstructure"]]
        if len(best) == 0:
            best = sub[sub["has_dos"]]
        if len(best) == 0:
            best = sub
        # Pick lowest e_above_hull
        if "energy_above_hull" in best.columns:
            best = best.sort_values("energy_above_hull")
        top = best.iloc[0]
        sel_lines.append(f"[selected.{label}]")
        sel_lines.append(f"material_id = \"{top['material_id']}\"")
        sel_lines.append(f"phonon_method = \"{top.get('recommended_method', 'pheasy')}\"")
        sel_lines.append(f"path_type = \"{top.get('recommended_path_type', 'latimer_munro')}\"")
        sel_lines.append(f"# has_BS={top['has_bandstructure']} has_DOS={top['has_dos']} has_FC={top.get('has_forceconstants',False)}")
        sel_lines.append("")

    sel_path = REPO / "materials_data" / "selected_materials_with_phonons_nextgen.toml"
    sel_path.write_text("\n".join(sel_lines), encoding="utf-8")
    print(f"\nSelection: {sel_path}")
    print(f"CSV: {csv_path}")
    print(f"MD:  {md_path}")

    # Quick summary
    has_bs = result_df["has_bandstructure"].sum()
    has_dos = result_df["has_dos"].sum()
    print(f"\nSummary: {has_bs} with BS, {has_dos} with DOS, {len(result_df)} total probed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
