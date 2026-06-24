#!/usr/bin/env python3
"""Search Materials Project for target material candidates.

Reads ``materials_data/material_targets.toml``, queries MP for each formula,
saves candidate tables, and writes ``materials_data/selected_materials.toml``
with auto-recommended mp-ids for human review.

Requires: ``pip install mp_api pymatgen``
API key:  set environment variable ``MP_API_KEY`` before running.

Usage::

    export MP_API_KEY="your_key"
    python scripts/mp_search_target_materials.py
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MATERIALS_DATA = REPO / "materials_data"


def get_mp_client():
    api_key = os.environ.get("MP_API_KEY", "")
    if not api_key:
        print("ERROR: MP_API_KEY environment variable not set.")
        print("  export MP_API_KEY='your_key'")
        print("  Obtain a key from https://materialsproject.org/api")
        sys.exit(1)
    try:
        from mp_api.client import MPRester
    except ImportError:
        print("ERROR: mp_api not installed.  Run: pip install mp_api pymatgen")
        sys.exit(1)
    return MPRester(api_key)


def search_material(mp: Any, formula: str, label: str) -> pd.DataFrame:
    """Search MP for a formula and return a candidate DataFrame."""
    print(f"  Searching MP for {formula} ({label}) ...")
    try:
        docs = mp.materials.summary.search(
            formula=formula,
            fields=[
                "material_id", "formula_pretty", "energy_above_hull",
                "is_stable", "symmetry", "density", "volume",
                "band_gap", "has_props", "theoretical",
            ],
        )
    except Exception as exc:
        print(f"    WARNING: MP search failed for {formula}: {exc}")
        return pd.DataFrame()

    rows = []
    for d in docs:
        sym = d.symmetry or {}
        rows.append({
            "material_id": d.material_id,
            "formula_pretty": d.formula_pretty,
            "energy_above_hull": getattr(d, "energy_above_hull", None),
            "is_stable": getattr(d, "is_stable", None),
            "spacegroup_symbol": getattr(sym, "symbol", None),
            "spacegroup_number": getattr(sym, "number", None),
            "crystal_system": getattr(sym, "crystal_system", None),
            "density": getattr(d, "density", None),
            "volume": getattr(d, "volume", None),
            "band_gap": getattr(d, "band_gap", None),
            "has_props": list(d.has_props) if d.has_props else [],
            "theoretical": getattr(d, "theoretical", None),
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        print(f"    WARNING: no results for {formula}")
        return df

    # Sort by energy_above_hull (NaN last), then stable first.
    df["_sort_eah"] = df["energy_above_hull"].fillna(999)
    df["_sort_stable"] = (~df["is_stable"].fillna(False)).astype(int)
    df = df.sort_values(["_sort_stable", "_sort_eah"]).drop(columns=["_sort_eah", "_sort_stable"])
    df = df.reset_index(drop=True)

    print(f"    Found {len(df)} candidate(s).  Top: {df.iloc[0]['material_id']} "
          f"(eah={df.iloc[0]['energy_above_hull']}, stable={df.iloc[0]['is_stable']})")
    return df


def recommend_candidate(df: pd.DataFrame, label: str, preferred: str) -> str | None:
    """Auto-recommend one mp-id."""
    if len(df) == 0:
        return None
    # Prefer stable, low energy_above_hull, experimental.
    stable = df[df["is_stable"] == True]
    pool = stable if len(stable) > 0 else df
    # Prefer non-theoretical.
    exp = pool[pool["theoretical"] != True]
    pool = exp if len(exp) > 0 else pool
    return str(pool.iloc[0]["material_id"])


def main() -> int:
    # Load targets.
    targets_path = MATERIALS_DATA / "material_targets.toml"
    if not targets_path.is_file():
        print(f"ERROR: {targets_path} not found")
        return 1
    with open(targets_path, "rb") as f:
        config = tomllib.load(f)

    materials_cfg = config.get("materials", {})
    mp = get_mp_client()

    all_candidates: dict[str, pd.DataFrame] = {}
    recommendations: dict[str, str | None] = {}

    for label, cfg in materials_cfg.items():
        formula = cfg.get("formula", label)
        preferred = cfg.get("preferred_structure", "")
        df = search_material(mp, formula, label)
        all_candidates[label] = df

        # Save per-material CSV.
        out_csv = MATERIALS_DATA / "mp_candidates" / f"{label}_candidates.csv"
        df.to_csv(out_csv, index=False)
        print(f"    Wrote {out_csv}")

        # Recommend.
        rec = recommend_candidate(df, label, preferred)
        recommendations[label] = rec
        if rec:
            print(f"    Recommended: {rec}")
        else:
            print(f"    WARNING: could not recommend a candidate for {label}")

    # Save combined summary.
    combined_parts = []
    for label, df in all_candidates.items():
        if len(df) == 0:
            continue
        df_copy = df.copy()
        df_copy["target_label"] = label
        combined_parts.append(df_copy)
    if combined_parts:
        combined = pd.concat(combined_parts, ignore_index=True)
        combined.to_csv(MATERIALS_DATA / "reports" / "material_candidates_summary.csv", index=False)
        print(f"\nCombined summary: {MATERIALS_DATA/'reports'/'material_candidates_summary.csv'}")

    # Write selected_materials.toml for human review.
    lines = [
        "# AUTO-GENERATED material selection recommendations.",
        "# HUMAN REVIEW REQUIRED: check each mp-id and confirm before downloading.",
        "# Uncomment and adjust as needed.",
        "",
    ]
    for label in materials_cfg:
        rec = recommendations.get(label)
        lines.append(f"[selected.{label}]")
        if rec:
            lines.append(f"# Recommended based on stability and energy_above_hull.")
            lines.append(f"material_id = \"{rec}\"")
        else:
            lines.append(f"# WARNING: no candidate found — manual entry required.")
            lines.append(f"# material_id = \"mp-xxxxx\"")
        lines.append("")
    sel_path = MATERIALS_DATA / "selected_materials.toml"
    sel_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSelection recommendations: {sel_path}")
    print("Please review and confirm before running mp_download_target_phonons.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
