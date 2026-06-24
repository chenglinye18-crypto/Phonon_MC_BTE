#!/usr/bin/env python3
"""Compute DMM phonon-phonon interface conductance G_pp for target material pairs.

Reads converted dispersion files and table-driven spectra, builds shared
omega grids, and integrates DMM G_pp.  Marks pairs with missing data as
``missing_data`` with explanatory notes.

Values based on MP high-symmetry phonon data are first-pass estimates only.

Usage::

    python scripts/compute_downloaded_interface_gpp.py --temperature 300
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MATERIALS_DATA = REPO / "materials_data"
PROCESSED = MATERIALS_DATA / "processed"
REPORTS = MATERIALS_DATA / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO))


def load_dispersion_as_spec(label: str) -> dict[str, Any] | None:
    """Load a converted dispersion file and build a table-driven spectrum dict."""
    from phonon_mc import mat_from_phonon_dispersion_file, build_spectral_grid

    disp_path = PROCESSED / label / f"phonon_dispersion_{label}.txt"
    if not disp_path.is_file():
        return None
    try:
        mat = mat_from_phonon_dispersion_file(disp_path, material_name=label)
        opts = {"T0": 300.0, "n_q": 500, "n_w": 500, "weight_by_Cv_for_Q": True}
        spec = build_spectral_grid(mat, opts)
        spec["material_key"] = label
        return spec
    except Exception as exc:
        print(f"    WARNING: could not build spec for {label}: {exc}")
        return None


def compute_gpp(spec_a: dict, spec_b: dict, T: float) -> dict[str, Any]:
    """Compute DMM G_pp for two spectra using the interface_tbc_models module."""
    from interface_tbc_models import dmm_transmission_from_spectra, dmm_phonon_conductance

    # Build compatible spectra dicts.
    def to_debye_like(spec):
        return {
            "omega": spec["w_mid"][0],
            "DOS_w_b": spec["DOS_w_b"],
            "vg_w_b": spec["vg_w"],
            "dw": spec["dw"],
        }
    si = to_debye_like(spec_a)
    sj = to_debye_like(spec_b)
    g = dmm_phonon_conductance(si, sj, T=T)
    dmm = dmm_transmission_from_spectra(si, sj)
    return {**g, "dmm": dmm}


def main() -> int:
    p = argparse.ArgumentParser(description="Compute DMM G_pp for downloaded MP material pairs")
    p.add_argument("--temperature", type=float, default=300.0, help="Temperature in K")
    args = p.parse_args()
    T = float(args.temperature)

    # Load interface pairs.
    targets_path = MATERIALS_DATA / "material_targets.toml"
    if not targets_path.is_file():
        print(f"ERROR: {targets_path} not found")
        return 1
    with open(targets_path, "rb") as f:
        config = tomllib.load(f)
    pairs = config.get("interfaces", {}).get("pairs", [])
    if not pairs:
        print("No interface pairs in config")
        return 1

    # Map material labels to mp-ids (from selected_materials.toml).
    sel_path = MATERIALS_DATA / "selected_materials.toml"
    label_to_mpid = {}
    if sel_path.is_file():
        with open(sel_path, "rb") as f:
            sel_cfg = tomllib.load(f)
        for label, cfg in sel_cfg.get("selected", {}).items():
            label_to_mpid[label] = cfg.get("material_id", "")

    # Try to load specs.
    specs: dict[str, dict | None] = {}
    for pair in pairs:
        for mat in pair:
            if mat not in specs:
                specs[mat] = load_dispersion_as_spec(mat)

    # Compute G_pp for each pair.
    rows = []
    for pair in pairs:
        a, b = pair[0], pair[1]
        spec_a = specs.get(a)
        spec_b = specs.get(b)
        row: dict[str, Any] = {
            "pair": f"{a}|{b}", "material_A": a, "material_B": b,
            "mp_id_A": label_to_mpid.get(a, ""),
            "mp_id_B": label_to_mpid.get(b, ""),
            "temperature_K": T,
            "G_pp_W_m2K": np.nan, "G_pp_GW_m2K": np.nan,
            "R_pp_m2K_W": np.nan,
            "status": "", "notes": "",
        }
        if spec_a is None:
            row["status"] = "missing_data"
            row["notes"] = f"{a}: no converted dispersion (needs Phonopy)"
        elif spec_b is None:
            row["status"] = "missing_data"
            row["notes"] = f"{b}: no converted dispersion (needs Phonopy)"
        else:
            try:
                g = compute_gpp(spec_a, spec_b, T)
                row["G_pp_W_m2K"] = g["G_pp_W_m2K"]
                row["G_pp_GW_m2K"] = g["G_pp_W_m2K"] * 1e-9
                row["R_pp_m2K_W"] = g["R_pp_m2K_W"]
                row["status"] = "computed"
                row["notes"] = (
                    "Values based on MP high-symmetry phonon data are "
                    "first-pass estimates only. vg from finite difference "
                    "is approximate. Full-BZ transport requires Phonopy."
                )
                # Save per-pair transmission.
                dmm = g.get("dmm", {})
                if dmm:
                    pair_dir = PROCESSED / "interfaces"
                    pair_dir.mkdir(parents=True, exist_ok=True)
                    pair_df = pd.DataFrame({
                        "omega_rad_s": dmm.get("omega", []),
                        "M_A": dmm.get("M_i", []),
                        "M_B": dmm.get("M_j", []),
                        "T_A_to_B": dmm.get("T_i_to_j", []),
                        "T_B_to_A": dmm.get("T_j_to_i", []),
                    })
                    pair_df.to_csv(pair_dir / f"{a}_{b}_dmm_transmission.csv", index=False)
            except Exception as exc:
                row["status"] = "failed"
                row["notes"] = f"computation failed: {exc}"
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = REPORTS / "interface_gpp_summary.csv"
    df.to_csv(csv_path, index=False)

    # Markdown report.
    md_lines = [
        "# Interface DMM G_pp Summary",
        "",
        "**Values based on MP high-symmetry phonon data are first-pass estimates only.**",
        "",
        f"Temperature: {T:.1f} K",
        "",
        "| pair | mp-id A | mp-id B | G_pp (W/m²K) | G_pp (GW/m²K) | R_pp (m²K/W) | status | notes |",
        "|------|---------|---------|--------------|---------------|--------------|--------|-------|",
    ]
    for _, r in df.iterrows():
        gpp = f"{r['G_pp_W_m2K']:.3e}" if np.isfinite(r['G_pp_W_m2K']) else "N/A"
        gpp_gw = f"{r['G_pp_GW_m2K']:.4f}" if np.isfinite(r['G_pp_GW_m2K']) else "N/A"
        rpp = f"{r['R_pp_m2K_W']:.3e}" if np.isfinite(r['R_pp_m2K_W']) else "N/A"
        notes = str(r['notes'])[:80]
        md_lines.append(
            f"| {r['pair']} | {r['mp_id_A']} | {r['mp_id_B']} | "
            f"{gpp} | {gpp_gw} | {rpp} | {r['status']} | {notes} |"
        )
    md_path = REPORTS / "interface_gpp_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # Print summary.
    computed = (df["status"] == "computed").sum()
    missing = (df["status"] == "missing_data").sum()
    print(f"\nInterface G_pp summary: {computed} computed, {missing} missing_data")
    print(f"CSV: {csv_path}")
    print(f"MD:  {md_path}")
    for _, r in df.iterrows():
        status_icon = "OK" if r["status"] == "computed" else "--"
        print(f"  {status_icon} {r['pair']:20s}  {r['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
