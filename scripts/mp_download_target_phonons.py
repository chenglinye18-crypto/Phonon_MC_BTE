#!/usr/bin/env python3
"""Download Materials Project structure and phonon data for selected materials.

Reads ``materials_data/selected_materials.toml``, downloads structure (CIF),
phonon bandstructure, phonon DOS, and metadata for each selected mp-id.

Usage::

    export MP_API_KEY="your_key"
    python scripts/mp_download_target_phonons.py
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MATERIALS_DATA = REPO / "materials_data"
MP_RAW = MATERIALS_DATA / "mp_raw"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_mp_client():
    api_key = os.environ.get("MP_API_KEY", "")
    if not api_key:
        print("ERROR: MP_API_KEY environment variable not set.")
        sys.exit(1)
    try:
        from mp_api.client import MPRester
    except ImportError:
        print("ERROR: pip install mp_api pymatgen")
        sys.exit(1)
    return MPRester(api_key)


def download_material(mp: Any, material_id: str, label: str) -> dict[str, Any]:
    """Download all available data for one material_id."""
    out_dir = MP_RAW / label
    out_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "label": label,
        "material_id": material_id,
        "download_time_utc": datetime.now(timezone.utc).isoformat(),
        "has_structure": False,
        "has_phonon_bandstructure": False,
        "has_phonon_dos": False,
        "needs_phonopy": False,
        "warnings": [],
    }

    # --- Structure ---
    print(f"  Downloading structure for {material_id} ...")
    try:
        structure = mp.get_structure_by_material_id(material_id)
        if structure:
            # Save CIF
            cif_path = out_dir / f"{label}_structure.cif"
            structure.to(filename=str(cif_path), fmt="cif")
            # Save JSON (as_dict)
            json_path = out_dir / f"{label}_structure.json"
            with open(json_path, "w") as f:
                json.dump(structure.as_dict(), f, cls=NumpyEncoder)
            meta["has_structure"] = True
            meta["structure_formula"] = str(structure.composition.reduced_formula)
            meta["structure_n_sites"] = len(structure)
            print(f"    OK: {cif_path.name}")
    except Exception as exc:
        meta["warnings"].append(f"structure download failed: {exc}")
        print(f"    WARNING: structure download failed: {exc}")

    # --- Phonon bandstructure ---
    print(f"  Downloading phonon bandstructure for {material_id} ...")
    try:
        ph_bs = mp.get_phonon_bandstructure_by_material_id(material_id)
        if ph_bs is not None:
            bs_path = out_dir / f"{label}_phonon_bandstructure.json"
            ph_dict = ph_bs.as_dict()
            with open(bs_path, "w") as f:
                json.dump(ph_dict, f, cls=NumpyEncoder)
            meta["has_phonon_bandstructure"] = True
            n_branches = len(ph_dict.get("branches", []))
            n_q = len(ph_dict.get("qpoints", []))
            meta["phonon_bandstructure_n_branches"] = n_branches
            meta["phonon_bandstructure_n_qpoints"] = n_q
            print(f"    OK: {bs_path.name}  ({n_branches} branches, {n_q} q-points)")
        else:
            meta["warnings"].append("phonon bandstructure returned None")
            print("    WARNING: no phonon bandstructure available")
    except Exception as exc:
        msg = f"phonon bandstructure download failed: {exc}"
        meta["warnings"].append(msg)
        print(f"    WARNING: {msg}")

    # --- Phonon DOS ---
    print(f"  Downloading phonon DOS for {material_id} ...")
    try:
        ph_dos = mp.get_phonon_dos_by_material_id(material_id)
        if ph_dos is not None:
            dos_path = out_dir / f"{label}_phonon_dos.json"
            dos_dict = ph_dos.as_dict()
            with open(dos_path, "w") as f:
                json.dump(dos_dict, f, cls=NumpyEncoder)
            meta["has_phonon_dos"] = True
            freq = dos_dict.get("frequencies", [])
            meta["phonon_dos_f_min_THz"] = float(np.min(freq)) if len(freq) else None
            meta["phonon_dos_f_max_THz"] = float(np.max(freq)) if len(freq) else None
            print(f"    OK: {dos_path.name}")
        else:
            meta["warnings"].append("phonon DOS returned None")
            print("    WARNING: no phonon DOS available")
    except Exception as exc:
        msg = f"phonon DOS download failed: {exc}"
        meta["warnings"].append(msg)
        print(f"    WARNING: {msg}")

    # --- Mark needs_phonopy ---
    if not meta["has_phonon_bandstructure"] and not meta["has_phonon_dos"]:
        meta["needs_phonopy"] = True
    elif not meta["has_phonon_bandstructure"]:
        meta["needs_phonopy"] = True  # need Phonopy for full dispersion

    # --- Save metadata ---
    meta_path = out_dir / f"{label}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, cls=NumpyEncoder)

    return meta


def main() -> int:
    sel_path = MATERIALS_DATA / "selected_materials.toml"
    if not sel_path.is_file():
        print(f"ERROR: {sel_path} not found.  Run mp_search_target_materials.py first.")
        return 1
    with open(sel_path, "rb") as f:
        config = tomllib.load(f)

    selected = config.get("selected", {})
    if not selected:
        print("ERROR: no [selected] entries in selected_materials.toml")
        return 1

    mp = get_mp_client()

    # Try to get MP database version.
    try:
        from mp_api.client import MPRester
        # The version info is not easily accessible via mp_api; use a fixed note.
        mp_version = "Materials Project (mp_api v2024+)"
    except Exception:
        mp_version = "Materials Project (unknown version)"

    all_meta = {}
    for label, cfg in selected.items():
        material_id = cfg.get("material_id", "")
        if not material_id:
            print(f"SKIP {label}: no material_id in config")
            continue
        print(f"\n{'='*50}")
        print(f"Downloading: {label}  (mp-id: {material_id})")
        meta = download_material(mp, material_id, label)
        meta["mp_database_version"] = mp_version
        all_meta[label] = meta

    # Summary.
    print(f"\n{'='*50}")
    print("Download summary:")
    for label, meta in all_meta.items():
        bs = "BS" if meta["has_phonon_bandstructure"] else "no-BS"
        dos = "DOS" if meta["has_phonon_dos"] else "no-DOS"
        needs = "NEEDS_PHONOPY" if meta.get("needs_phonopy") else "OK"
        print(f"  {label:10s}  {bs:5s}  {dos:5s}  {needs}")

    # Save overall metadata.
    overall_meta = {
        "download_time_utc": datetime.now(timezone.utc).isoformat(),
        "mp_database_version": mp_version,
        "materials": all_meta,
    }
    with open(MP_RAW / "download_manifest.json", "w") as f:
        json.dump(overall_meta, f, indent=2, cls=NumpyEncoder)
    print(f"\nManifest: {MP_RAW / 'download_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
