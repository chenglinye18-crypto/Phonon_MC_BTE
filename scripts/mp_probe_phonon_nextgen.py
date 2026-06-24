#!/usr/bin/env python3
"""Probe a single Materials Project material for phonon data via the next-gen API.

Uses ``mpr.materials.phonon`` sub-client (pheasy/dfpt methods).

Usage::

    export MP_API_KEY="your_key"
    python scripts/mp_probe_phonon_nextgen.py --material-id mp-30 --label Cu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
NEXTGEN_DIR = REPO / "materials_data" / "mp_raw_nextgen"
NEXTGEN_DIR.mkdir(parents=True, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


def _robust_serialize(obj: Any) -> Any:
    """Convert pymatgen/emmet objects to JSON-serializable dicts."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_robust_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _robust_serialize(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # pymatgen objects
    for method in ["as_dict", "model_dump"]:
        if hasattr(obj, method):
            try:
                return _robust_serialize(getattr(obj, method)())
            except Exception:
                pass
    # Fallback
    try:
        return dict(obj)
    except Exception:
        return str(obj)


def probe_material(mpr, material_id: str, label: str,
                   phonon_method: str = "pheasy",
                   path_type: str = "latimer_munro") -> dict[str, Any]:
    """Probe one material and save all available phonon data."""
    out_dir = NEXTGEN_DIR / f"{label}_{material_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "material_id": material_id,
        "label": label,
        "probe_time_utc": datetime.now(timezone.utc).isoformat(),
        "tried_methods": [],
        "tried_path_types": [],
        "has_phonon_doc": False,
        "has_bandstructure": False,
        "has_dos": False,
        "has_forceconstants": False,
        "chosen_phonon_method": None,
        "chosen_path_type": None,
        "n_docs": 0,
        "n_branches": 0,
        "n_qpoints": 0,
        "n_freqs_dos": 0,
        "frequency_min_THz": None,
        "frequency_max_THz": None,
        "frequency_min_THz_dos": None,
        "frequency_max_THz_dos": None,
        "fc_length": 0,
        "errors": [],
    }

    methods_to_try = [phonon_method] if phonon_method else ["pheasy", "dfpt", None]
    path_types_to_try = [path_type] if path_type else ["latimer_munro", "setyawan_curtarolo"]

    # --- Phonon search ---
    chosen_method = None
    for method in methods_to_try:
        meta["tried_methods"].append(str(method))
        try:
            kwargs = {"material_ids": [material_id]}
            if method is not None:
                kwargs["phonon_method"] = method
            docs = mpr.materials.phonon.search(**kwargs)
            if docs:
                meta["n_docs"] = len(docs)
                meta["has_phonon_doc"] = True
                chosen_method = method or "default"
                meta["chosen_phonon_method"] = chosen_method

                # Save phonon doc
                doc_data = _robust_serialize(docs[0])
                doc_path = out_dir / f"{label}_{material_id}_phonon_doc_{chosen_method}.json"
                with open(doc_path, "w") as f:
                    json.dump(doc_data, f, indent=2, cls=NumpyEncoder)
                print(f"  [OK] phonon doc ({chosen_method}): {doc_path.name}")

                # Extract structure info
                if isinstance(doc_data, dict):
                    meta["formula_pretty"] = doc_data.get("formula_pretty", "")
                break
        except Exception as e:
            meta["errors"].append(f"phonon.search({method}): {e}")

    if not meta["has_phonon_doc"]:
        meta["errors"].append("no phonon data found with any method")
        # Still save metadata
        meta_path = out_dir / f"{label}_{material_id}_phonon_probe_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [WARN] No phonon data for {material_id}")
        return meta

    # --- Bandstructure ---
    for pt in path_types_to_try:
        meta["tried_path_types"].append(pt)
        try:
            bs = mpr.materials.phonon.get_bandstructure_from_material_id(
                material_id=material_id, phonon_method=chosen_method, path_type=pt
            )
            if bs is not None:
                bs_data = _robust_serialize(bs)
                bs_path = out_dir / f"{label}_{material_id}_phonon_bandstructure_{chosen_method}_{pt}.json"
                with open(bs_path, "w") as f:
                    json.dump(bs_data, f, indent=2, cls=NumpyEncoder)
                meta["has_bandstructure"] = True
                meta["chosen_path_type"] = pt

                # Extract metadata from BS
                if isinstance(bs_data, dict):
                    # Check for both 'frequencies' (nextgen) and 'branches' (legacy) keys
                    freq_data = bs_data.get("frequencies", bs_data.get("branches", []))
                    qpoints = bs_data.get("qpoints", [])
                    meta["n_branches"] = len(freq_data)
                    meta["n_qpoints"] = len(qpoints)
                    if freq_data:
                        all_freqs = []
                        for b in freq_data:
                            all_freqs.extend(b if isinstance(b, list) else [])
                        if all_freqs:
                            meta["frequency_min_THz"] = float(np.min(all_freqs))
                            meta["frequency_max_THz"] = float(np.max(all_freqs))
                        # Check for imaginary modes
                        neg_freqs = [f for f in all_freqs if f < -0.1]
                        if neg_freqs:
                            meta["errors"].append(
                                f"imaginary modes detected: {len(neg_freqs)} frequencies below -0.1 THz"
                            )
                print(f"  [OK] bandstructure ({chosen_method}, {pt}): {bs_path.name} "
                      f"({meta['n_branches']} branches, {meta['n_qpoints']} q-points)")
                break
        except Exception as e:
            meta["errors"].append(f"bandstructure({pt}): {e}")

    # --- DOS ---
    try:
        dos = mpr.materials.phonon.get_dos_from_material_id(
            material_id=material_id, phonon_method=chosen_method
        )
        if dos is not None:
            dos_data = _robust_serialize(dos)
            dos_path = out_dir / f"{label}_{material_id}_phonon_dos_{chosen_method}.json"
            with open(dos_path, "w") as f:
                json.dump(dos_data, f, indent=2, cls=NumpyEncoder)
            meta["has_dos"] = True
            if isinstance(dos_data, dict):
                freqs = dos_data.get("frequencies", [])
                meta["n_freqs_dos"] = len(freqs)
                if freqs:
                    meta["frequency_min_THz_dos"] = float(np.min(freqs))
                    meta["frequency_max_THz_dos"] = float(np.max(freqs))
            print(f"  [OK] DOS ({chosen_method}): {dos_path.name} ({meta['n_freqs_dos']} freq points)")
    except Exception as e:
        meta["errors"].append(f"DOS: {e}")

    # --- Force constants (pheasy only) ---
    if chosen_method == "pheasy":
        try:
            fc = mpr.materials.phonon.get_forceconstants_from_material_id(
                material_id=material_id, phonon_method="pheasy"
            )
            if fc is not None:
                fc_data = _robust_serialize(fc)
                fc_path = out_dir / f"{label}_{material_id}_forceconstants_{chosen_method}.json"
                with open(fc_path, "w") as f:
                    json.dump(fc_data, f, cls=NumpyEncoder)
                meta["has_forceconstants"] = True
                meta["fc_length"] = len(fc_data) if isinstance(fc_data, list) else 0
                print(f"  [OK] force constants ({chosen_method}): {fc_path.name} ({meta['fc_length']} entries)")
        except Exception as e:
            meta["errors"].append(f"forceconstants: {e}")
            meta["forceconstants_failed"] = True
            print(f"  [WARN] force constants failed: {e}")

    # --- Save metadata ---
    meta_path = out_dir / f"{label}_{material_id}_phonon_probe_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [OK] metadata: {meta_path.name}")

    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe single MP material for next-gen phonon data")
    p.add_argument("--material-id", required=True, help="MP material ID e.g. mp-30")
    p.add_argument("--label", default="", help="Material label e.g. Cu")
    p.add_argument("--phonon-method", default="pheasy",
                   choices=["pheasy", "dfpt", ""],
                   help="Phonon method (default: pheasy)")
    p.add_argument("--path-type", default="latimer_munro",
                   choices=["latimer_munro", "setyawan_curtarolo", ""],
                   help="Bandstructure path type (default: latimer_munro)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("MP_API_KEY", "")
    if not api_key:
        print("ERROR: MP_API_KEY environment variable not set.")
        print("  export MP_API_KEY='your_key'")
        return 1

    try:
        from mp_api.client import MPRester
    except ImportError:
        print("ERROR: pip install -U mp_api pymatgen monty")
        return 1

    label = args.label or args.material_id
    mpr = MPRester(api_key=api_key)

    print(f"Probing {args.material_id} ({label}) "
          f"method={args.phonon_method or 'auto'} "
          f"path_type={args.path_type or 'auto'}")
    print("-" * 50)

    meta = probe_material(mpr, args.material_id, label,
                          phonon_method=args.phonon_method or "",
                          path_type=args.path_type or "")

    # Summary
    print()
    print(f"{'='*50}")
    print(f"Summary for {label} ({args.material_id}):")
    print(f"  phonon doc:    {'YES' if meta['has_phonon_doc'] else 'NO'}")
    print(f"  bandstructure: {'YES' if meta['has_bandstructure'] else 'NO'} "
          f"({meta['n_branches']} br, {meta['n_qpoints']} q)")
    print(f"  DOS:           {'YES' if meta['has_dos'] else 'NO'} "
          f"({meta['n_freqs_dos']} pts)")
    print(f"  force consts:  {'YES' if meta['has_forceconstants'] else 'NO'}")
    print(f"  method:        {meta['chosen_phonon_method']}")
    print(f"  path_type:     {meta['chosen_path_type']}")
    if meta["errors"]:
        print(f"  errors:        {len(meta['errors'])}")
        for e in meta["errors"][:5]:
            print(f"    - {str(e)[:120]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
