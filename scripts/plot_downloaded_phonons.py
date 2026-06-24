#!/usr/bin/env python3
"""Plot downloaded phonon bandstructures, vg, and DOS for all target materials.

Outputs: materials_data/plots/{label}_phonon_bandstructure.png
         materials_data/plots/{label}_vg_vs_frequency.png
         materials_data/plots/{label}_phonon_dos.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
MATERIALS_DATA = REPO / "materials_data"
MP_RAW = MATERIALS_DATA / "mp_raw"
PROCESSED = MATERIALS_DATA / "processed"
PLOTS = MATERIALS_DATA / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def plot_bandstructure(label: str) -> bool:
    bs_path = MP_RAW / label / f"{label}_phonon_bandstructure.json"
    if not bs_path.is_file():
        print(f"  {label}: no bandstructure JSON → skip")
        return False
    with open(bs_path) as f:
        data = json.load(f)
    branches = data.get("branches", [])
    qpoints = data.get("qpoints", [])
    if not branches:
        print(f"  {label}: empty bandstructure → skip")
        return False

    dist = [0.0]
    for i in range(1, len(qpoints)):
        dist.append(dist[-1] + np.linalg.norm(np.array(qpoints[i]) - np.array(qpoints[i-1])))

    freq = np.array(branches, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for b in range(freq.shape[0]):
        ax.plot(dist, freq[b], linewidth=1.0, color="#0f4c81")
    ax.set_xlabel("q-path distance (Å⁻¹)")
    ax.set_ylabel("Frequency (THz)")
    ax.set_title(f"{label}: Phonon bandstructure")
    ax.set_ylim(bottom=0)
    fig.savefig(PLOTS / f"{label}_phonon_bandstructure.png", dpi=150)
    plt.close(fig)
    print(f"  {label}: bandstructure plot OK")
    return True


def plot_dos(label: str) -> bool:
    dos_path = MP_RAW / label / f"{label}_phonon_dos.json"
    if not dos_path.is_file():
        print(f"  {label}: no DOS JSON → skip")
        return False
    with open(dos_path) as f:
        data = json.load(f)
    freq = np.array(data.get("frequencies", []), dtype=np.float64)
    dos_vals = np.array(data.get("densities", []), dtype=np.float64)
    if freq.size == 0:
        return False
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(freq, dos_vals, color="#bf360c", linewidth=1.5)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Phonon DOS")
    ax.set_title(f"{label}: Phonon DOS")
    fig.savefig(PLOTS / f"{label}_phonon_dos.png", dpi=150)
    plt.close(fig)
    print(f"  {label}: DOS plot OK")
    return True


def plot_dispersion_vg(label: str) -> bool:
    disp_path = PROCESSED / label / f"phonon_dispersion_{label}.txt"
    if not disp_path.is_file():
        print(f"  {label}: no dispersion txt → skip vg plot")
        return False
    try:
        data = np.loadtxt(disp_path, comments="#")
    except Exception:
        return False
    if data.ndim == 1:
        data = data.reshape(1, -1)
    branch_id = np.round(data[:, 0]).astype(int)
    freq_thz = data[:, 2]
    vg = np.abs(data[:, 3])
    unique_branches = np.unique(branch_id)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    colors = plt.cm.tab10.colors
    for i, b in enumerate(unique_branches):
        mask = branch_id == b
        order = np.argsort(freq_thz[mask])
        ax.plot(freq_thz[mask][order], vg[mask][order], linewidth=1.0,
                color=colors[i % len(colors)], label=f"branch {b}")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("|v_g| (m/s)")
    ax.set_title(f"{label}: Group velocity vs frequency (approximate)")
    if len(unique_branches) <= 6:
        ax.legend(fontsize=8)
    ax.set_yscale("log")
    fig.savefig(PLOTS / f"{label}_vg_vs_frequency.png", dpi=150)
    plt.close(fig)
    print(f"  {label}: vg plot OK")
    return True


def main() -> int:
    labels = sorted([
        d.name for d in MP_RAW.iterdir()
        if d.is_dir() and (d / f"{d.name}_metadata.json").is_file()
    ])
    if not labels:
        print("No downloaded data. Run mp_download_target_phonons.py first.")
        return 0

    for label in labels:
        print(f"Plotting: {label}")
        ok_bs = plot_bandstructure(label)
        ok_dos = plot_dos(label)
        ok_vg = plot_dispersion_vg(label)
        if not (ok_bs or ok_dos or ok_vg):
            print(f"  WARNING: no plot data for {label}")

    print(f"\nPlots saved to: {PLOTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
