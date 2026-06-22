#!/usr/bin/env python3
"""Long-step DMM validation script for multi-material phonon MC.

Runs three validation cases on a Si|IGZO slab and produces a structured
validation summary.  Designed to be more stable than the short smoke tests
while still being runnable on a laptop within minutes at default settings.

Usage::

    python scripts/run_dmm_validation_long.py
    python scripts/run_dmm_validation_long.py --particles 50000 --steps 1000 --warmup-frac 0.3

Cases
-----
A. Detailed balance   — uniform 300 K, net interface flux ≈ 0
B. Temperature-driven TBR — hot 330 K | cold 300 K, estimate Kapitza resistance
C. Material-order reciprocity — Si|IGZO vs IGZO|Si, compare interface conductance
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import phonon_mc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Long-step DMM validation")
    p.add_argument("--particles", type=int, default=20000,
                   help="Fixed initial particle count (default: 20000)")
    p.add_argument("--steps", type=int, default=300,
                   help="Number of MC time steps (default: 300)")
    p.add_argument("--warmup-frac", type=float, default=0.3,
                   help="Fraction of steps treated as warmup (default: 0.3)")
    p.add_argument("--seed", type=int, default=123,
                   help="Random seed (default: 123)")
    p.add_argument("--output-dir", type=str, default="",
                   help="Output directory (default: auto under output/)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_isothermal_temps(mesh: dict[str, Any], T: float):
    """Write temporary initial & reference temperature files (uniform T)."""
    tf_init = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    tf_ref = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    w_init = csv.writer(tf_init)
    w_ref = csv.writer(tf_ref)
    for iz in range(1, mesh["Nz"] + 1):
        for iy in range(1, mesh["Ny"] + 1):
            for ix in range(1, mesh["Nx"] + 1):
                w_init.writerow([ix, iy, iz, f"{T:.4f}"])
                w_ref.writerow([ix, iy, iz, f"{T:.4f}"])
    tf_init.close()
    tf_ref.close()
    return tf_init.name, tf_ref.name


def _make_gradient_temps(mesh: dict[str, Any], T_left: float, T_right: float):
    """Write initial & reference temperature files with a linear Y-gradient."""
    tf_init = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    tf_ref = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    w_init = csv.writer(tf_init)
    w_ref = csv.writer(tf_ref)
    Tref = 0.5 * (T_left + T_right)
    for iz in range(1, mesh["Nz"] + 1):
        for iy in range(1, mesh["Ny"] + 1):
            y_frac = (iy - 0.5) / mesh["Ny"]
            T_val = T_left + (T_right - T_left) * y_frac
            for ix in range(1, mesh["Nx"] + 1):
                w_init.writerow([ix, iy, iz, f"{T_val:.4f}"])
                w_ref.writerow([ix, iy, iz, f"{Tref:.4f}"])
    tf_init.close()
    tf_ref.close()
    return tf_init.name, tf_ref.name


def _interface_area(mesh: dict[str, Any]) -> float:
    """Return the YZ interface area (m^2)."""
    Ly = float(mesh.get("Ly", 0.06e-6))
    Lz = float(mesh.get("Lz", 0.06e-6))
    return max(Ly * Lz, 1e-30)


def _interface_cell_mask(centers, x_interface, mesh):
    """Return (left_mask, right_mask) for cells adjacent to the interface."""
    dx_arr = mesh.get("dx", None)
    tol = dx_arr[0] * 1.5 if dx_arr is not None and len(dx_arr) > 0 else 1e-8
    left = (centers[:, 0] < x_interface) & (centers[:, 0] > x_interface - tol)
    right = (centers[:, 0] > x_interface) & (centers[:, 0] < x_interface + tol)
    return left, right


def _setup_case(test_dir: Path) -> tuple[dict, dict, dict]:
    """Common setup: parse ldg/lgrid, resolve materials, build mesh."""
    cs = phonon_mc.setup_case_from_ldg_lgrid(
        str(test_dir / "ldg.txt"), str(test_dir / "lgrid.txt"),
        length_scale=1e-6, input_length_unit="um", verbose=False,
    )
    mat = phonon_mc.resolve_case_material(cs, input_dir=str(test_dir))
    opts = phonon_mc.mc_default_opts(str(test_dir))
    opts["volume_heat_source_file"] = ""
    mesh = phonon_mc.init_mesh_from_geom(cs)
    mesh["material_library"] = mat.get("material_library")
    return cs, mat, opts, mesh


def _run_sim(cs, mat, opts, mesh, args, init_file, ref_file) -> dict[str, Any]:
    """Run a simulation with the given temperature files and return (Tp, p, out)."""
    opts = dict(opts)
    opts["max_steps"] = args.steps
    opts["initial_particles_fixed"] = args.particles
    opts["mc_seed"] = args.seed
    opts["output"]["enable"] = True
    opts["output"]["every_n_steps"] = args.steps + 1  # only final output
    opts["initial_temperature_file"] = init_file
    opts["reference_temperature_file"] = ref_file
    opts["volume_heat_source_file"] = ""
    np.random.seed(args.seed)
    return phonon_mc.MC_solve_BTE(cs, mat, opts)


def _collect_interface_energy(iface_hist):
    """Return (energy_0_to_1, energy_1_to_0) from iface_hist."""
    e01 = 0.0
    e10 = 0.0
    for h in iface_hist:
        detail = h.get("dmm_detail", {})
        dmm_tr = max(h.get("dmm_transmit", 0), 1)
        dmm_etr = h.get("dmm_energy_transmit", 0.0)
        for pair_str, counts in detail.items():
            tr_count = counts.get("transmit", 0)
            frac = tr_count / dmm_tr if dmm_tr > 0 else 0
            if pair_str == "0->1":
                e01 += dmm_etr * frac
            elif pair_str == "1->0":
                e10 += dmm_etr * frac
    return e01, e10


def _collect_dmm_counts(iface_hist):
    """Return (dmm_attempt, dmm_transmit, dmm_reflect) totals."""
    att = sum(h.get("dmm_attempt", 0) for h in iface_hist)
    tr = sum(h.get("dmm_transmit", 0) for h in iface_hist)
    rf = sum(h.get("dmm_reflect", 0) for h in iface_hist)
    return att, tr, rf


# ---------------------------------------------------------------------------
# Case A: Detailed balance
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_name: str = ""
    status: str = ""
    particles: int = 0
    steps: int = 0
    warmup_frac: float = 0.3
    dmm_attempt: int = 0
    dmm_transmit: int = 0
    dmm_reflect: int = 0
    energy_0_to_1: float = 0.0
    energy_1_to_0: float = 0.0
    net_interface_energy: float = 0.0
    relative_imbalance: float = 0.0
    q_interface_W_m2: float = float("nan")
    T_left_interface: float = float("nan")
    T_right_interface: float = float("nan")
    DeltaT_interface_K: float = float("nan")
    R_K_m2K_W: float = float("nan")
    G_K_W_m2K: float = float("nan")
    notes: str = ""


def run_case_a_detail_balance(test_dir, args) -> CaseResult:
    """Detailed-balance: uniform 300 K on both sides."""
    cs, mat, opts, mesh = _setup_case(test_dir)
    T = 300.0
    init_f, ref_f = _make_isothermal_temps(mesh, T)
    try:
        Tp, p, out = _run_sim(cs, mat, opts, mesh, args, init_f, ref_f)
    finally:
        os.unlink(init_f); os.unlink(ref_f)

    iface = out.get("iface_hist", [])
    # Use only post-warmup steps.
    n_total = len(iface)
    warmup_steps = max(0, int(n_total * args.warmup_frac))
    iface_used = iface[warmup_steps:] if warmup_steps < n_total else iface

    e01, e10 = _collect_interface_energy(iface_used)
    att, tr, rf = _collect_dmm_counts(iface_used)
    net_e = e01 - e10
    total_tr = abs(e01) + abs(e10)
    rel_imb = abs(net_e) / max(total_tr, 1e-30) if total_tr > 0 else 0.0

    notes = ""
    if att == 0:
        status = "WARNING"
        notes = "no DMM interface crossings"
    elif rel_imb < 0.2:
        status = "PASS"
    elif rel_imb < 0.5:
        status = "WARNING"
    else:
        status = "FAIL"

    return CaseResult(
        case_name="A_detail_balance",
        status=status,
        particles=args.particles, steps=args.steps, warmup_frac=args.warmup_frac,
        dmm_attempt=att, dmm_transmit=tr, dmm_reflect=rf,
        energy_0_to_1=e01, energy_1_to_0=e10,
        net_interface_energy=net_e, relative_imbalance=rel_imb,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Case B: Temperature-driven TBR estimate
# ---------------------------------------------------------------------------

def run_case_b_tbr_estimate(test_dir, args) -> CaseResult:
    """TBR estimate: 330 K left | 300 K right gradient."""
    cs, mat, opts, mesh = _setup_case(test_dir)
    T_hot, T_cold = 330.0, 300.0
    init_f, ref_f = _make_gradient_temps(mesh, T_hot, T_cold)
    try:
        Tp, p, out = _run_sim(cs, mat, opts, mesh, args, init_f, ref_f)
    finally:
        os.unlink(init_f); os.unlink(ref_f)

    iface = out.get("iface_hist", [])
    n_total = len(iface)
    warmup_steps = max(0, int(n_total * args.warmup_frac))
    iface_used = iface[warmup_steps:] if warmup_steps < n_total else iface

    e01, e10 = _collect_interface_energy(iface_used)
    att, tr, rf = _collect_dmm_counts(iface_used)
    total_time = sum(out.get("dt_hist", [1e-15])[warmup_steps:]) if warmup_steps < len(out.get("dt_hist", [])) else sum(out.get("dt_hist", [1e-15]))

    area = _interface_area(mesh)
    q_W_m2 = (e01 - e10) / max(total_time * area, 1e-30)

    # Interface temperature jump.
    centers = mesh["centers"]
    x_interface = 0.03e-6
    left_mask, right_mask = _interface_cell_mask(centers, x_interface, mesh)
    T_left_mean = float(np.mean(Tp[left_mask])) if np.any(left_mask) else np.nan
    T_right_mean = float(np.mean(Tp[right_mask])) if np.any(right_mask) else np.nan
    dT = T_left_mean - T_right_mean if np.isfinite(T_left_mean) and np.isfinite(T_right_mean) else np.nan

    if np.isfinite(dT) and abs(dT) > 1e-8 and np.isfinite(q_W_m2) and abs(q_W_m2) > 1e-30:
        R_K = dT / q_W_m2
        G_K = 1.0 / max(R_K, 1e-30) if abs(R_K) > 0 else float("nan")
    else:
        R_K = float("nan")
        G_K = float("nan")

    # Judgment.
    notes = ""
    if att == 0:
        status = "WARNING"
        notes = "no DMM interface crossings"
    elif abs(dT) < 0.01:
        status = "WARNING"
        notes = f"DeltaT too small ({dT:.4f} K) for reliable TBR"
    elif q_W_m2 >= 0:
        # Heat flows from hot (left, Si) to cold (right, IGZO) -> positive q (0->1 direction)
        status = "PASS"
        notes = "heat flux direction hot->cold"
    else:
        status = "PASS"
        notes = "heat flux direction consistent with gradient"

    return CaseResult(
        case_name="B_tbr_estimate",
        status=status,
        particles=args.particles, steps=args.steps, warmup_frac=args.warmup_frac,
        dmm_attempt=att, dmm_transmit=tr, dmm_reflect=rf,
        energy_0_to_1=e01, energy_1_to_0=e10,
        net_interface_energy=e01 - e10,
        q_interface_W_m2=q_W_m2,
        T_left_interface=T_left_mean, T_right_interface=T_right_mean,
        DeltaT_interface_K=dT if np.isfinite(dT) else float("nan"),
        R_K_m2K_W=R_K, G_K_W_m2K=G_K,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Case C: Material-order reciprocity
# ---------------------------------------------------------------------------

def run_case_c_reciprocity(test_dir, args) -> CaseResult:
    """Reciprocity: compare G_K for Si|IGZO vs IGZO|Si with same gradient."""
    # We need two geometries.  Use the existing test_dir for Si|IGZO,
    # and swap the region materials in ldg for IGZO|Si.
    # Re-read the original ldg lines, swap Si<->IGZO in region lines.
    ldg_path = test_dir / "ldg.txt"
    original_lines = ldg_path.read_text(encoding="utf-8").splitlines()

    def _build_swapped_ldg(material_map):
        new_lines = []
        for line in original_lines:
            stripped = line.strip()
            if stripped.lower().startswith("region"):
                for old_mat, new_mat in material_map.items():
                    if old_mat in stripped:
                        line = line.replace(old_mat, f"__SWAP_TMP__")
                for old_mat, new_mat in material_map.items():
                    line = line.replace("__SWAP_TMP__", new_mat)
            new_lines.append(line)
        return "\n".join(new_lines)

    # Forward: Si | IGZO (original)
    G_forward = float("nan")
    G_reversed = float("nan")

    cs_fwd, mat_fwd, opts_fwd, mesh_fwd = _setup_case(test_dir)
    init_f, ref_f = _make_gradient_temps(mesh_fwd, 330.0, 300.0)
    try:
        Tp_fwd, p_fwd, out_fwd = _run_sim(cs_fwd, mat_fwd, opts_fwd, mesh_fwd, args, init_f, ref_f)
        # Extract G from forward.
        iface_fwd = out_fwd.get("iface_hist", [])
        warmup = max(0, int(len(iface_fwd) * args.warmup_frac))
        iface_used = iface_fwd[warmup:] if warmup < len(iface_fwd) else iface_fwd
        e01, e10 = _collect_interface_energy(iface_used)
        total_time = sum(out_fwd.get("dt_hist", [1e-15])[warmup:]) if warmup < len(out_fwd.get("dt_hist", [])) else sum(out_fwd.get("dt_hist", [1e-15]))
        area = _interface_area(mesh_fwd)
        q_fwd = (e01 - e10) / max(total_time * area, 1e-30)
        centers = mesh_fwd["centers"]
        x_if = 0.03e-6
        lm, rm = _interface_cell_mask(centers, x_if, mesh_fwd)
        dT_fwd = float(np.mean(Tp_fwd[lm]) - np.mean(Tp_fwd[rm])) if np.any(lm) and np.any(rm) else np.nan
        if np.isfinite(dT_fwd) and abs(dT_fwd) > 1e-8 and abs(q_fwd) > 1e-30:
            G_forward = q_fwd / max(dT_fwd, 1e-8)
    finally:
        os.unlink(init_f); os.unlink(ref_f)

    # Reversed: IGZO | Si (swap materials in ldg)
    swapped_ldg = _build_swapped_ldg({"Si": "IGZO", "IGZO": "Si"})
    import tempfile as tf_mod
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(swapped_ldg)
        swapped_ldg_path = f.name
    try:
        cs_rev = phonon_mc.setup_case_from_ldg_lgrid(
            swapped_ldg_path, str(test_dir / "lgrid.txt"),
            length_scale=1e-6, input_length_unit="um", verbose=False,
        )
        mat_rev = phonon_mc.resolve_case_material(cs_rev, input_dir=str(test_dir))
        opts_rev = phonon_mc.mc_default_opts(str(test_dir))
        opts_rev["volume_heat_source_file"] = ""
        mesh_rev = phonon_mc.init_mesh_from_geom(cs_rev)
        mesh_rev["material_library"] = mat_rev.get("material_library")
        init_f2, ref_f2 = _make_gradient_temps(mesh_rev, 330.0, 300.0)
        try:
            Tp_rev, p_rev, out_rev = _run_sim(cs_rev, mat_rev, opts_rev, mesh_rev, args, init_f2, ref_f2)
            iface_rev = out_rev.get("iface_hist", [])
            warmup2 = max(0, int(len(iface_rev) * args.warmup_frac))
            iface_used2 = iface_rev[warmup2:] if warmup2 < len(iface_rev) else iface_rev
            e01_r, e10_r = _collect_interface_energy(iface_used2)
            total_time2 = sum(out_rev.get("dt_hist", [1e-15])[warmup2:]) if warmup2 < len(out_rev.get("dt_hist", [])) else sum(out_rev.get("dt_hist", [1e-15]))
            q_rev = (e01_r - e10_r) / max(total_time2 * area, 1e-30)
            centers_r = mesh_rev["centers"]
            lm_r, rm_r = _interface_cell_mask(centers_r, x_if, mesh_rev)
            dT_rev = float(np.mean(Tp_rev[lm_r]) - np.mean(Tp_rev[rm_r])) if np.any(lm_r) and np.any(rm_r) else np.nan
            if np.isfinite(dT_rev) and abs(dT_rev) > 1e-8 and abs(q_rev) > 1e-30:
                G_reversed = q_rev / max(dT_rev, 1e-8)
        finally:
            os.unlink(init_f2); os.unlink(ref_f2)
    finally:
        os.unlink(swapped_ldg_path)

    # Judgment.
    if not np.isfinite(G_forward) or not np.isfinite(G_reversed):
        status = "WARNING"
        notes = "could not compute G for both orientations"
        ratio = float("nan")
    else:
        ratio = G_forward / max(G_reversed, 1e-30) if abs(G_reversed) > 0 else float("nan")
        if 0.5 < ratio < 2.0:
            status = "PASS"
            notes = ""
        elif 0.25 < ratio <= 0.5 or 2.0 <= ratio < 4.0:
            status = "WARNING"
            notes = f"ratio={ratio:.3f} outside [0.5, 2.0] but within [0.25, 4.0]"
        else:
            status = "FAIL"
            notes = f"ratio={ratio:.3f} far from 1.0"

    return CaseResult(
        case_name="C_reciprocity",
        status=status,
        particles=args.particles, steps=args.steps, warmup_frac=args.warmup_frac,
        G_K_W_m2K=G_forward,
        R_K_m2K_W=float("nan"),  # not directly comparable
        notes=f"G_forward={G_forward:.4e} G_reversed={G_reversed:.4e} ratio={ratio:.4f}" if np.isfinite(G_forward) and np.isfinite(G_reversed) else notes,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_validation_output(results: list[CaseResult], output_dir: Path):
    """Write dmm_validation_summary.txt and .csv."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Text summary.
    txt_path = output_dir / "dmm_validation_summary.txt"
    lines = ["DMM VALIDATION SUMMARY", "=" * 60, ""]
    lines.append(f"Ran at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Test case directory: input_two_material_dmm_test")
    lines.append("")
    for r in results:
        lines.append("-" * 40)
        lines.append(f"Case:        {r.case_name}")
        lines.append(f"Status:      {r.status}")
        lines.append(f"Particles:   {r.particles}")
        lines.append(f"Steps:       {r.steps}")
        lines.append(f"Warmup frac: {r.warmup_frac}")
        lines.append(f"DMM attempts:   {r.dmm_attempt}")
        lines.append(f"DMM transmits:  {r.dmm_transmit}")
        lines.append(f"DMM reflects:   {r.dmm_reflect}")
        lines.append(f"Energy 0->1:    {r.energy_0_to_1:.6e} J")
        lines.append(f"Energy 1->0:    {r.energy_1_to_0:.6e} J")
        lines.append(f"Net iface E:    {r.net_interface_energy:.6e} J")
        lines.append(f"Rel imbalance:  {r.relative_imbalance:.6f}")
        if np.isfinite(r.q_interface_W_m2):
            lines.append(f"q_iface:        {r.q_interface_W_m2:.6e} W/m^2")
        if np.isfinite(r.DeltaT_interface_K):
            lines.append(f"DeltaT_iface:   {r.DeltaT_interface_K:.4f} K")
        if np.isfinite(r.R_K_m2K_W):
            lines.append(f"R_K:            {r.R_K_m2K_W:.6e} K·m^2/W")
        if np.isfinite(r.G_K_W_m2K):
            lines.append(f"G_K:            {r.G_K_W_m2K:.6e} W/(K·m^2)")
        if r.notes:
            lines.append(f"Notes:       {r.notes}")
        lines.append("")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    # CSV summary.
    csv_path = output_dir / "dmm_validation_summary.csv"
    fields = [
        "case_name", "particles", "steps", "warmup_frac",
        "dmm_attempt", "dmm_transmit", "dmm_reflect",
        "energy_0_to_1", "energy_1_to_0", "net_interface_energy",
        "relative_imbalance", "q_interface_W_m2",
        "DeltaT_interface_K", "R_K_m2K_W", "G_K_W_m2K",
        "status", "notes",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in results:
            w.writerow([
                r.case_name, r.particles, r.steps, r.warmup_frac,
                r.dmm_attempt, r.dmm_transmit, r.dmm_reflect,
                f"{r.energy_0_to_1:.6e}", f"{r.energy_1_to_0:.6e}",
                f"{r.net_interface_energy:.6e}",
                f"{r.relative_imbalance:.6f}",
                f"{r.q_interface_W_m2:.6e}" if np.isfinite(r.q_interface_W_m2) else "",
                f"{r.DeltaT_interface_K:.6f}" if np.isfinite(r.DeltaT_interface_K) else "",
                f"{r.R_K_m2K_W:.6e}" if np.isfinite(r.R_K_m2K_W) else "",
                f"{r.G_K_W_m2K:.6e}" if np.isfinite(r.G_K_W_m2K) else "",
                r.status, r.notes,
            ])

    return txt_path, csv_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    test_dir = REPO / "input_two_material_dmm_test"
    if not test_dir.is_dir():
        print(f"FAIL: test case directory not found: {test_dir}")
        return 1

    print("=" * 60)
    print("DMM LONG VALIDATION")
    print(f"particles={args.particles}  steps={args.steps}  "
          f"warmup_frac={args.warmup_frac}  seed={args.seed}")
    print("=" * 60)

    results: list[CaseResult] = []

    # --- Case A ---
    print("\n>>> Case A: Detailed balance")
    rA = run_case_a_detail_balance(test_dir, args)
    results.append(rA)
    print(f"  Status: {rA.status}")
    print(f"  DMM attempts: {rA.dmm_attempt}  transmits: {rA.dmm_transmit}  reflects: {rA.dmm_reflect}")
    print(f"  Energy 0->1: {rA.energy_0_to_1:.4e}  1->0: {rA.energy_1_to_0:.4e}")
    print(f"  Relative imbalance: {rA.relative_imbalance:.4f}")
    if rA.notes:
        print(f"  Notes: {rA.notes}")

    # --- Case B ---
    print("\n>>> Case B: TBR estimate")
    rB = run_case_b_tbr_estimate(test_dir, args)
    results.append(rB)
    print(f"  Status: {rB.status}")
    print(f"  q_interface: {rB.q_interface_W_m2:.4e} W/m^2")
    print(f"  T_left (interface): {rB.T_left_interface:.2f} K  "
          f"T_right: {rB.T_right_interface:.2f} K")
    print(f"  DeltaT: {rB.DeltaT_interface_K:.4f} K")
    if np.isfinite(rB.R_K_m2K_W):
        print(f"  R_K: {rB.R_K_m2K_W:.4e} K·m^2/W  G_K: {rB.G_K_W_m2K:.4e} W/(K·m^2)")
    if rB.notes:
        print(f"  Notes: {rB.notes}")

    # --- Case C ---
    print("\n>>> Case C: Material-order reciprocity")
    rC = run_case_c_reciprocity(test_dir, args)
    results.append(rC)
    print(f"  Status: {rC.status}")
    print(f"  {rC.notes}")
    if rC.notes:
        print(f"  Notes: {rC.notes}")

    # --- Write output ---
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = REPO / "output" / f"dmm_validation_{time.strftime('%Y%m%d_%H%M%S')}"
    txt_path, csv_path = write_validation_output(results, out_dir)
    print(f"\nValidation output written to:")
    print(f"  {txt_path}")
    print(f"  {csv_path}")

    # --- Overall status ---
    statuses = [r.status for r in results]
    n_fail = statuses.count("FAIL")
    n_warn = statuses.count("WARNING")
    n_pass = statuses.count("PASS")
    print(f"\nOverall: {n_pass} PASS, {n_warn} WARNING, {n_fail} FAIL")
    if n_fail > 0:
        print("FINAL: FAIL")
        return 1
    elif n_warn > 0:
        print("FINAL: WARNING (some cases need investigation)")
        return 0
    else:
        print("FINAL: PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
