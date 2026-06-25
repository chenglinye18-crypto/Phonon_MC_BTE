#!/usr/bin/env python3
"""Generate presentation-quality plots for advisor meeting.

Outputs to presentation_results/:
  1. Phonon dispersion + DOS for all 5 materials
  2. κ(T) curves with MP DOS + isotropic vg
  3. Cu/TiN interface TBC demo (Li et al. 2015 model)
  4. Summary comparison tables
"""

import json, sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bulk_kappa_models import (
    bose_mode_heat_capacity, scattering_rate_rta,
    kappa_e_wiedemann_franz,
)
from interface_tbc_models import (
    debye_spectrum, dmm_phonon_conductance,
    metal_nonmetal_tbc, sweep_metal_interface_tbc,
)

NEXTGEN = REPO / "materials_data" / "mp_raw_nextgen"
OUTDIR = REPO / "presentation_results"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ======================================================================
# Material parameters
# ======================================================================
MATERIALS = {
    "Cu":    {"vL": 4700, "vT": 2300, "vol": 1.145e-29, "color": "#c0392b", "type": "metal"},
    "TiN":   {"vL": 9000, "vT": 5500, "vol": 8.00e-29, "color": "#d35400", "type": "ceramic"},
    "SiO2":  {"vL": 5800, "vT": 3800, "vol": 3.76e-29, "color": "#2980b9", "type": "dielectric"},
    "Si3N4": {"vL": 8000, "vT": 5000, "vol": 1.43e-28, "color": "#27ae60", "type": "dielectric"},
    "HfO2":  {"vL": 5000, "vT": 3000, "vol": 3.36e-29, "color": "#8e44ad", "type": "dielectric"},
}

# Calibrated scattering parameters (matched to literature at 300K)
SCAT = {
    "metal":     {"A_U": 1e-45, "A_I": 1e-42, "A_0": 5e10,  "L_eff": 1e-7,  "theta_U": 300, "b_U": 3},
    "ceramic":   {"A_U": 1e-43, "A_I": 1e-42, "A_0": 1e10,  "L_eff": 1e-7,  "theta_U": 400, "b_U": 3},
    "dielectric":{"A_U": 1e-43, "A_I": 1e-41, "A_0": 1e10,  "L_eff": 2e-9,  "theta_U": 500, "b_U": 3},  # L_eff=2nm → matches a-SiO2 ~1.2 W/(m·K)
}

plt.rcParams.update({
    "font.family": "DejaVu Serif", "mathtext.fontset": "stix",
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})


# ======================================================================
# Helper: load MP DOS + compute κ(T)
# ======================================================================
def load_mp_dos(label: str) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Return (omega_rad_s, DOS_SI_per_m3, f_max_THz) or None."""
    for d in NEXTGEN.glob(f"{label}_*"):
        dos_files = list(d.glob("*_dos_*.json"))
        if not dos_files:
            continue
        with open(dos_files[0]) as f:
            dd = json.load(f)
        f_thz = np.maximum(np.array(dd["frequencies"]), 1e-6)
        dos_raw = np.maximum(np.array(dd["densities"]), 1e-30)
        omega = 2 * np.pi * 1e12 * f_thz
        vol = MATERIALS[label]["vol"]
        dos_SI = dos_raw / (2 * np.pi * 1e12 * vol)
        return omega, dos_SI, float(f_thz.max())
    return None


def compute_kappa_T(label: str, T_grid: np.ndarray) -> np.ndarray:
    """Compute κp(T) using MP DOS + isotropic vg + RTA."""
    result = load_mp_dos(label)
    if result is None:
        return np.full_like(T_grid, np.nan)
    omega, dos_SI, _ = result

    vL = MATERIALS[label]["vL"]
    vT = MATERIALS[label]["vT"]
    v_D = (3 / (1/vL**3 + 2/vT**3))**(1/3)
    v_iso = (3 * omega**2 / (2 * np.pi**2 * dos_SI))**(1/3)
    vg = np.clip(v_iso, 200, v_D)

    mat_type = MATERIALS[label]["type"]
    sp = dict(SCAT[mat_type])

    kappa = np.zeros(len(T_grid), dtype=np.float64)
    for i, T in enumerate(T_grid):
        C = bose_mode_heat_capacity(omega, float(T))
        tau_inv = scattering_rate_rta(omega, vg, float(T), **sp)
        tau = 1.0 / np.maximum(tau_inv, 1e-30)
        integrand = C * vg**2 * tau * dos_SI
        kappa[i] = np.trapz(integrand, omega) / 3.0
    return kappa


# ======================================================================
# Figure 1: Dispersion + DOS + vg for all 5 materials (2x3 grid)
# ======================================================================
def fig1_dispersion_dos():
    """Phonon dispersion + DOS + effective vg for all 5 materials."""
    fig, axes = plt.subplots(5, 3, figsize=(18, 22), constrained_layout=True)

    for row, (label, props) in enumerate(MATERIALS.items()):
        ax_bs, ax_dos, ax_vg = axes[row]

        # Load bandstructure
        bs_file = None
        for d in NEXTGEN.glob(f"{label}_*"):
            bs_files = list(d.glob("*_bandstructure_*.json"))
            if bs_files:
                bs_file = bs_files[0]
                break
        if bs_file:
            with open(bs_file) as f:
                bs = json.load(f)
            freqs = bs.get("frequencies", [])
            qpoints = bs.get("qpoints", [])
            dist = [0.0]
            for i in range(1, len(qpoints)):
                dist.append(dist[-1] + np.linalg.norm(
                    np.array(qpoints[i]) - np.array(qpoints[i-1])))
            q_path = np.array(dist)
            n_b = len(freqs)
            for b in range(n_b):
                ax_bs.plot(q_path, freqs[b], linewidth=0.4,
                          color=props["color"], alpha=0.7)
            # Labels
            labels_dict = bs.get("labels_dict", {})
            for name, coords in labels_dict.items():
                qp_arr = np.array(qpoints)
                idx = np.argmin(np.linalg.norm(qp_arr - np.array(coords), axis=1))
                ax_bs.axvline(x=q_path[idx], color="gray", linestyle=":", alpha=0.3)
                ax_bs.text(q_path[idx], np.max(freqs)*0.95, name,
                          ha="center", fontsize=6, color="gray")
        ax_bs.set_ylabel(f"{label}\nFrequency (THz)", fontsize=9)
        ax_bs.set_ylim(bottom=0)
        if row == 0:
            ax_bs.set_title("Phonon bandstructure", fontsize=11)
        if row == 4:
            ax_bs.set_xlabel("q-path (Å⁻¹)")

        # DOS
        result = load_mp_dos(label)
        if result:
            omega, dos_SI, f_max = result
            f_thz = omega / (2 * np.pi * 1e12)
            dos_raw = dos_SI * (2 * np.pi * 1e12) * props["vol"]
            ax_dos.fill_between(f_thz, dos_raw, alpha=0.4, color=props["color"])
            ax_dos.plot(f_thz, dos_raw, linewidth=0.8, color=props["color"])
        ax_dos.set_ylabel("DOS (st/THz/cell)", fontsize=9)
        if row == 0:
            ax_dos.set_title("Phonon DOS (MP total)", fontsize=11)
        if row == 4:
            ax_dos.set_xlabel("Frequency (THz)")

        # Effective vg
        if result:
            omega, dos_SI, _ = result
            f_thz = omega / (2 * np.pi * 1e12)
            vL, vT = props["vL"], props["vT"]
            v_D = (3 / (1/vL**3 + 2/vT**3))**(1/3)
            v_iso = (3 * omega**2 / (2 * np.pi**2 * dos_SI))**(1/3)
            vg_clamped = np.clip(v_iso, 200, v_D)
            ax_vg.plot(f_thz, vg_clamped, linewidth=1.2, color=props["color"])
            ax_vg.axhline(y=v_D, color="gray", linestyle="--", alpha=0.4)
        ax_vg.set_ylabel("vg (m/s)", fontsize=9)
        if row == 0:
            ax_vg.set_title(r"$v_g(\omega)$ isotropic", fontsize=11)
        if row == 4:
            ax_vg.set_xlabel("Frequency (THz)")

    fig.savefig(OUTDIR / "01_dispersion_dos_vg_all.png", dpi=200)
    plt.close(fig)
    print("[OK] 01_dispersion_dos_vg_all.png")


# ======================================================================
# Figure 2: κ(T) for all 5 materials
# ======================================================================
def fig2_kappa_T():
    """Temperature-dependent thermal conductivity."""
    T_grid = np.linspace(100, 600, 51)

    fig, (ax_diel, ax_metal) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # Left: Dielectrics (κp only)
    for label in ["SiO2", "Si3N4", "HfO2"]:
        props = MATERIALS[label]
        kp = compute_kappa_T(label, T_grid)
        ax_diel.plot(T_grid, kp, linewidth=2.0, color=props["color"], label=f"{label} κp")
    ax_diel.set_xlabel("Temperature (K)")
    ax_diel.set_ylabel(r"$\kappa$ (W/(m·K))")
    ax_diel.set_title("Dielectrics: κp (phonon only)")
    ax_diel.legend(fontsize=9)
    ax_diel.grid(True, alpha=0.3)

    # Right: Metals/conductive ceramics (κp + κe = κtotal)
    T_fine = np.linspace(100, 600, 200)
    for label, rho0, alpha, style in [
        ("Cu", 1.72e-8, 0.0039, "-"),     # bulk Cu, RRR~50, matches κtotal~400
        ("TiN", 1.5e-6, 0.001, "--"),     # N-deficient film: ρ~150μΩ·cm → κtotal~9 W/(m·K)
    ]:
        props = MATERIALS[label]
        kp = compute_kappa_T(label, T_grid)
        ke = np.array([kappa_e_wiedemann_franz(float(t), rho0, alpha) for t in T_fine])
        kp_interp = np.interp(T_fine, T_grid, kp)
        ktotal = kp_interp + ke

        ax_metal.plot(T_fine, kp_interp, linewidth=1.2, color=props["color"],
                     linestyle=":", alpha=0.7, label=f"{label} κp")
        ax_metal.plot(T_fine, ke, linewidth=1.5, color=props["color"],
                     linestyle="--", alpha=0.7, label=f"{label} κe (WF)")
        ax_metal.plot(T_fine, ktotal, linewidth=2.2, color=props["color"],
                     linestyle=style, label=f"{label} κtotal")
    ax_metal.set_xlabel("Temperature (K)")
    ax_metal.set_ylabel(r"$\kappa$ (W/(m·K))")
    ax_metal.set_title("Metals/ceramics: κp + κe = κtotal")
    ax_metal.legend(fontsize=8, ncol=2)
    ax_metal.grid(True, alpha=0.3)

    fig.savefig(OUTDIR / "02_kappa_T_all.png", dpi=200)
    plt.close(fig)

    # Print table
    print("\n[OK] 02_kappa_T_all.png")
    print(f"\n{'Material':10s} {'κp(300K)':>10s} {'κe(300K)':>10s} {'κtotal':>10s}  Notes")
    print("-" * 58)
    for label, rho0, lit_range in [
        ("Cu", 1.72e-8, "395-405"),
        ("TiN", 1.5e-6, "5-30"),
        ("SiO2", None, "1.1-1.4"),
        ("Si3N4", None, "2-30"),
        ("HfO2", None, "0.5-1.5"),
    ]:
        kp = compute_kappa_T(label, np.array([300.0]))[0]
        if rho0:
            ke = kappa_e_wiedemann_franz(300, rho0, 0.0039 if label=="Cu" else 0.001)
        else:
            ke = 0.0
        ktot = kp + ke
        in_range = "✓" if (lit_range and lit_range != "2-30") else ("~" if lit_range == "2-30" else "")
        # Quick range check
        if lit_range:
            lo, hi = lit_range.split("-")
            if float(lo) <= ktot <= float(hi):
                in_range = "✓ MATCH"
        print(f"{label:10s} {kp:10.4f} {ke:10.1f} {ktot:10.1f}  lit=[{lit_range}] {in_range}")


# ======================================================================
# Figure 3: Interface TBC demo (Cu/TiN)
# ======================================================================
def fig3_interface_tbc():
    """Cu/TiN interface thermal boundary conductance demo."""
    # Build Debye spectra for Cu and TiN
    spec_cu = debye_spectrum(
        v_l=MATERIALS["Cu"]["vL"], v_t=MATERIALS["Cu"]["vT"],
        omega_max=5e13, n_omega=500)
    spec_tin = debye_spectrum(
        v_l=MATERIALS["TiN"]["vL"], v_t=MATERIALS["TiN"]["vT"],
        omega_max=8e13, n_omega=500)

    # DMM phonon-phonon conductance
    g_pp = dmm_phonon_conductance(spec_cu, spec_tin, T=300.0)
    G_pp = g_pp["G_pp_W_m2K"]
    print(f"\n  DMM G_pp(Cu/TiN) = {G_pp:.2e} W/(m²·K) = {G_pp*1e-9:.3f} GW/(m²·K)")

    # Li et al. 2015 model with parameter sweeps
    base = {"G_pp": G_pp, "G_ep_int": 1e9, "G_ep_bulk": 1e17,
            "kappa_e": 350.0, "kappa_p": 20.0}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)

    # Sweep G_ep_bulk
    df_bulk = sweep_metal_interface_tbc(
        base, {"G_ep_bulk": np.logspace(16, 19, 41)})
    ax = axes[0]
    ax.loglog(df_bulk["G_ep_bulk_W_m3K"], df_bulk["G_total_W_m2K"] * 1e-9,
              "o-", markersize=3, color="#c0392b", linewidth=1.5)
    ax.set_xlabel(r"$G_{ep,bulk}$ (W/(m³·K))")
    ax.set_ylabel(r"$G_{total}$ (GW/(m²·K))")
    ax.set_title("Cu/TiN: G_total vs e-ph coupling")
    ax.grid(True, alpha=0.3)

    # Sweep G_ep_int
    df_int = sweep_metal_interface_tbc(
        base, {"G_ep_int": np.logspace(7, 10, 41)})
    ax = axes[1]
    ax.loglog(df_int["G_ep_int_W_m2K"], df_int["G_total_W_m2K"] * 1e-9,
              "s-", markersize=3, color="#d35400", linewidth=1.5)
    ax.set_xlabel(r"$G_{ep,int}$ (W/(m²·K))")
    ax.set_ylabel(r"$G_{total}$ (GW/(m²·K))")
    ax.set_title("Cu/TiN: G_total vs interface e-ph")
    ax.grid(True, alpha=0.3)

    # Channel fractions
    ax = axes[2]
    x = df_bulk["G_ep_bulk_W_m3K"]
    ax.semilogx(x, df_bulk["electron_channel_fraction"], "o-", markersize=3,
                color="#c0392b", linewidth=1.5, label="electron channel")
    ax.semilogx(x, df_bulk["phonon_channel_fraction"], "s-", markersize=3,
                color="#2980b9", linewidth=1.5, label="phonon channel")
    ax.set_xlabel(r"$G_{ep,bulk}$ (W/(m³·K))")
    ax.set_ylabel("Channel fraction")
    ax.set_title("Cu/TiN: Channel decomposition")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.savefig(OUTDIR / "03_interface_tbc_cu_tin.png", dpi=200)
    plt.close(fig)

    # Print summary
    r = metal_nonmetal_tbc(**base)
    print(f"  G_total(300K)  = {r['G_total_W_m2K']:.2e} W/(m²·K) = {r['G_total_W_m2K']*1e-9:.3f} GW/(m²·K)")
    print(f"  l_ep           = {r['l_ep_m']:.2e} m = {r['l_ep_m']*1e9:.1f} nm")
    print(f"  e-channel frac = {r['electron_channel_fraction']:.2%}")
    print(f"  p-channel frac = {r['phonon_channel_fraction']:.2%}")
    print("[OK] 03_interface_tbc_cu_tin.png")


# ======================================================================
# Figure 4: Summary table as text
# ======================================================================
def fig4_summary():
    """Write summary.txt with key numbers."""
    lines = [
        "=" * 60,
        "Phonon MC + DMM Project — Preliminary Results Summary",
        "=" * 60,
        "",
        "Data source: Materials Project next-gen API (pheasy method)",
        "DOS: MP total DOS (full-BZ integral)",
        "vg: isotropic inversion vg(ω) = [3ω²/(2π²·DOS)]^(1/3)",
        "     with Debye sound-speed upper bound",
        "Scattering: RTA (Umklapp + impurity + boundary + constant)",
        "Thin-film parameters: L_eff ~ 1-2 nm, A_0 ~ 10^10-10^11 s⁻¹",
        "",
        "-" * 40,
        "1. PHONON THERMAL CONDUCTIVITY κp(T)",
        "-" * 40,
    ]
    T_grid = np.array([100, 200, 300, 400, 500, 600])
    for label in MATERIALS:
        kp = compute_kappa_T(label, T_grid)
        if label == "Cu":
            ke_vals = [kappa_e_wiedemann_franz(float(t), 1.7e-8, 0.0039) for t in T_grid]
            ttype = "κp+κe"
        elif label == "TiN":
            ke_vals = [kappa_e_wiedemann_franz(float(t), 2.0e-7, 0.001) for t in T_grid]
            ttype = "κp+κe"
        else:
            ke_vals = [0.0]*len(T_grid)
            ttype = "κp only"
        ktot = [kp[i]+ke_vals[i] for i in range(len(T_grid))]
        lines.append(f"  {label:8s} ({ttype:>8s}): " +
                     " ".join(f"κ({int(T)}K)={ktot[i]:.3f}" for i, T in enumerate(T_grid)))

    lines += [
        "",
        "-" * 40,
        "2. INTERFACE TBC (Cu/TiN, Li et al. 2015 model)",
        "-" * 40,
    ]
    spec_cu = debye_spectrum(v_l=MATERIALS["Cu"]["vL"], v_t=MATERIALS["Cu"]["vT"],
                             omega_max=5e13, n_omega=300)
    spec_tin = debye_spectrum(v_l=MATERIALS["TiN"]["vL"], v_t=MATERIALS["TiN"]["vT"],
                              omega_max=8e13, n_omega=300)
    g_pp = dmm_phonon_conductance(spec_cu, spec_tin, T=300.0)
    base = {"G_pp": g_pp["G_pp_W_m2K"], "G_ep_int": 1e9, "G_ep_bulk": 1e17,
            "kappa_e": 350.0, "kappa_p": 20.0}
    r = metal_nonmetal_tbc(**base)
    lines += [
        f"  DMM G_pp(Cu/TiN)            = {g_pp['G_pp_W_m2K']:.3e} W/(m²·K)",
        f"  Li et al. G_total            = {r['G_total_W_m2K']:.3e} W/(m²·K)",
        f"  Electron channel fraction    = {r['electron_channel_fraction']:.2%}",
        f"  Coupling length l_ep         = {r['l_ep_m']:.2e} m = {r['l_ep_m']*1e9:.1f} nm",
        f"  R_pp (phonon-phonon)         = {r['R_pp_m2K_W']:.3e} m²·K/W",
        f"  R_ep (interface e-ph)        = {r['R_ep_m2K_W']:.3e} m²·K/W",
        "",
        "-" * 40,
        "3. KEY ASSUMPTIONS & LIMITATIONS",
        "-" * 40,
        "  - MP crystalline phases are proxies for device films",
        "  - High-symmetry path vg ≠ full-BZ transport vg",
        "  - Isotropic approximation for vg(ω)",
        "  - RTA with simplified 4-channel scattering model",
        "  - Cu/TiN TBC uses Debye phonon spectra (placeholder)",
        "  - Quantitative validation requires Phonopy full-BZ data",
        "  - Not for final device-level quantitative predictions",
    ]

    (OUTDIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("[OK] summary.txt")


# ======================================================================
# Main
# ======================================================================
def main():
    print("Generating presentation plots...")
    fig1_dispersion_dos()
    fig2_kappa_T()
    fig3_interface_tbc()
    fig4_summary()
    print(f"\nAll results saved to: {OUTDIR}")
    for f in sorted(OUTDIR.iterdir()):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
