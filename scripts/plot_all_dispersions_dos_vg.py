import json, sys; sys.path.insert(0, '/home/ic/3dmc_Si_ylx_mod/Phonon_MC_py')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

NEXTGEN = Path('/home/ic/3dmc_Si_ylx_mod/Phonon_MC_py/materials_data/mp_raw_nextgen')
PLOTS = Path('/home/ic/3dmc_Si_ylx_mod/Phonon_MC_py/materials_data/plots')
PLOTS.mkdir(parents=True, exist_ok=True)

MATS = {
    'Cu':    {'vL': 4700, 'vT': 2300},
    'TiN':   {'vL': 9000, 'vT': 5500},
    'SiO2':  {'vL': 5800, 'vT': 3800},
    'Si3N4': {'vL': 8000, 'vT': 5000},
    'HfO2':  {'vL': 5000, 'vT': 3000},
}

for label, props in MATS.items():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    ax_bs, ax_dos, ax_vg = axes
    
    # --- Panel 1: Phonon bandstructure ω(q) ---
    for d in NEXTGEN.glob(f'{label}_*'):
        bs_files = list(d.glob('*_bandstructure_*.json'))
        dos_files = list(d.glob('*_dos_*.json'))
        if not bs_files or not dos_files: continue
        
        with open(bs_files[0]) as f:
            bs = json.load(f)
        freqs_bs = bs.get('frequencies', [])
        qpoints = bs.get('qpoints', [])
        labels_dict = bs.get('labels_dict', {})
        
        # Compute q-path distance (Å⁻¹)
        dist = [0.0]
        for i in range(1, len(qpoints)):
            dq = np.linalg.norm(np.array(qpoints[i]) - np.array(qpoints[i-1]))
            dist.append(dist[-1] + dq)
        q_path = np.array(dist)
        
        n_b = len(freqs_bs)
        colors = plt.cm.tab20(np.linspace(0, 1, n_b))
        for b in range(n_b):
            ax_bs.plot(q_path, freqs_bs[b], linewidth=0.6, color=colors[b % 20])
        
        # High-symmetry point labels
        for name, coords in labels_dict.items():
            # Find closest q-point
            qp_arr = np.array(qpoints)
            label_pos = np.array(coords)
            dists = np.linalg.norm(qp_arr - label_pos, axis=1)
            idx = np.argmin(dists)
            ax_bs.axvline(x=q_path[idx], color='gray', linestyle=':', alpha=0.5)
            ax_bs.text(q_path[idx], ax_bs.get_ylim()[1]*0.95, name,
                      ha='center', fontsize=7, color='gray')
        
        ax_bs.set_xlabel('q-path distance (Å⁻¹)')
        ax_bs.set_ylabel('Frequency (THz)')
        ax_bs.set_title(f'{label}: Phonon bandstructure ({n_b} branches)')
        ax_bs.set_ylim(bottom=0)
        
        # --- Panel 2: Phonon DOS ---
        with open(dos_files[0]) as f:
            dd = json.load(f)
        f_dos = np.array(dd['frequencies'])
        dos_val = np.array(dd['densities'])
        ax_dos.fill_between(f_dos, dos_val, alpha=0.5, color='#0f4c81')
        ax_dos.plot(f_dos, dos_val, color='#0f4c81', linewidth=1.2)
        ax_dos.set_xlabel('Frequency (THz)')
        ax_dos.set_ylabel('DOS (states/THz/cell)')
        ax_dos.set_title(f'{label}: Phonon DOS (MP total)')
        
        # --- Panel 3: Effective vg(ω) from isotropic model ---
        omega = 2 * np.pi * 1e12 * np.maximum(f_dos, 1e-6)
        dos_SI = np.maximum(dos_val, 1e-30) / (2 * np.pi * 1e12)  # per (rad/s)
        vL, vT = props['vL'], props['vT']
        v_D_debye = (3 / (1/vL**3 + 2/vT**3))**(1/3)
        v_iso = np.where(dos_SI > 0,
            (3 * omega**2 / (2 * np.pi**2 * dos_SI))**(1/3), 0)
        v_eff = np.minimum(v_iso, v_D_debye)
        v_eff = np.clip(v_eff, 200, v_D_debye)
        
        ax_vg.plot(f_dos, v_eff, color='#bf360c', linewidth=1.5, label='v_eff(ω) from isotropic DOS')
        ax_vg.axhline(y=v_D_debye, color='gray', linestyle='--', alpha=0.6,
                     label=f'Debye v_D = {v_D_debye:.0f} m/s')
        ax_vg.set_xlabel('Frequency (THz)')
        ax_vg.set_ylabel('Effective group velocity (m/s)')
        ax_vg.set_title(f'{label}: Effective vg from isotropic DOS inversion')
        ax_vg.legend(fontsize=8)
        ax_vg.set_ylim(bottom=0)
        
        out = PLOTS / f'{label}_dispersion_dos_vg.png'
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'{label}: saved {out}')
        break

print(f'\nPlots saved to: {PLOTS}')
