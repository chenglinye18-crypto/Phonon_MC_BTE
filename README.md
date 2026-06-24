# Phonon_MC_BTE

A phonon Boltzmann Transport Equation (BTE) Monte Carlo simulation framework supporting both single-material and multi-material phonon transport with Diffuse Mismatch Model (DMM) interface scattering.

## Project Overview

This codebase implements an energy-based phonon particle Monte Carlo method to solve the phonon BTE. It supports:

- Single-material phonon transport with material-specific phonon dispersion, density of states (DOS), group velocity, and scattering rates.
- Multi-material simulations with distinct regions in a single mesh, each assigned a different material.
- Cross-material phonon transport via the **Diffuse Mismatch Model (DMM)**, including branch remapping, frequency-preserving transmission, and diffuse reflection.

The code is a research prototype developed at Peking University (original MATLAB code by Chenglin Ye) and ported to Python with multi-material extensions.

## Current Features

- **Energy-based phonon particle Monte Carlo** — deviational and absolute mode.
- **Material parsing from `ldg.txt`** — `region` lines carry material names (e.g., `IGZO`, `Si`, `SILICON`).
- **Material-specific phonon dispersion** — loaded from `phonon_dispersion_<NAME>.txt` files with branch metadata in comment headers.
- **Material-specific scattering parameters** — per-material `[scattering.<NAME>]` sections in `solver_params.toml`.
- **Temperature-energy LUT** — per-material internal energy ↔ temperature lookup tables.
- **Multi-material `ParticleBlock.material_id`** — every particle tracks which material it belongs to.
- **DMM interface transport** — transmission probability `T(ω)` from projected DOS × |vg|, diffuse reflection.
- **Branch remapping on DMM transmission** — phonon branch resampled in the target material weighted by `DOS_b(ω) × |vg_b(ω)|`.
- **DMM empirical transmission diagnostics** — per-(material_pair, omega_bin) attempt/transmit/reflect statistics written to `dmm_bin_stats.txt`.
- **Validation tests** — detailed balance check, TBR sanity estimation, material-order reciprocity.
- **Smoke tests** — fast multi-material integration test.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick smoke test (multi-material DMM, ~10 seconds)
python scripts/smoke_test_multimaterial_dmm.py

# Detailed balance test
python scripts/test_dmm_detailed_balance.py

# TBR sanity check
python scripts/test_dmm_tbr_estimate.py

# Long validation (adjust --particles and --steps as needed)
python scripts/run_dmm_validation_long.py --particles 20000 --steps 300 --warmup-frac 0.3

# Run a single-material case
python main.py
```

## Input Structure

Each simulation case is defined by a directory containing these files:

| File | Purpose |
|------|---------|
| `ldg.txt` | Geometry layout — `region` lines specify material; `planerule`/`lanerule` define boundary conditions; `RESERVOIR` defines fixed-temperature regions |
| `lgrid.txt` | Mesh grid definition per axis (`X N: {anchor_list}`) |
| `solver_params.toml` | Solver configuration — particle count, time step, temperature range, per-material scattering |
| `phonon_dispersion_<NAME>.txt` | Phonon dispersion data — columns: `branch_id, q(1/m), f(THz), vg(m/s)`. Branch metadata in header comments: `# branch_names=TA,TA,LA; degeneracy=1,1,1` |
| `initial_temperature.csv` | Initial temperature field — `idx, idy, idz, T_init` |
| `reference_temperature.txt` | Reference temperature for deviational mode — `idx, idy, idz, T_ref` |
| `volume_heat_source.txt` | (optional) Volumetric heat source |
| `heat_flux_monitors.txt` | (optional) Heat flux measurement planes |
| `input_manifest.txt` | (optional) Metadata |

### Multi-material example (`ldg.txt`)

```text
$Lx$ 0.06
$Ly$ 0.06
$Lz$ 0.06

region 0      0.03  0 $Ly$ 0 $Lz$  Si
region 0.03 $Lx$    0 $Ly$ 0 $Lz$  IGZO
```

### Per-material scattering (`solver_params.toml`)

```toml
[scattering.SILICON]
BL = 1.18e-24
BTN = 1.05e-12
BTU = 2.89e-18
tau_LTO_ps = 3.5
A_imp = 1.32e-45
C_imp = 0.0
PB_Tsi = 0
PB_bulk_L = 0
PB_bulk_F = 0

[scattering.IGZO]
BL = 3.0e-22
BTN = 1.0076e-12
BTU = 5.0e-15
A_imp = 1.0e-46
B_imp = 3.0e-22
C_imp = 1.0e9
```

A flat `[scattering]` section (without per-material sub-tables) applies to all materials for backward compatibility.

### Material aliases

Add a `[material_aliases]` section in `solver_params.toml` to map common short names to canonical keys:

```toml
[material_aliases]
Si = SILICON
Si_100 = SILICON
```

Built-in aliases: `Si` → `SILICON`, `Si_100` → `SILICON`, `Silicon_100` → `SILICON`.

## DMM Model

The Diffuse Mismatch Model is implemented as:

### Transmission probability

For a phonon of frequency ω crossing from material *i* to material *j*:

```
M_i(ω) = Σ_b DOS_{i,b}(ω) × |vg_{i,b}(ω)|
T_{i→j}(ω) = M_j(ω) / [M_i(ω) + M_j(ω)]
```

where `DOS_{i,b}(ω)` is the per-branch density of states and `|vg_{i,b}(ω)|` is the branch-resolved group velocity magnitude.

### Branch remapping on transmission

When a phonon is transmitted into a new material, its branch is resampled from the target material's branch distribution at the same frequency:

```
P_b ∝ DOS_b(ω) × |vg_b(ω)|
```

The phonon frequency `ω`, energy `E`, and propagation direction unit vector are preserved. Wavevector `q` and group velocity magnitude `|vg|` are recomputed from the target material's dispersion.

### Reflection

Reflected phonons are diffusely scattered back into the original material with a random direction in the hemisphere pointing into the original cell. The particle returns to its original cell.

## Validation

| Test | Script | Description |
|------|--------|-------------|
| Smoke test | `scripts/smoke_test_multimaterial_dmm.py` | Multi-material integration, DMM stats, material_id check |
| Detailed balance | `scripts/test_dmm_detailed_balance.py` | Uniform 300 K slab, net interface flux ≈ 0 |
| TBR sanity | `scripts/test_dmm_tbr_estimate.py` | 310/300 K gradient, estimate Kapitza resistance |
| Long validation | `scripts/run_dmm_validation_long.py` | Three cases: detailed balance, TBR estimate, reciprocity |

## Metal/nonmetal Interface TBC Analytical Model

In addition to the phonon MC + DMM framework for phonon-phonon interface transport, the repository includes an **analytical metal/nonmetal TBC model** based on Li et al. 2015, *"Thermal boundary conductance across metal-nonmetal interfaces: effects of electron-phonon coupling both in metal and at interface."*

### Motivation

Metal/nonmetal (or metal/ceramic) interface thermal conductance cannot be described by phonon DMM alone — electrons in the metal carry significant heat and couple to phonons both in the bulk metal and at the interface.

### Model

The total interface conductance is a parallel combination of electron-mediated and phonon-mediated channels:

```
G_total = 1/(R_e_m + R_ep) + 1/(R_p_m + R_pp)
```

where:

| Symbol | Meaning | Unit |
|--------|---------|------|
| `l_ep` | Electron-phonon coupling length: `(G_ep_bulk/κ_e + G_ep_bulk/κ_p)^(-1/2)` | m |
| `R_e_m` | Electron transport resistance in metal: `l_ep / κ_e` | m² K/W |
| `R_p_m` | Phonon transport resistance in metal: `l_ep / κ_p` | m² K/W |
| `R_ep` | Interface e-ph coupling resistance: `1 / G_ep_int` | m² K/W |
| `R_pp` | Interface ph-ph coupling resistance: `1 / G_pp` | m² K/W |
| `G_pp` | Phonon-phonon interface conductance (from DMM/Debye/MC/Phonopy) | W/(m² K) |
| `G_ep_int` | Interface electron-phonon conductance | W/(m² K) |
| `G_ep_bulk` | Bulk electron-phonon coupling constant | W/(m³ K) |
| `κ_e` | Electronic thermal conductivity of metal | W/(m K) |
| `κ_p` | Lattice (phonon) thermal conductivity of metal | W/(m K) |

### Usage

```python
from interface_tbc_models import *

# Build Debye placeholder spectra
spec_cu = debye_spectrum(v_l=4700, v_t=2300, omega_max=5e13)
spec_tin = debye_spectrum(v_l=9000, v_t=5500, omega_max=8e13)

# DMM phonon-phonon conductance
g = dmm_phonon_conductance(spec_cu, spec_tin, T=300)
G_pp = g["G_pp_W_m2K"]

# Li et al. 2015 full model
result = metal_nonmetal_tbc(
    G_pp=G_pp, G_ep_int=1e9, G_ep_bulk=1e17,
    kappa_e=350, kappa_p=20,
)
print(f"G_total = {result['G_total_W_m2K']:.2e} W/(m^2 K)")
```

### Cu/TiN Example

```bash
python scripts/sweep_cu_tin_tbc.py --temperature 300 --output-dir output_cu_tin_tbc
```

**⚠️ The Cu/TiN parameters are first-pass Debye placeholders.** Replace by literature, Materials Project, or Phonopy data for quantitative studies. TiN is a conductive ceramic — it has both electron and phonon contributions but is treated here as the phonon-accepting side in a simplified effective model.

### Scripts

| Script | Description |
|--------|-------------|
| `scripts/sweep_cu_tin_tbc.py` | Cu/TiN TBC parameter sweep with Debye placeholder spectra |
| `scripts/test_interface_tbc_models.py` | Unit tests for Debye, DMM, and Li et al. model |

## Materials Data Pipeline for Interface TBC

The `materials_data/` directory contains a pipeline for downloading and
processing Materials Project structural and phonon data for target materials
used in interface TBC calculations.

### Target materials

| Material | Role | MP crystalline proxy |
|----------|------|---------------------|
| Cu | metal interconnect | fcc Cu (mp-30) |
| TiN | barrier / conductive ceramic | rocksalt TiN (mp-492) |
| SiO₂ | device oxide | quartz or low-E crystalline (mp-7000) |
| Si₃N₄ | device nitride | α/β Si₃N₄ (mp-988) |
| HfO₂ | RRAM switching layer | monoclinic HfO₂ (mp-352) |

### Interface pairs

Cu\|TiN, TiN\|HfO₂, Cu\|HfO₂, TiN\|SiO₂, TiN\|Si₃N₄, HfO₂\|SiO₂, HfO₂\|Si₃N₄

### Next-gen phonon API (recommended)

Materials Project phonon data requires the **next-gen API** (`mpr.materials.phonon` sub-client) with `phonon_method="pheasy"`. The old `mp.get_phonon_bandstructure_by_material_id()` method may fail for materials that only have pheasy data.

```bash
pip install -U mp_api pymatgen monty
export MP_API_KEY="your_key"

# Step 0: Probe a single material
python scripts/mp_probe_phonon_nextgen.py --material-id mp-30 --label Cu --phonon-method pheasy

# Step 1: Search MP for candidates (uses old summary API)
python scripts/mp_search_target_materials.py

# Step 2: Batch probe phonon availability (top 3 candidates per formula)
python scripts/mp_probe_all_candidate_phonons_nextgen.py

# Step 3: Review selected_materials_with_phonons_nextgen.toml (manual)

# Step 4: Convert nextgen phonon bandstructure to project format
python scripts/mp_convert_phonons_to_project_format.py

# Step 5: Quality check (auto-detects nextgen vs legacy data)
python scripts/check_downloaded_phonon_data.py

# Step 6: Plot phonon data
python scripts/plot_downloaded_phonons.py

# Step 7: Compute interface G_pp
python scripts/compute_downloaded_interface_gpp.py --temperature 300
```

### Phonon availability (next-gen API, pheasy method)

| Material | mp-id | BS | DOS | FC | Branches | Q-points | Freq (THz) |
|----------|-------|-----|-----|-----|----------|----------|-------------|
| Cu | mp-30 | ✅ | ✅ | ✅ | 3 | 1010 | 0 – 7.97 |
| TiN | mp-492 | ✅ | ✅ | ✅ | 6 | 1010 | downloaded |
| SiO₂ | mp-7000 | ✅ | ✅ | ✅ | 27 | 1111 | downloaded |
| Si₃N₄ | mp-988 | ✅ | ✅ | ✅ | 42 | 909 | downloaded |
| HfO₂ | mp-352 | ✅ | ✅ | ✅ | probed | probed | downloaded |

### Data limitations

- **MP phonon data coverage is limited** — most of Cu, TiN, SiO₂, Si₃N₄,
  HfO₂ do not have DFPT phonon bandstructures in the current MP database.
  All 5 materials are marked `needs_phonopy` for phonon dispersion.
- **MP data is crystalline** — device SiO₂, Si₃N₄, HfO₂ are often amorphous
  or polycrystalline.  MP crystalline phases are first-pass proxies.
- **High-symmetry path vg is approximate** — group velocity from finite
  differences along MP bandstructure paths is not a full-BZ transport
  spectrum.  Quantitative TBC requires Phonopy with full-BZ sampling.
- **All structures are downloaded** — ready for Phonopy calculations as the
  next step.

## Known Limitations

- **DMM assumes elastic diffuse interface scattering** — no acoustic mismatch model (AMM), no specular reflection component.
- **No AGF** (Atomistic Green's Function) — interface properties come only from bulk phonon DOS.
- **No inelastic interface scattering** — phonon frequency is preserved across the interface.
- **TBR is estimated statistically** from the Monte Carlo flux and temperature jump — not imposed as a boundary condition.
- **Quantitative accuracy** requires convergence tests with larger particle counts and longer time averaging than the default smoke-test settings.
- **No phonon-phonon scattering calibration** for multi-material — scattering parameters must be fitted independently for each material.

## Suggested Citation / Project Status

This is a **research prototype**, not a production simulator. If you use this code in your research, please cite the original MATLAB code author (Chenglin Ye, Peking University) and note the Python port with multi-material extensions.

## License

See the repository license file.

## Version History

See [CHANGELOG.md](CHANGELOG.md).
