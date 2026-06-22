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
