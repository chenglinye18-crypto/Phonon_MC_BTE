# Changelog

## v0.2-dmm-multimaterial

- **Multi-material phonon MC infrastructure** — generic material discovery from dispersion files, `material_key()` alias system, `ParticleBlock.material_id` field.
- **Per-material spectral grids** — shared frequency bins across all materials, per-material `build_spectral_grid()`, DOS, group velocity, heat capacity.
- **Per-material scattering parameters** — `[scattering.MATERIAL]` sections in `solver_params.toml`, per-material Numba-jitted relaxation time computation.
- **Per-material energy-temperature LUTs** — per-material `build_E_T_lookup()`, `build_pp_scattering_T_lookup()`, dispatched by `cell_material_index`.
- **Per-material PP scattering resampling** — `particle_scattering()` groups particles by `(material_id, cell)`, uses correct per-material spec.
- **DMM interface transport** — transmission probability `T(ω)` from branch-resolved `DOS × |vg|`, diffuse hemisphere reflection, position/cell restoration.
- **Branch remapping on DMM transmission** — phonon branch resampled in target material weighted by `DOS_b(ω) × |vg_b(ω)|`; frequency, energy, and direction preserved.
- **DMM empirical transmission diagnostics** — per-(material_pair, omega_bin) attempt/transmit/reflect stats accumulated across steps, written to `dmm_bin_stats.txt` with `empirical_T` vs `table_T` comparison.
- **Multi-material output** — `write_periodic_output()` supports spec list, branch stats include `material_id` and `material_key` columns.
- **Multi-material time step CFL** — uses maximum `vg_max` across all material specs.
- **Multi-material `initial_particles_fixed`** — global `E_eff` from total energy weight across all materials with proportional sampling.
- **Zero-energy-weight fallback** — deviational mode at uniform temperature falls back to absolute mode so particles are still generated.
- **Two-material test case** — `input_two_material_dmm_test/` with Si+IGZO slab, per-material scattering, short steps.
- **Smoke test** — `scripts/smoke_test_multimaterial_dmm.py` verifies multi-material path, material_ids, DMM stats, branch stats.
- **Detailed balance test** — `scripts/test_dmm_detailed_balance.py` uniform-temperature slab, checks net interface flux ≈ 0.
- **TBR sanity check** — `scripts/test_dmm_tbr_estimate.py` gradient-driven interface, estimates Kapitza resistance.
- **Long validation script** — `scripts/run_dmm_validation_long.py` with CLI arguments, three cases (detailed balance, TBR, reciprocity), warmup/time-averaging, CSV/TXT output.
- **Backward compatible** — single-material IGZO cases and scripts run unchanged.

## v1.0 (prior)

- Initial Python port of MATLAB phonon BTE Monte Carlo code.
- Single-material IGZO simulation with deviational mode.
- `ldg.txt` / `lgrid.txt` geometry parsing.
- Numba-jitted scattering rate computation.
- Batch runner for temperature-labeled MC cases.
