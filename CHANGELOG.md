# Changelog

## v0.5-bulk-kappa-calibration

- **RTA bulk thermal conductivity model** (`bulk_kappa_models.py`): Bose heat capacity, scattering rate (Umklapp + impurity + boundary + constant), κp from spectrum integration, Wiedemann-Franz κe.
- **Temperature-dependent κp(T) from phonon spectra** — supports Debye and MP-processed table-driven spectra.
- **Scattering parameter calibration** (`calibrate_bulk_kappa_T.py`): fits A_U, A_I, A_0, L_eff to target κ(T) data using scipy least_squares.
- **Sensitivity sweep** (`sweep_bulk_kappa_sensitivity.py`): L_eff, A_0, A_U sweeps with diagnostic plots for all 5 materials.
- **Material-specific guidance**: Cu (metal, κe dominant), TiN (conductive ceramic, uncertain split), SiO₂/Si₃N₄/HfO₂ (dielectrics).
- **Template target data** and initial parameter configs provided.
- **17 unit tests** covering C_v, scattering rates, κp monotonicity, WF law, Cu electron warning.

## v0.4.1-nextgen-phonon-api

- **Next-gen Materials Project phonon API support** — migrated from old `mp.get_phonon_bandstructure_by_material_id()` to `mpr.materials.phonon` sub-client with `phonon_method="pheasy"`.
- **All 5 target materials confirmed with pheasy phonon data**: Cu (mp-30, 3 br), TiN (mp-492, 6 br), SiO₂ (mp-7000, 27 br), Si₃N₄ (mp-988, 42 br), HfO₂ (mp-352).
- **New probe script** (`mp_probe_phonon_nextgen.py`) — single-material probe with pheasy/dfpt method + latimer_munro/setyawan_curtarolo path type support.
- **Batch phonon availability scanner** (`mp_probe_all_candidate_phonons_nextgen.py`) — probes top-3 candidates per formula, incremental CSV saving, writes `selected_materials_with_phonons_nextgen.toml`.
- **Updated conversion** — `mp_convert_phonons_to_project_format.py` supports nextgen `frequencies` key (in addition to legacy `branches`), q-path from reciprocal coordinates → 1/m, imaginary mode detection + clipping.
- **Regression test** (`test_mp_phonon_nextgen_parsing.py`) — validates Cu mp-30 BS parsing, q-path generation, vg estimation, DOS reading. All 8 tests pass.
- **README updated** — next-gen API pipeline commands, phonon availability table for all 5 materials.
- Legacy `materials_data/mp_raw/` data preserved; nextgen data stored in `materials_data/mp_raw_nextgen/`.

## v0.4-materials-data-pipeline

- **Materials Project data pipeline** — new `materials_data/` directory with structured pipeline for downloading MP structures and phonon data.
- **Target material config** (`material_targets.toml`) — Cu, TiN, SiO₂, Si₃N₄, HfO₂ with 7 interface pairs.
- **MP candidate search** (`mp_search_target_materials.py`) — queries MP for each formula, saves candidate tables, auto-recommends mp-ids.
- **MP data download** (`mp_download_target_phonons.py`) — downloads structure (CIF), phonon bandstructure, phonon DOS, metadata; graceful fallback when phonon data is unavailable.
- **Phonon conversion** (`mp_convert_phonons_to_project_format.py`) — MP bandstructure → `phonon_dispersion_{label}.txt` with finite-difference vg estimates.
- **Quality check** (`check_downloaded_phonon_data.py`) — validates structures, dispersion files, frequency non-negativity, vg finiteness; outputs CSV/MD status report.
- **Phonon plotting** (`plot_downloaded_phonons.py`) — bandstructure, vg vs frequency, DOS plots per material.
- **Interface G_pp computation** (`compute_downloaded_interface_gpp.py`) — DMM G_pp for all target interface pairs using converted dispersion files.
- **Current status**: all 5 materials have structures; none have MP DFPT phonon data → all marked `needs_phonopy`.  Pipeline framework is complete and ready for Phonopy data ingestion.

## v0.3-interface-tbc-analytical

- **Analytical metal/nonmetal interface TBC model** — new independent module `interface_tbc_models.py`.
- **Debye spectrum generator** (`debye_spectrum()`) — branch-resolved DOS and vg from sound speeds, first-pass placeholder for Phonopy/MP data.
- **DMM transmission and conductance** (`dmm_transmission_from_spectra()`, `dmm_phonon_conductance()`) — works with any spectrum dict (Debye or MC), auto-interpolation to common omega grid.
- **Li et al. 2015 series-parallel resistor network** (`metal_nonmetal_tbc()`) — electron + phonon channel analysis with coupling length l_ep, channel fractions, and full resistance breakdown.
- **Parameter sweep function** (`sweep_metal_interface_tbc()`) — Cartesian sweep over any TBC parameter, outputs pandas DataFrame.
- **Cu/TiN placeholder sweep** (`scripts/sweep_cu_tin_tbc.py`) — Debye spectra, DMM G_pp, G_ep_bulk and G_ep_int sweeps, diagnostic plots and CSVs.
- **Unit tests** (`scripts/test_interface_tbc_models.py`) — Debye non-negativity, DMM T in [0,1], self-T≈0.5, monotonicity, input validation.

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
