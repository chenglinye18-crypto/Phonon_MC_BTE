# Legacy MC Inputs

This directory contains historical IGZO Monte Carlo simulation inputs:

- `input_y*`: Length and temperature sweep cases (L=10–1120 nm, T=300/323/373 K)
- `in_plane_inputs/`: Width-dependent in-plane transport studies (w=2–400 nm)
- `in_plane_scripts/`: Scripts for in-plane scattering cases
- `tmp_*`: Temporary/intermediate test runs
- `run_*.sh`: Legacy shell batch runners for sweeps
- `output_cu_tin_tbc/`: Cu/TiN TBC analysis output (TBC model, not MC)
- `phonon_q_omega_vg_really63.txt`: Extra phonon dispersion data

These are preserved for reference but not needed for the current multi-material + DMM + bulk kappa pipeline.

The active reference case is `input11/` (IGZO single-material).
The multi-material DMM test case is `input_two_material_dmm_test/`.
