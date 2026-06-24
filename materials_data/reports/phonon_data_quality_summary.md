# Phonon Data Quality Summary

| label | status | structure | BS | DOS | disp | branches | qpoints | f_range_THz |
|-------|--------|-----------|-----|------|------|----------|---------|-------------|
| Cu | structure_only_needs_phonopy | Y | N | N | N | 0 | 0 | N/A |
| HfO2 | structure_only_needs_phonopy | Y | N | N | N | 0 | 0 | N/A |
| Si3N4 | structure_only_needs_phonopy | Y | N | N | N | 0 | 0 | N/A |
| SiO2 | structure_only_needs_phonopy | Y | N | N | N | 0 | 0 | N/A |
| TiN | structure_only_needs_phonopy | Y | N | N | N | 0 | 0 | N/A |

## Status Legend
- `ready_for_first_pass_dmm`: dispersion file available, can compute G_pp
- `partial_dos_only`: only phonon DOS, no dispersion (needs Phonopy)
- `structure_only_needs_phonopy`: only structure (needs Phonopy for phonons)
- `failed`: data corrupted or unreadable