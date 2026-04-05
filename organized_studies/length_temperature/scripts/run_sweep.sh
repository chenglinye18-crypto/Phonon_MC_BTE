#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

temps=(323 373)
lengths=(1120 560)

for length_nm in "${lengths[@]}"; do
  for temp_k in "${temps[@]}"; do
    input_dir="input_y${length_nm}nm_10nm_Eeff5e-19_T${temp_k}K"
    run_tag="run_${input_dir}"
    echo "[run] ${input_dir} -> ${run_tag}"
    conda run -n 3dmc python run_current_case.py "$ROOT/${input_dir}" --run-tag "${run_tag}"
  done
done
