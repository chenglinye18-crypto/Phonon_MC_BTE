#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

temps=(323 373)
lengths=(1120 560)

grid_nm_for_length() {
  local length_nm="$1"
  case "$length_nm" in
    280|140) echo 10 ;;
    70|50|20|10) echo 1 ;;
    *) echo "unsupported length: $length_nm" >&2; return 1 ;;
  esac
}

for length_nm in "${lengths[@]}"; do
  grid_nm="$(grid_nm_for_length "$length_nm")"
  for temp_k in "${temps[@]}"; do
    input_dir="input_y${length_nm}nm_${grid_nm}nm_Eeff5e-19_T${temp_k}K"
    run_tag="run_${input_dir}"
    echo "[run] ${input_dir} -> ${run_tag}"
    conda run -n 3dmc python run_current_case.py "$ROOT/${input_dir}" --run-tag "${run_tag}"
  done
done
