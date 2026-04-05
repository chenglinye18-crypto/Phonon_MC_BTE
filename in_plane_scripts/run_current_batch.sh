#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/home/ic/miniconda3/envs/3dmc/bin/python}"

INPUT_ROOT="$ROOT_DIR/in_plane_inputs"
MANIFEST_DIR="$ROOT_DIR/output/in_plane_current_batch"
MANIFEST_CSV="$MANIFEST_DIR/case_manifest.csv"

mkdir -p "$MANIFEST_DIR"

cat > "$MANIFEST_CSV" <<'EOF'
run_tag,input_dir,run_dir,width_nm,width_label,temperature_K,deltaT_K,transport_length_m,monitor_label
EOF

widths=(002 005 010 020 050)
temps=(300 323 373)

for width in "${widths[@]}"; do
  width_nm=$((10#$width))
  width_label="${width_nm} nm"
  for temp in "${temps[@]}"; do
    case_name="w${width}nm_t${temp}k_s5050_n100k"
    input_dir="$INPUT_ROOT/$case_name"
    run_tag="in_plane_${case_name}"
    run_dir="$ROOT_DIR/output/run_${run_tag}"
    printf '%s,%s,%s,%s,"%s",%s,10.0,1.120000000000e-06,flux_plane_004\n' \
      "$run_tag" \
      "$input_dir" \
      "$run_dir" \
      "$width_nm" \
      "$width_label" \
      "$temp" >> "$MANIFEST_CSV"
    if [[ -f "$run_dir/final_summary.txt" ]]; then
      echo "[skip] $run_tag"
    else
      echo "[run]  $run_tag"
      "$PYTHON_BIN" run_current_case.py "$input_dir" --run-tag "$run_tag"
    fi
  done
done

echo "[done] manifest: $MANIFEST_CSV"
