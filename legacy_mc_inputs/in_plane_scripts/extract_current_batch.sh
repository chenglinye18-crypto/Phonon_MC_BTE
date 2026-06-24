#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/home/ic/miniconda3/envs/3dmc/bin/python}"

MANIFEST_CSV="$ROOT_DIR/output/in_plane_current_batch/case_manifest.csv"
SUMMARY_DIR="$ROOT_DIR/output/in_plane_current_results"
BOXPLOT_DIR="$ROOT_DIR/output/in_plane_current_results_boxplot"

mkdir -p "$SUMMARY_DIR" "$BOXPLOT_DIR"

"$PYTHON_BIN" scripts/summarize_in_plane_scatter_runs.py \
  --manifest-csv "$MANIFEST_CSV" \
  --step 32000 40000 \
  --output-dir "$SUMMARY_DIR"

"$PYTHON_BIN" scripts/plot_in_plane_scatter_kappa_boxplot_vs_width.py \
  --manifest-csv "${MANIFEST_CSV#$ROOT_DIR/}" \
  --step-min 32000 \
  --tail-count 5 \
  --window-max 5 \
  --monitor-label flux_plane_004 \
  --output-dir "${BOXPLOT_DIR#$ROOT_DIR/}"

echo "[done] avg summary: $SUMMARY_DIR/heat_flux_kappa_summary.csv"
echo "[done] boxplot raw: $BOXPLOT_DIR/in_plane_scatter_kappa_boxplot_interval_tail5.csv"
echo "[done] boxplot summary: $BOXPLOT_DIR/in_plane_scatter_kappa_boxplot_interval_tail5_summary.csv"
