#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

run_case() {
  local input_dir="$1"
  local run_tag="$2"
  echo "[run] ${input_dir} -> ${run_tag}"
  conda run -n 3dmc python run_current_case.py "$input_dir" --run-tag "$run_tag"
}

run_case "input_y280nm_10nm_Eeff5e-19_T323K_pm2p5K" "y280nm_323K_pm2p5K"
run_case "input_y280nm_10nm_Eeff5e-19_T323K_pm10K" "y280nm_323K_pm10K"
run_case "input_y280nm_10nm_Eeff5e-19_T323K_pm20K" "y280nm_323K_pm20K"
