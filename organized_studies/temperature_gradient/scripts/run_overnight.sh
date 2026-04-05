#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PY_BIN="/home/ic/miniconda3/envs/3dmc/bin/python"
STEP_START=32000
STEP_END=40000
L_M=2.8e-7
SUMMARY_DIR="output/y280nm_gradient_matrix_summary"
PER_CASE_DIR="${SUMMARY_DIR}/per_case"
mkdir -p "$PER_CASE_DIR"

run_if_needed() {
  local input_dir="$1"
  local run_tag="$2"
  local output_dir="$3"
  if [[ -f "${output_dir}/steps/step_40000/heat_flux.txt" ]]; then
    echo "[skip] ${output_dir} already complete"
    return 0
  fi
  if [[ -z "$run_tag" ]]; then
    echo "[error] missing run_tag for unfinished case ${input_dir}" >&2
    return 1
  fi
  echo "[run] ${input_dir} -> ${output_dir}"
  "$PY_BIN" run_current_case.py "$input_dir" --run-tag "$run_tag"
}

export_case_csv() {
  local case_name="$1"
  local output_dir="$2"
  local out_csv="${PER_CASE_DIR}/${case_name}__heat_flux_interval_steps_${STEP_START}_${STEP_END}_avg.csv"
  echo "[post] ${case_name}"
  "$PY_BIN" scripts/export_heat_flux.py \
    --run-dir "$output_dir" \
    --step "$STEP_START" "$STEP_END" \
    --stat-type interval \
    --output-csv "$out_csv"
}

# temperature_K|delta_half_K|case_name|input_dir|run_tag|output_dir
CASES=(
  "300|2.5|T300K_pm2p5K|input_y280nm_10nm_Eeff5e-19_T300K_pm2p5K|y280nm_300K_pm2p5K|output/run_y280nm_300K_pm2p5K"
  "300|5.0|T300K_pm5K|input_y280nm_10nm_Eeff5e-19_T300K_pm5K||output/run_run_input_y280nm_10nm_Eeff5e-19_T300K"
  "300|10.0|T300K_pm10K|input_y280nm_10nm_Eeff5e-19_T300K_pm10K|y280nm_300K_pm10K|output/run_y280nm_300K_pm10K"
  "300|20.0|T300K_pm20K|input_y280nm_10nm_Eeff5e-19_T300K_pm20K|y280nm_300K_pm20K|output/run_y280nm_300K_pm20K"
  "300|50.0|T300K_pm50K|input_y280nm_10nm_Eeff5e-19_T300K_pm50K|y280nm_300K_pm50K|output/run_y280nm_300K_pm50K"
  "323|2.5|T323K_pm2p5K|input_y280nm_10nm_Eeff5e-19_T323K_pm2p5K||output/run_y280nm_323K_pm2p5K"
  "323|5.0|T323K_pm5K|input_y280nm_10nm_Eeff5e-19_T323K_pm5K||output/run_run_input_y280nm_10nm_Eeff5e-19_T323K"
  "323|10.0|T323K_pm10K|input_y280nm_10nm_Eeff5e-19_T323K_pm10K||output/run_y280nm_323K_pm10K"
  "323|20.0|T323K_pm20K|input_y280nm_10nm_Eeff5e-19_T323K_pm20K||output/run_y280nm_323K_pm20K"
  "323|50.0|T323K_pm50K|input_y280nm_10nm_Eeff5e-19_T323K_pm50K|y280nm_323K_pm50K|output/run_y280nm_323K_pm50K"
  "373|2.5|T373K_pm2p5K|input_y280nm_10nm_Eeff5e-19_T373K_pm2p5K|y280nm_373K_pm2p5K|output/run_y280nm_373K_pm2p5K"
  "373|5.0|T373K_pm5K|input_y280nm_10nm_Eeff5e-19_T373K_pm5K||output/run_run_input_y280nm_10nm_Eeff5e-19_T373K"
  "373|10.0|T373K_pm10K|input_y280nm_10nm_Eeff5e-19_T373K_pm10K|y280nm_373K_pm10K|output/run_y280nm_373K_pm10K"
  "373|20.0|T373K_pm20K|input_y280nm_10nm_Eeff5e-19_T373K_pm20K|y280nm_373K_pm20K|output/run_y280nm_373K_pm20K"
  "373|50.0|T373K_pm50K|input_y280nm_10nm_Eeff5e-19_T373K_pm50K|y280nm_373K_pm50K|output/run_y280nm_373K_pm50K"
)

for entry in "${CASES[@]}"; do
  IFS="|" read -r temperature_K delta_half_K case_name input_dir run_tag output_dir <<<"$entry"
  run_if_needed "$input_dir" "$run_tag" "$output_dir"
done

for entry in "${CASES[@]}"; do
  IFS="|" read -r temperature_K delta_half_K case_name input_dir run_tag output_dir <<<"$entry"
  export_case_csv "$case_name" "$output_dir"
done

"$PY_BIN" - <<'PY'
from pathlib import Path
import csv
import pandas as pd

root = Path("/home/ic/3dmc_Si_ylx_mod/Phonon_MC_py")
summary_dir = root / "output" / "y280nm_gradient_matrix_summary"
per_case_dir = summary_dir / "per_case"
step_start = 32000
step_end = 40000
L_m = 2.8e-7
cases = [
    (300.0, 2.5, "T300K_pm2p5K", root / "output" / "run_y280nm_300K_pm2p5K"),
    (300.0, 5.0, "T300K_pm5K", root / "output" / "run_run_input_y280nm_10nm_Eeff5e-19_T300K"),
    (300.0, 10.0, "T300K_pm10K", root / "output" / "run_y280nm_300K_pm10K"),
    (300.0, 20.0, "T300K_pm20K", root / "output" / "run_y280nm_300K_pm20K"),
    (300.0, 50.0, "T300K_pm50K", root / "output" / "run_y280nm_300K_pm50K"),
    (323.0, 2.5, "T323K_pm2p5K", root / "output" / "run_y280nm_323K_pm2p5K"),
    (323.0, 5.0, "T323K_pm5K", root / "output" / "run_run_input_y280nm_10nm_Eeff5e-19_T323K"),
    (323.0, 10.0, "T323K_pm10K", root / "output" / "run_y280nm_323K_pm10K"),
    (323.0, 20.0, "T323K_pm20K", root / "output" / "run_y280nm_323K_pm20K"),
    (323.0, 50.0, "T323K_pm50K", root / "output" / "run_y280nm_323K_pm50K"),
    (373.0, 2.5, "T373K_pm2p5K", root / "output" / "run_y280nm_373K_pm2p5K"),
    (373.0, 5.0, "T373K_pm5K", root / "output" / "run_run_input_y280nm_10nm_Eeff5e-19_T373K"),
    (373.0, 10.0, "T373K_pm10K", root / "output" / "run_y280nm_373K_pm10K"),
    (373.0, 20.0, "T373K_pm20K", root / "output" / "run_y280nm_373K_pm20K"),
    (373.0, 50.0, "T373K_pm50K", root / "output" / "run_y280nm_373K_pm50K"),
]

manifest_rows = []
summary_rows = []
for temperature_K, delta_half_K, case_name, run_dir in cases:
    csv_path = per_case_dir / f"{case_name}__heat_flux_interval_steps_{step_start}_{step_end}_avg.csv"
    tbl = pd.read_csv(csv_path)
    delta_total_K = 2.0 * float(delta_half_K)
    gradient = delta_total_K / L_m
    manifest_rows.append(
        {
            "case_name": case_name,
            "temperature_K": float(temperature_K),
            "delta_half_K": float(delta_half_K),
            "delta_total_K": delta_total_K,
            "L_m": L_m,
            "gradient_K_per_m": gradient,
            "run_dir": str(run_dir),
            "heat_flux_csv": str(csv_path),
        }
    )
    for row in tbl.to_dict(orient="records"):
        heat_flux = float(row["heat_flux_W_m2"])
        summary_rows.append(
            {
                "case_name": case_name,
                "temperature_K": float(temperature_K),
                "delta_half_K": float(delta_half_K),
                "delta_total_K": delta_total_K,
                "L_m": L_m,
                "gradient_K_per_m": gradient,
                "label": str(row["label"]),
                "heat_flux_W_m2": heat_flux,
                "kappa_div_grad_W_mK": heat_flux / gradient,
                "kappa_fourier_negq_over_grad_W_mK": -heat_flux / gradient,
                "step_start": step_start,
                "step_end": step_end,
                "step_count": int(row.get("step_count", 0)) if pd.notna(row.get("step_count", 0)) else 0,
                "steps_used": str(row.get("steps_used", "")),
                "run_dir": str(run_dir),
            }
        )

manifest_path = summary_dir / "case_manifest.csv"
summary_path = summary_dir / "heat_flux_kappa_summary.csv"
pd.DataFrame(manifest_rows).sort_values(["temperature_K", "delta_half_K"]).to_csv(manifest_path, index=False)
pd.DataFrame(summary_rows).sort_values(["temperature_K", "delta_half_K", "label"]).to_csv(summary_path, index=False)
print(manifest_path)
print(summary_path)
PY

echo "[done] summary directory: ${SUMMARY_DIR}"
