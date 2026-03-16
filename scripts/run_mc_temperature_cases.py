from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phonon_mc import MC_solve_BTE, mc_default_opts, resolve_case_material, setup_case_from_ldg_lgrid, write_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and run multiple MC cases with different reference temperatures, "
            "linear initial temperature fields, and case-tagged outputs."
        )
    )
    parser.add_argument("--source-input", default="input", help="Source input directory. Default: input")
    parser.add_argument("--temperatures", nargs="+", type=float, required=True, help="Reference temperatures in K, e.g. 300 323 373")
    parser.add_argument("--delta-T", type=float, default=10.0, help="Total temperature drop/rise across the domain. Default: 10 K")
    parser.add_argument("--et-span", type=float, default=10.0, help="Half-range for ET lookup table around Tref. Default: +/-10 K")
    parser.add_argument(
        "--prepared-root",
        default="output/prepared_inputs",
        help="Root directory for generated per-case input folders. Default: output/prepared_inputs",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional batch manifest CSV path. Default: output/mc_temperature_cases_<timestamp>.csv",
    )
    return parser.parse_args()


def write_temperature_csv(path: Path, nx: int, ny: int, nz: int, values_y: np.ndarray, header_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idxcell", "idycell", "idzcell", header_name])
        for idx in range(1, nx + 1):
            for idy in range(1, ny + 1):
                for idz in range(1, nz + 1):
                    writer.writerow([idx, idy, idz, f"{float(values_y[idy - 1]):.12f}"])


def replace_or_fail(text: str, key: str, value: float) -> str:
    pattern = rf"(?m)^({re.escape(key)}\s*=\s*)([^#\n]+)"
    new_text, count = re.subn(pattern, rf"\g<1>{float(value):.12f}", text, count=1)
    if count != 1:
        raise ValueError(f"failed to update {key}")
    return new_text


def prepare_case_input(source_input: Path, prepared_root: Path, tref: float, delta_T: float, et_span: float) -> Path:
    case_name = f"T{tref:.0f}K_ref{tref:.0f}K_pm{delta_T/2.0:.0f}K_dT{delta_T:.0f}K"
    case_dir = prepared_root / case_name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(source_input, case_dir)

    cs = setup_case_from_ldg_lgrid(case_dir / "ldg.txt", case_dir / "lgrid.txt", length_scale=1e-6, input_length_unit="um", verbose=False)
    mesh = dict(cs["mesh"])
    nx = int(mesh["Nx"])
    ny = int(mesh["Ny"])
    nz = int(mesh["Nz"])

    half_delta = 0.5 * float(delta_T)
    initial_y = np.linspace(float(tref) - half_delta, float(tref) + half_delta, ny, dtype=np.float64)
    reference_y = np.full(ny, float(tref), dtype=np.float64)

    write_temperature_csv(case_dir / "initial_temperature.csv", nx, ny, nz, initial_y, "Temperature")
    write_temperature_csv(case_dir / "reference_temperature.txt", nx, ny, nz, reference_y, "Tref")

    solver_params_path = case_dir / "solver_params.toml"
    solver_text = solver_params_path.read_text(encoding="utf-8")
    solver_text = replace_or_fail(solver_text, "ET_table_T_min", float(tref) - float(et_span))
    solver_text = replace_or_fail(solver_text, "ET_table_T_max", float(tref) + float(et_span))
    solver_params_path.write_text(solver_text, encoding="utf-8")
    return case_dir


def default_manifest_path() -> Path:
    return ROOT / "output" / f"mc_temperature_cases_{time.strftime('%Y%m%d_%H%M%S')}.csv"


def main() -> int:
    args = parse_args()
    source_input = Path(args.source_input).expanduser().resolve()
    if not source_input.is_dir():
        raise FileNotFoundError(f"source input directory not found: {source_input}")

    prepared_root = Path(args.prepared_root).expanduser().resolve()
    batch_tag = time.strftime("%Y%m%d_%H%M%S")
    prepared_root = prepared_root / batch_tag
    prepared_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else default_manifest_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for tref in [float(v) for v in args.temperatures]:
        case_dir = prepare_case_input(source_input, prepared_root, tref, float(args.delta_T), float(args.et_span))
        run_tag = f"{batch_tag}_T{tref:.0f}K_ref{tref:.0f}K_pm{args.delta_T/2.0:.0f}K"
        print(f"[case] prepared input -> {case_dir}")
        print(f"[case] start run_tag={run_tag}")
        cs = setup_case_from_ldg_lgrid(case_dir / "ldg.txt", case_dir / "lgrid.txt", length_scale=1e-6, input_length_unit="um", verbose=True)
        mat = resolve_case_material(cs)
        opts = mc_default_opts(case_dir)
        opts["viz"]["enable"] = False
        opts["log"]["on"] = False
        opts["log"]["to_file"] = True
        opts["log"]["filename"] = "mc_log.txt"
        opts["log"]["print_every"] = int(opts.get("output", {}).get("every_n_steps", 2000))
        opts["output"]["run_tag"] = run_tag
        final_tp, particles, out = MC_solve_BTE(cs, mat, opts)
        out = dict(out)
        if out.get("output_dir"):
            write_csv_rows(
                Path(out["output_dir"]) / "final_summary.txt",
                [
                    ["steps", out["nsteps"]],
                    ["converged", int(bool(out["converged"]))],
                    ["reservoir_refresh_steps", str(out["reservoir_refresh_steps"])],
                    ["Np", len(particles)],
                    ["Tmin_K", float(np.min(final_tp))],
                    ["Tmean_K", float(np.mean(final_tp))],
                    ["Tmax_K", float(np.max(final_tp))],
                ],
            )
        print(
            f"FINAL_OK steps={out['nsteps']} converged={int(bool(out['converged']))} "
            f"refreshes={out['reservoir_refresh_steps']} Np={len(particles)} "
            f"Tmin={float(np.min(final_tp)):.6f} Tmean={float(np.mean(final_tp)):.6f} "
            f"Tmax={float(np.max(final_tp)):.6f} output={out.get('output_dir', '')}"
        )
        rows.append(
            {
                "temperature_K": tref,
                "delta_T_K": float(args.delta_T),
                "ET_table_T_min_K": float(tref) - float(args.et_span),
                "ET_table_T_max_K": float(tref) + float(args.et_span),
                "prepared_input_dir": str(case_dir),
                "run_tag": run_tag,
                "output_dir": str(out.get("output_dir", "")),
                "steps": int(out.get("nsteps", 0)),
                "converged": int(bool(out.get("converged", False))),
                "reservoir_refresh_steps": str(out.get("reservoir_refresh_steps", "")),
                "final_T_min_K": float(np.min(final_tp)) if final_tp.size else np.nan,
                "final_T_mean_K": float(np.mean(final_tp)) if final_tp.size else np.nan,
                "final_T_max_K": float(np.max(final_tp)) if final_tp.size else np.nan,
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
