from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_scattering_rate_vs_energy import (
    branch_rates,
    build_opts,
    build_spectral_grid,
    compute_thermal_conductivity,
    resolve_material_entries,
)


FIT_PARAM_BOUNDS = {
    "BL": (1.0e-28, 1.0e-20),
    "A_imp": (1.0e-47, 1.0e-35),
    "BTN": (1.0e-15, 1.0e-8),
    "BTU": (1.0e-19, 1.0e-12),
    "PB_Tsi": (1.0e-9, 1.0e-4),
    "PB_Delta": (1.0e-12, 1.0e-7),
}

DEFAULT_FIT_PARAMS = ("BL", "A_imp", "BTN", "BTU", "PB_Tsi", "PB_Delta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit scattering parameters against literature thermal conductivity targets "
            "without changing the main solver logic."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="input",
        help="Input directory containing solver_params.toml, ldg.txt and lgrid.txt. Default: input",
    )
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        help="Target point in the form T:kappa, for example 300:2.58 . Repeat this flag for multiple temperatures.",
    )
    parser.add_argument(
        "--material",
        default="",
        help="Optional material name/key. Default: first material used by the current case.",
    )
    parser.add_argument(
        "--fit-param",
        action="append",
        choices=tuple(FIT_PARAM_BOUNDS),
        help=(
            "Parameter to fit. Repeat to override the defaults. "
            "Default: BL, A_imp, BTN, BTU, PB_Tsi, PB_Delta"
        ),
    )
    parser.add_argument("--cos-beta", type=float, default=1.0, help="Same definition as export_scattering_rate_vs_energy.py")
    parser.add_argument("--de-maxiter", type=int, default=20, help="Differential evolution max iterations. Default: 20")
    parser.add_argument("--de-popsize", type=int, default=10, help="Differential evolution population size. Default: 10")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible fitting. Default: 0")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write fitted scattering parameters back into input-dir/solver_params.toml",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional CSV path for fitted-vs-target summary. Default: output/scattering_param_fit/fit_summary.csv",
    )
    return parser.parse_args()


def parse_targets(raw_targets: list[str]) -> tuple[np.ndarray, np.ndarray]:
    temps: list[float] = []
    kappas: list[float] = []
    for item in raw_targets:
        if ":" not in item:
            raise ValueError(f"invalid target format: {item!r}, expected T:kappa")
        left, right = item.split(":", 1)
        temps.append(float(left))
        kappas.append(float(right))
    order = np.argsort(np.asarray(temps, dtype=np.float64))
    return np.asarray(temps, dtype=np.float64)[order], np.asarray(kappas, dtype=np.float64)[order]


def format_float(value: float) -> str:
    return f"{float(value):.12e}"


def choose_material(input_dir: Path, requested_name: str) -> dict[str, object]:
    entries = resolve_material_entries(input_dir, [requested_name] if requested_name else [])
    if not entries:
        raise ValueError(f"no material resolved from {input_dir}")
    return entries[0]


def predict_total_kappa(
    material_entry: dict[str, object],
    base_opts: dict[str, object],
    temperatures: np.ndarray,
    fit_param_names: tuple[str, ...],
    fit_values: np.ndarray,
    cos_beta: float,
) -> np.ndarray:
    mat = dict(material_entry["mat"])
    opts_override = {name: float(value) for name, value in zip(fit_param_names, fit_values, strict=True)}
    predictions = np.empty_like(temperatures, dtype=np.float64)
    for idx, temperature in enumerate(temperatures):
        opts = dict(base_opts)
        opts["T0"] = float(temperature)
        opts.update(opts_override)
        spec = build_spectral_grid(mat, opts)
        rates = branch_rates(spec, opts, float(temperature), cos_beta)
        table = compute_thermal_conductivity(material_entry, spec, rates, float(temperature))
        predictions[idx] = float(table.loc[table["branch"] == "TOTAL", "thermal_conductivity_W_mK"].iloc[0])
    return predictions


def fit_parameters(
    material_entry: dict[str, object],
    base_opts: dict[str, object],
    temperatures: np.ndarray,
    target_kappa: np.ndarray,
    fit_param_names: tuple[str, ...],
    cos_beta: float,
    seed: int,
    de_maxiter: int,
    de_popsize: int,
) -> tuple[np.ndarray, np.ndarray]:
    log_bounds = np.log10(np.asarray([FIT_PARAM_BOUNDS[name] for name in fit_param_names], dtype=np.float64))

    def unpack(log_values: np.ndarray) -> np.ndarray:
        return np.power(10.0, np.asarray(log_values, dtype=np.float64))

    def residual_vector(log_values: np.ndarray) -> np.ndarray:
        prediction = predict_total_kappa(
            material_entry,
            base_opts,
            temperatures,
            fit_param_names,
            unpack(log_values),
            cos_beta,
        )
        return (prediction - target_kappa) / np.maximum(target_kappa, np.finfo(np.float64).tiny)

    de_result = differential_evolution(
        lambda x: float(np.sum(residual_vector(x) ** 2)),
        bounds=[tuple(bound) for bound in log_bounds],
        maxiter=int(de_maxiter),
        popsize=int(de_popsize),
        polish=False,
        seed=int(seed),
        workers=1,
    )
    ls_result = least_squares(
        residual_vector,
        de_result.x,
        bounds=(log_bounds[:, 0], log_bounds[:, 1]),
        xtol=1e-6,
        ftol=1e-6,
        gtol=1e-6,
        max_nfev=120,
    )
    best_values = unpack(ls_result.x)
    best_prediction = predict_total_kappa(
        material_entry,
        base_opts,
        temperatures,
        fit_param_names,
        best_values,
        cos_beta,
    )
    return best_values, best_prediction


def write_solver_params(solver_params_path: Path, fit_param_names: tuple[str, ...], fit_values: np.ndarray) -> None:
    text = solver_params_path.read_text(encoding="utf-8")
    for name, value in zip(fit_param_names, fit_values, strict=True):
        pattern = rf"(?m)^({re.escape(name)}\s*=\s*)([^#\n]+)"
        replacement = rf"\g<1>{format_float(float(value))}"
        new_text, count = re.subn(pattern, replacement, text, count=1)
        if count != 1:
            raise ValueError(f"failed to update {name} in {solver_params_path}")
        text = new_text
    solver_params_path.write_text(text, encoding="utf-8")


def default_output_csv(input_dir: Path) -> Path:
    path = ROOT / "output" / "scattering_param_fit"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"fit_summary__{input_dir.name}.csv"


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    solver_params_path = input_dir / "solver_params.toml"
    if not solver_params_path.is_file():
        raise FileNotFoundError(f"missing solver_params.toml under {input_dir}")
    temperatures, target_kappa = parse_targets(list(args.target))
    fit_param_names = tuple(args.fit_param) if args.fit_param else DEFAULT_FIT_PARAMS
    material_entry = choose_material(input_dir, str(args.material))
    base_opts = build_opts(input_dir, float(temperatures[0]))

    best_values, best_prediction = fit_parameters(
        material_entry=material_entry,
        base_opts=base_opts,
        temperatures=temperatures,
        target_kappa=target_kappa,
        fit_param_names=fit_param_names,
        cos_beta=float(args.cos_beta),
        seed=int(args.seed),
        de_maxiter=int(args.de_maxiter),
        de_popsize=int(args.de_popsize),
    )

    summary = pd.DataFrame(
        {
            "temperature_K": temperatures,
            "target_kappa_W_mK": target_kappa,
            "fitted_kappa_W_mK": best_prediction,
            "relative_error": (best_prediction - target_kappa) / np.maximum(target_kappa, np.finfo(np.float64).tiny),
        }
    )
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else default_output_csv(input_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)

    print("[fit] targets vs fitted")
    print(summary.to_string(index=False))
    print("[fit] fitted scattering parameters")
    for name, value in zip(fit_param_names, best_values, strict=True):
        print(f"{name} = {format_float(float(value))}")

    if args.write:
        write_solver_params(solver_params_path, fit_param_names, best_values)
        print(f"[fit] updated {solver_params_path}")

    print(f"[fit] summary csv: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
