from __future__ import annotations

import argparse
import math
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
    "BL": (1.0e-24, 3.0e-22),
    "A_imp": (1.0e-50, 1.0e-46),
    "B_imp": (1.0e-24, 3.0e-22),
    "C_imp": (1.0e5, 1.0e9),
    "BTN": (3.0e-13, 3.0e-12),
    "BTU": (3.0e-16, 5.0e-15),
}

DEFAULT_FIT_PARAMS = ("BL", "A_imp", "B_imp", "C_imp", "BTN", "BTU")
DIAGNOSTIC_TEMPERATURES = np.asarray([10.0, 20.0, 30.0, 50.0, 80.0, 120.0, 180.0, 250.0], dtype=np.float64)
LOW_TEMP_PEAK_LIMIT_W_MK = 50.0
LOW_TEMP_SHAPE_RATIO_LIMIT = 5
PEAK_PENALTY_WEIGHT = 10.0
MONOTONICITY_PENALTY_WEIGHT = 8.0
LOW_TEMP_SHAPE_WEIGHT = 8.0


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
        default=[],
        choices=tuple(FIT_PARAM_BOUNDS),
        help=(
            "Parameter to fit. Repeat to override the defaults. "
            "Default: BL, A_imp, B_imp, C_imp, BTN, BTU"
        ),
    )
    parser.add_argument("--cos-beta", type=float, default=1.0, help="Same definition as export_scattering_rate_vs_energy.py")
    parser.add_argument("--de-maxiter", type=int, default=20, help="Differential evolution max iterations. Default: 20")
    parser.add_argument("--de-popsize", type=int, default=10, help="Differential evolution population size. Default: 10")
    parser.add_argument("--restarts", type=int, default=3, help="Number of global+local optimization restarts. Default: 3")
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


def resolve_fit_param_names(requested: list[str]) -> tuple[str, ...]:
    return tuple(requested) if requested else DEFAULT_FIT_PARAMS


def build_spec_cache(material_entry: dict[str, object], base_opts: dict[str, object], temperatures: np.ndarray) -> dict[float, dict[str, object]]:
    mat = dict(material_entry["mat"])
    cache: dict[float, dict[str, object]] = {}
    for temperature in np.asarray(temperatures, dtype=np.float64):
        opts = dict(base_opts)
        opts["T0"] = float(temperature)
        cache[float(temperature)] = build_spectral_grid(mat, opts)
    return cache


def predict_total_kappa(
    material_entry: dict[str, object],
    base_opts: dict[str, object],
    spec_cache: dict[float, dict[str, object]],
    temperatures: np.ndarray,
    fit_param_names: tuple[str, ...],
    fit_values: np.ndarray,
    cos_beta: float,
) -> np.ndarray:
    opts_override = {name: float(value) for name, value in zip(fit_param_names, fit_values, strict=True)}
    predictions = np.empty_like(temperatures, dtype=np.float64)
    for idx, temperature in enumerate(temperatures):
        opts = dict(base_opts)
        opts["T0"] = float(temperature)
        opts.update(opts_override)
        spec = spec_cache[float(temperature)]
        rates = branch_rates(spec, opts, float(temperature), cos_beta)
        table = compute_thermal_conductivity(material_entry, spec, rates, float(temperature))
        predictions[idx] = float(table.loc[table["branch"] == "TOTAL", "thermal_conductivity_W_mK"].iloc[0])
    return predictions


def build_penalized_residuals(
    target_kappa: np.ndarray,
    target_prediction: np.ndarray,
    diagnostic_temperatures: np.ndarray,
    diagnostic_prediction: np.ndarray,
    penalty_temperatures: np.ndarray,
    penalty_prediction: np.ndarray,
) -> np.ndarray:
    tiny = np.finfo(np.float64).tiny
    residuals: list[float] = list(((target_prediction - target_kappa) / np.maximum(target_kappa, tiny)).astype(np.float64))

    low80_mask = diagnostic_temperatures <= 80.0
    if np.any(low80_mask):
        peak_low80 = float(np.max(diagnostic_prediction[low80_mask]))
        excess_peak = max(0.0, peak_low80 - LOW_TEMP_PEAK_LIMIT_W_MK) / LOW_TEMP_PEAK_LIMIT_W_MK
        residuals.append(math.sqrt(PEAK_PENALTY_WEIGHT) * excess_peak)

    high_mask = penalty_temperatures >= 120.0
    if np.count_nonzero(high_mask) >= 2:
        k_high = np.asarray(penalty_prediction[high_mask], dtype=np.float64)
        upward = np.maximum(np.diff(k_high), 0.0) / np.maximum(k_high[:-1], tiny)
        residuals.extend((math.sqrt(MONOTONICITY_PENALTY_WEIGHT) * upward).tolist())

    low50_mask = diagnostic_temperatures <= 50.0
    if np.any(low50_mask):
        peak_low50 = float(np.max(diagnostic_prediction[low50_mask]))
        idx_50 = int(np.argmin(np.abs(diagnostic_temperatures - 50.0)))
        k_50 = float(diagnostic_prediction[idx_50])
        ratio = peak_low50 / max(k_50, tiny)
        excess_ratio = max(0.0, ratio - LOW_TEMP_SHAPE_RATIO_LIMIT) / LOW_TEMP_SHAPE_RATIO_LIMIT
        residuals.append(math.sqrt(LOW_TEMP_SHAPE_WEIGHT) * excess_ratio)

    return np.asarray(residuals, dtype=np.float64)


def fit_parameters(
    material_entry: dict[str, object],
    base_opts: dict[str, object],
    temperatures: np.ndarray,
    target_kappa: np.ndarray,
    diagnostic_temperatures: np.ndarray,
    fit_param_names: tuple[str, ...],
    cos_beta: float,
    seed: int,
    de_maxiter: int,
    de_popsize: int,
    restarts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_bounds = np.log10(np.asarray([FIT_PARAM_BOUNDS[name] for name in fit_param_names], dtype=np.float64))
    penalty_temperatures = np.unique(np.concatenate((np.asarray(temperatures, dtype=np.float64), np.asarray(diagnostic_temperatures, dtype=np.float64))))
    spec_cache = build_spec_cache(material_entry, base_opts, penalty_temperatures)
    target_idx = np.searchsorted(penalty_temperatures, np.asarray(temperatures, dtype=np.float64))
    diagnostic_idx = np.searchsorted(penalty_temperatures, np.asarray(diagnostic_temperatures, dtype=np.float64))

    def unpack(log_values: np.ndarray) -> np.ndarray:
        return np.power(10.0, np.asarray(log_values, dtype=np.float64))

    def residual_vector(log_values: np.ndarray) -> np.ndarray:
        penalty_prediction = predict_total_kappa(
            material_entry,
            base_opts,
            spec_cache,
            penalty_temperatures,
            fit_param_names,
            unpack(log_values),
            cos_beta,
        )
        return build_penalized_residuals(
            target_kappa=np.asarray(target_kappa, dtype=np.float64),
            target_prediction=penalty_prediction[target_idx],
            diagnostic_temperatures=np.asarray(diagnostic_temperatures, dtype=np.float64),
            diagnostic_prediction=penalty_prediction[diagnostic_idx],
            penalty_temperatures=penalty_temperatures,
            penalty_prediction=penalty_prediction,
        )

    best_cost = np.inf
    best_values = np.power(10.0, np.mean(log_bounds, axis=1))
    best_prediction = predict_total_kappa(
        material_entry,
        base_opts,
        spec_cache,
        np.asarray(temperatures, dtype=np.float64),
        fit_param_names,
        best_values,
        cos_beta,
    )
    best_diag_prediction = predict_total_kappa(
        material_entry,
        base_opts,
        spec_cache,
        np.asarray(diagnostic_temperatures, dtype=np.float64),
        fit_param_names,
        best_values,
        cos_beta,
    )
    for i in range(max(1, int(restarts))):
        de_result = differential_evolution(
            lambda x: float(np.sum(residual_vector(x) ** 2)),
            bounds=[tuple(bound) for bound in log_bounds],
            maxiter=int(de_maxiter),
            popsize=int(de_popsize),
            polish=False,
            seed=int(seed) + 1009 * i,
            workers=1,
        )
        ls_result = least_squares(
            residual_vector,
            de_result.x,
            bounds=(log_bounds[:, 0], log_bounds[:, 1]),
            xtol=1e-6,
            ftol=1e-6,
            gtol=1e-6,
            max_nfev=160,
        )
        candidate_values = unpack(ls_result.x)
        candidate_penalty_prediction = predict_total_kappa(
            material_entry,
            base_opts,
            spec_cache,
            penalty_temperatures,
            fit_param_names,
            candidate_values,
            cos_beta,
        )
        candidate_prediction = candidate_penalty_prediction[target_idx]
        candidate_diag_prediction = candidate_penalty_prediction[diagnostic_idx]
        candidate_cost = float(
            np.sum(
                build_penalized_residuals(
                    target_kappa=np.asarray(target_kappa, dtype=np.float64),
                    target_prediction=candidate_prediction,
                    diagnostic_temperatures=np.asarray(diagnostic_temperatures, dtype=np.float64),
                    diagnostic_prediction=candidate_diag_prediction,
                    penalty_temperatures=penalty_temperatures,
                    penalty_prediction=candidate_penalty_prediction,
                )
                ** 2
            )
        )
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_values = candidate_values
            best_prediction = candidate_prediction
            best_diag_prediction = candidate_diag_prediction
    return best_values, best_prediction, best_diag_prediction


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
    material_entry = choose_material(input_dir, str(args.material))
    base_opts = build_opts(input_dir, float(temperatures[0]))
    fit_param_names = resolve_fit_param_names(list(args.fit_param))

    best_values, best_prediction, diagnostic_prediction = fit_parameters(
        material_entry=material_entry,
        base_opts=base_opts,
        temperatures=temperatures,
        target_kappa=target_kappa,
        diagnostic_temperatures=DIAGNOSTIC_TEMPERATURES,
        fit_param_names=fit_param_names,
        cos_beta=float(args.cos_beta),
        seed=int(args.seed),
        de_maxiter=int(args.de_maxiter),
        de_popsize=int(args.de_popsize),
        restarts=int(args.restarts),
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
    diagnostic_df = pd.DataFrame(
        {
            "temperature_K": DIAGNOSTIC_TEMPERATURES,
            "predicted_kappa_W_mK": diagnostic_prediction,
        }
    )
    print("[fit] diagnostic temperatures")
    print(diagnostic_df.to_string(index=False))

    if args.write:
        write_solver_params(solver_params_path, fit_param_names, best_values)
        print(f"[fit] updated {solver_params_path}")

    print(f"[fit] summary csv: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
