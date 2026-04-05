from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_mfp_vs_temperature import compute_mfp_table, plot_total_mfp
from scripts.export_scattering_rate_vs_energy import (
    branch_rates,
    build_opts,
    build_spectral_grid,
    collapsed_total_rates_table,
    plot_material_total_rates_multi_temperature,
    plot_material_total_rates_single_temperature,
    resolve_material_entries,
    sanitize_name,
)
from scripts.fit_scattering_params_to_kappa import DEFAULT_FIT_PARAMS, FIT_PARAM_BOUNDS, format_float
from scripts.fit_scattering_params_to_kappa import parse_targets
from scripts.plot_kappa_vs_temperature import compute_curve, plot_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit scattering parameters using only the three target kappa points, without low-temperature "
            "penalties, then export kappa(T), MFP(T), and total scattering-rate-vs-E into one bundle."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="input_y280nm_10nm_Eeff5e-19_T323K",
        help="Baseline input directory. It will not be modified; a copied fitted input is written into the bundle.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=["300:1.76", "323:1.08", "373:0.79"],
        help="Target point in the form T:kappa. Default: 300:1.76, 323:1.08, 373:0.79",
    )
    parser.add_argument("--material", default="IGZO", help="Material name/key. Default: IGZO")
    parser.add_argument("--T-min", type=float, default=250.0, help="Minimum temperature for kappa(T)/MFP(T). Default: 250")
    parser.add_argument("--T-max", type=float, default=400.0, help="Maximum temperature for kappa(T)/MFP(T). Default: 400")
    parser.add_argument("--num-points", type=int, default=151, help="Temperature samples for kappa(T)/MFP(T). Default: 151")
    parser.add_argument("--cos-beta", type=float, default=1.0, help="Same definition as export_scattering_rate_vs_energy.py")
    parser.add_argument("--de-maxiter", type=int, default=30, help="Differential evolution max iterations. Default: 30")
    parser.add_argument("--de-popsize", type=int, default=12, help="Differential evolution population size. Default: 12")
    parser.add_argument("--restarts", type=int, default=4, help="Number of DE+LS restarts. Default: 4")
    parser.add_argument("--seed", type=int, default=0, help="Random seed. Default: 0")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI. Default: 300")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional bundle directory. Default: output/refit_three_points_<timestamp>",
    )
    return parser.parse_args()


def choose_material(input_dir: Path, requested_name: str) -> dict[str, object]:
    entries = resolve_material_entries(input_dir, [requested_name] if requested_name else [])
    if not entries:
        raise ValueError(f"no material resolved from {input_dir}")
    return entries[0]


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
    from scripts.export_scattering_rate_vs_energy import compute_thermal_conductivity

    opts_override = {name: float(value) for name, value in zip(fit_param_names, fit_values, strict=True)}
    predictions = np.empty_like(temperatures, dtype=np.float64)
    for idx, temperature in enumerate(np.asarray(temperatures, dtype=np.float64)):
        opts = dict(base_opts)
        opts["T0"] = float(temperature)
        opts.update(opts_override)
        spec = spec_cache[float(temperature)]
        rates = branch_rates(spec, opts, float(temperature), cos_beta)
        table = compute_thermal_conductivity(material_entry, spec, rates, float(temperature))
        predictions[idx] = float(table.loc[table["branch"] == "TOTAL", "thermal_conductivity_W_mK"].iloc[0])
    return predictions


def fit_target_only(
    material_entry: dict[str, object],
    base_opts: dict[str, object],
    temperatures: np.ndarray,
    target_kappa: np.ndarray,
    fit_param_names: tuple[str, ...],
    cos_beta: float,
    seed: int,
    de_maxiter: int,
    de_popsize: int,
    restarts: int,
) -> tuple[np.ndarray, np.ndarray]:
    log_bounds = np.log10(np.asarray([FIT_PARAM_BOUNDS[name] for name in fit_param_names], dtype=np.float64))
    spec_cache = build_spec_cache(material_entry, base_opts, np.asarray(temperatures, dtype=np.float64))

    def unpack(log_values: np.ndarray) -> np.ndarray:
        return np.power(10.0, np.asarray(log_values, dtype=np.float64))

    def residuals(log_values: np.ndarray) -> np.ndarray:
        prediction = predict_total_kappa(
            material_entry,
            base_opts,
            spec_cache,
            np.asarray(temperatures, dtype=np.float64),
            fit_param_names,
            unpack(log_values),
            cos_beta,
        )
        return (prediction - target_kappa) / np.maximum(target_kappa, np.finfo(np.float64).tiny)

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
    for i in range(max(1, int(restarts))):
        de_result = differential_evolution(
            lambda x: float(np.sum(residuals(x) ** 2)),
            bounds=[tuple(bound) for bound in log_bounds],
            maxiter=int(de_maxiter),
            popsize=int(de_popsize),
            polish=False,
            seed=int(seed) + 1009 * i,
            workers=1,
        )
        ls_result = least_squares(
            residuals,
            de_result.x,
            bounds=(log_bounds[:, 0], log_bounds[:, 1]),
            xtol=1e-7,
            ftol=1e-7,
            gtol=1e-7,
            max_nfev=300,
        )
        candidate_values = unpack(ls_result.x)
        candidate_prediction = predict_total_kappa(
            material_entry,
            base_opts,
            spec_cache,
            np.asarray(temperatures, dtype=np.float64),
            fit_param_names,
            candidate_values,
            cos_beta,
        )
        candidate_cost = float(np.sum(((candidate_prediction - target_kappa) / np.maximum(target_kappa, np.finfo(np.float64).tiny)) ** 2))
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_values = candidate_values
            best_prediction = candidate_prediction
    return best_values, best_prediction


def patch_solver_params(solver_params_path: Path, fit_param_names: tuple[str, ...], fit_values: np.ndarray) -> None:
    import re

    text = solver_params_path.read_text(encoding="utf-8")
    for name, value in zip(fit_param_names, fit_values, strict=True):
        pattern = rf"(?m)^({re.escape(name)}\s*=\s*)([^#\n]+)"
        replacement = rf"\g<1>{format_float(float(value))}"
        new_text, count = re.subn(pattern, replacement, text, count=1)
        if count != 1:
            raise ValueError(f"failed to update {name} in {solver_params_path}")
        text = new_text
    solver_params_path.write_text(text, encoding="utf-8")


def resolve_output_dir(output_dir: str) -> Path:
    if output_dir:
        path = Path(output_dir).expanduser().resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = (ROOT / "output" / f"refit_three_points_{stamp}").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser()
    input_dir = input_dir.resolve() if input_dir.is_absolute() else (ROOT / input_dir).resolve()
    output_dir = resolve_output_dir(args.output_dir)

    temperatures, target_kappa = parse_targets(list(args.target))
    material_entry = choose_material(input_dir, str(args.material))
    base_opts = build_opts(input_dir, float(temperatures[0]))
    fit_param_names = tuple(DEFAULT_FIT_PARAMS)
    fit_values, fitted_targets = fit_target_only(
        material_entry=material_entry,
        base_opts=base_opts,
        temperatures=temperatures,
        target_kappa=target_kappa,
        fit_param_names=fit_param_names,
        cos_beta=float(args.cos_beta),
        seed=int(args.seed),
        de_maxiter=int(args.de_maxiter),
        de_popsize=int(args.de_popsize),
        restarts=int(args.restarts),
    )

    fitted_input_dir = output_dir / "fitted_input"
    shutil.copytree(input_dir, fitted_input_dir)
    patch_solver_params(fitted_input_dir / "solver_params.toml", fit_param_names, fit_values)

    fit_summary = pd.DataFrame(
        {
            "temperature_K": temperatures,
            "target_kappa_W_mK": target_kappa,
            "fitted_kappa_W_mK": fitted_targets,
            "relative_error": (fitted_targets - target_kappa) / np.maximum(target_kappa, np.finfo(np.float64).tiny),
        }
    )
    fit_params_df = pd.DataFrame({"parameter": fit_param_names, "value": fit_values})
    fit_summary.to_csv(output_dir / "fit_summary.csv", index=False)
    fit_params_df.to_csv(output_dir / "fitted_parameters.csv", index=False)
    (output_dir / "fitted_parameters.txt").write_text(
        "\n".join(f"{name} = {format_float(float(value))}" for name, value in zip(fit_param_names, fit_values, strict=True)) + "\n",
        encoding="utf-8",
    )

    # kappa(T)
    kappa_dir = output_dir / "kappa_vs_temperature"
    kappa_dir.mkdir(exist_ok=True)
    curve_temperatures = np.linspace(float(args.T_min), float(args.T_max), int(args.num_points), dtype=np.float64)
    fitted_material_entry = choose_material(fitted_input_dir, str(args.material))
    curve_df = compute_curve(fitted_material_entry, fitted_input_dir, curve_temperatures, float(args.cos_beta), jobs=1)
    target_df = pd.DataFrame({"temperature_K": temperatures, "thermal_conductivity_W_mK": target_kappa})
    safe_name = sanitize_name(str(fitted_material_entry["name"]))
    stem = f"kappa_vs_temperature_{safe_name}_{curve_temperatures[0]:.0f}K_{curve_temperatures[-1]:.0f}K"
    kappa_csv = kappa_dir / f"{stem}.csv"
    kappa_png = kappa_dir / f"{stem}.png"
    kappa_pdf = kappa_dir / f"{stem}.pdf"
    export_df = curve_df.copy()
    target_export = target_df.copy()
    target_export.insert(0, "series", "target")
    export_df = pd.concat([export_df, target_export], ignore_index=True)
    export_df.to_csv(kappa_csv, index=False)
    plot_curve(str(fitted_material_entry["name"]), curve_df, target_df, kappa_png, kappa_pdf, int(args.dpi))

    # MFP(T)
    mfp_dir = output_dir / "mfp_vs_temperature"
    mfp_dir.mkdir(exist_ok=True)
    mfp_opts = build_opts(fitted_input_dir, float(0.5 * (curve_temperatures[0] + curve_temperatures[-1])))
    spec = build_spectral_grid(dict(fitted_material_entry["mat"]), mfp_opts)
    mfp_df = compute_mfp_table(fitted_material_entry, spec, curve_temperatures, mfp_opts)
    mfp_stem = f"mfp_vs_temperature_{safe_name}_{curve_temperatures[0]:.0f}K_{curve_temperatures[-1]:.0f}K"
    mfp_csv = mfp_dir / f"{mfp_stem}.csv"
    mfp_png = mfp_dir / f"{mfp_stem}.png"
    mfp_pdf = mfp_dir / f"{mfp_stem}.pdf"
    mfp_df.to_csv(mfp_csv, index=False)
    plot_total_mfp(mfp_df, mfp_png, mfp_pdf, int(args.dpi))

    # total scattering rate vs E
    scatter_dir = output_dir / "scattering_rate_vs_energy"
    scatter_dir.mkdir(exist_ok=True)
    scatter_temps = np.asarray([300.0, 323.0, 373.0], dtype=np.float64)
    curves: list[tuple[float, dict[str, object], dict[str, np.ndarray]]] = []
    scatter_manifest_rows: list[dict[str, object]] = []
    for temp in scatter_temps:
        opts = build_opts(fitted_input_dir, float(temp))
        spec = build_spectral_grid(dict(fitted_material_entry["mat"]), opts)
        rates = branch_rates(spec, opts, float(temp), float(args.cos_beta))
        curves.append((float(temp), spec, rates))
        total_csv = scatter_dir / f"total_scattering_rate_vs_E_{safe_name}_T{temp:.3f}K.csv"
        total_png = scatter_dir / f"total_scattering_rate_vs_E_{safe_name}_T{temp:.3f}K.png"
        collapsed_total_rates_table(fitted_material_entry, spec, rates, float(temp)).to_csv(total_csv, index=False)
        plot_material_total_rates_single_temperature(fitted_material_entry, spec, rates, float(temp), total_png, int(args.dpi))
        scatter_manifest_rows.append(
            {
                "temperature_K": float(temp),
                "csv_path": str(total_csv.resolve()),
                "png_path": str(total_png.resolve()),
            }
        )
    multi_png = scatter_dir / f"total_scattering_rate_vs_E_{safe_name}_T300.000K_323.000K_373.000K.png"
    plot_material_total_rates_multi_temperature(fitted_material_entry, curves, multi_png, int(args.dpi))
    pd.DataFrame(scatter_manifest_rows).to_csv(scatter_dir / "manifest.csv", index=False)

    manifest = pd.DataFrame(
        [
            {"kind": "input_source", "path": str(input_dir.resolve())},
            {"kind": "fitted_input_dir", "path": str(fitted_input_dir.resolve())},
            {"kind": "fit_summary_csv", "path": str((output_dir / 'fit_summary.csv').resolve())},
            {"kind": "fitted_parameters_csv", "path": str((output_dir / 'fitted_parameters.csv').resolve())},
            {"kind": "kappa_dir", "path": str(kappa_dir.resolve())},
            {"kind": "mfp_dir", "path": str(mfp_dir.resolve())},
            {"kind": "scattering_rate_dir", "path": str(scatter_dir.resolve())},
        ]
    )
    manifest.to_csv(output_dir / "bundle_manifest.csv", index=False)

    print(f"[ok] bundle -> {output_dir}")
    print("[ok] fitted parameters")
    for name, value in zip(fit_param_names, fit_values, strict=True):
        print(f"{name} = {format_float(float(value))}")
    print("[ok] fit summary")
    print(fit_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
