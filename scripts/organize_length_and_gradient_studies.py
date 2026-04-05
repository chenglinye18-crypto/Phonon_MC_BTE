from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "organized_studies"


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def replace_symlink(link_path: Path, target: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            raise RuntimeError(f"refusing to overwrite real directory: {link_path}")
        link_path.unlink()
    rel = Path(target).resolve().relative_to(ROOT.resolve()) if str(target.resolve()).startswith(str(ROOT.resolve())) else target.resolve()
    # Create repo-relative symlinks where possible to keep the tree portable inside the workspace.
    if isinstance(rel, Path) and not rel.is_absolute():
        relative_target = Path("..") / Path("..")
        for _ in link_path.relative_to(OUT_ROOT).parts[:-1]:
            relative_target /= ".."
        relative_target = Path(
            Path(
                Path(
                    Path.cwd()
                )
            )
        )
    # Simpler and robust: compute a filesystem-relative symlink target from link parent.
    relative_target = Path(
        Path(
            Path(target.resolve()).relative_to(link_path.parent.resolve())
            if False
            else target.resolve()
        )
    )
    import os

    os.symlink(os.path.relpath(target.resolve(), start=link_path.parent.resolve()), link_path)


def write_manifest(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    ensure_clean_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def link_items(base_dir: Path, specs: list[dict[str, object]], manifest_path: Path, note_field: str = "note") -> None:
    ensure_clean_dir(base_dir)
    rows: list[dict[str, object]] = []
    for spec in specs:
        name = str(spec["name"])
        target = Path(spec["target"])
        category = str(spec.get("category", "item"))
        note = str(spec.get(note_field, ""))
        link_path = base_dir / name
        if target.exists():
            replace_symlink(link_path, target)
            status = "linked"
        else:
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            status = "missing"
        rows.append(
            {
                "name": name,
                "category": category,
                "status": status,
                "target": str(target.resolve()),
                note_field: note,
            }
        )
    write_manifest(manifest_path, rows, ["name", "category", "status", "target", note_field])


def length_temperature_layout() -> None:
    study_root = OUT_ROOT / "length_temperature"
    inputs_dir = study_root / "inputs"
    results_dir = study_root / "results"
    scripts_dir = study_root / "scripts"
    ensure_clean_dir(inputs_dir)
    ensure_clean_dir(results_dir)
    ensure_clean_dir(scripts_dir)

    input_specs: list[dict[str, object]] = []
    lengths = [10, 20, 50, 70, 140, 280, 560, 1120]
    temps = [300, 323, 373]
    for length_nm in lengths:
        grid_nm = 1 if length_nm in {10, 20, 50, 70} else 10
        for temp_k in temps:
            input_specs.append(
                {
                    "name": f"L{length_nm:04d}nm_T{temp_k}K",
                    "category": "input",
                    "target": ROOT / f"input_y{length_nm}nm_{grid_nm}nm_Eeff5e-19_T{temp_k}K",
                    "note": f"length={length_nm}nm,temp={temp_k}K,grid={grid_nm}nm",
                }
            )
    link_items(inputs_dir, input_specs, study_root / "input_manifest.csv")

    result_specs: list[dict[str, object]] = []
    fallback_runs = {
        (560, 300): ROOT / "output" / "run_test_y560_10nm_300K_Eeff5e-19_Tloc_after_debug",
        (1120, 300): ROOT / "output" / "run_test_y1120_10nm_300K_Eeff5e-19_Tloc_after_debug",
    }
    for length_nm in lengths:
        grid_nm = 1 if length_nm in {10, 20, 50, 70} else 10
        for temp_k in temps:
            target = fallback_runs.get(
                (length_nm, temp_k),
                ROOT / "output" / f"run_run_input_y{length_nm}nm_{grid_nm}nm_Eeff5e-19_T{temp_k}K",
            )
            note = "fallback run dir" if (length_nm, temp_k) in fallback_runs else "primary run dir"
            result_specs.append(
                {
                    "name": f"run_L{length_nm:04d}nm_T{temp_k}K",
                    "category": "run_result",
                    "target": target,
                    "note": note,
                }
            )
    summary_dirs = [
        ("boxplot_default", ROOT / "output" / "kappa_boxplot_length_temp"),
        ("boxplot_cumulative", ROOT / "output" / "kappa_boxplot_length_temp_cumulative"),
        ("boxplot_interval_rolling", ROOT / "output" / "kappa_boxplot_length_temp_interval_rolling_tail5"),
        ("boxplot_mixed1120", ROOT / "output" / "kappa_boxplot_length_temp_mixed1120"),
    ]
    for name, target in summary_dirs:
        result_specs.append(
            {
                "name": name,
                "category": "summary_result",
                "target": target,
                "note": "derived summary directory",
            }
        )
    link_items(results_dir, result_specs, study_root / "result_manifest.csv")

    script_specs = [
        {"name": "run_sweep.sh", "category": "shell_script", "target": ROOT / "run_length_temperature_sweep.sh", "note": "batch runner"},
        {"name": "plot_boxplot.py", "category": "python_script", "target": ROOT / "scripts" / "plot_kappa_boxplot_vs_length_temp.py", "note": "length-temperature boxplot"},
        {"name": "export_heat_flux.py", "category": "python_script", "target": ROOT / "scripts" / "export_heat_flux.py", "note": "heat-flux extractor"},
        {"name": "export_temperature_field.py", "category": "python_script", "target": ROOT / "scripts" / "export_temperature_field.py", "note": "temperature-field extractor"},
    ]
    link_items(scripts_dir, script_specs, study_root / "script_manifest.csv")


def gradient_layout() -> None:
    study_root = OUT_ROOT / "temperature_gradient"
    inputs_dir = study_root / "inputs"
    results_dir = study_root / "results"
    scripts_dir = study_root / "scripts"
    ensure_clean_dir(inputs_dir)
    ensure_clean_dir(results_dir)
    ensure_clean_dir(scripts_dir)

    temps = [300, 323, 373]
    delta_labels = [("2p5", 2.5), ("5", 5.0), ("10", 10.0), ("20", 20.0), ("50", 50.0)]

    input_specs: list[dict[str, object]] = []
    for temp_k in temps:
        for suffix, delta_half in delta_labels:
            input_specs.append(
                {
                    "name": f"T{temp_k}K_pm{suffix}K",
                    "category": "input",
                    "target": ROOT / f"input_y280nm_10nm_Eeff5e-19_T{temp_k}K_pm{suffix}K",
                    "note": f"temperature={temp_k}K,delta_half={delta_half}K",
                }
            )
    link_items(inputs_dir, input_specs, study_root / "input_manifest.csv")

    run_map = {
        (300, "2p5"): ROOT / "output" / "run_y280nm_300K_pm2p5K_rerun",
        (300, "5"): ROOT / "output" / "run_run_input_y280nm_10nm_Eeff5e-19_T300K",
        (300, "10"): ROOT / "output" / "run_y280nm_300K_pm10K",
        (300, "20"): ROOT / "output" / "run_y280nm_300K_pm20K",
        (300, "50"): ROOT / "output" / "run_y280nm_300K_pm50K",
        (323, "2p5"): ROOT / "output" / "run_y280nm_323K_pm2p5K",
        (323, "5"): ROOT / "output" / "run_run_input_y280nm_10nm_Eeff5e-19_T323K",
        (323, "10"): ROOT / "output" / "run_y280nm_323K_pm10K",
        (323, "20"): ROOT / "output" / "run_y280nm_323K_pm20K",
        (323, "50"): ROOT / "output" / "run_y280nm_323K_pm50K",
        (373, "2p5"): ROOT / "output" / "run_y280nm_373K_pm2p5K",
        (373, "5"): ROOT / "output" / "run_run_input_y280nm_10nm_Eeff5e-19_T373K",
        (373, "10"): ROOT / "output" / "run_y280nm_373K_pm10K",
        (373, "20"): ROOT / "output" / "run_y280nm_373K_pm20K",
        (373, "50"): ROOT / "output" / "run_y280nm_373K_pm50K",
    }
    result_specs: list[dict[str, object]] = []
    for temp_k in temps:
        for suffix, delta_half in delta_labels:
            result_specs.append(
                {
                    "name": f"run_T{temp_k}K_pm{suffix}K",
                    "category": "run_result",
                    "target": run_map[(temp_k, suffix)],
                    "note": f"temperature={temp_k}K,delta_half={delta_half}K",
                }
            )
    summary_dirs = [
        ("kappa_vs_gradient", ROOT / "output" / "y280_center_plane_kappa_vs_gradient"),
        ("kappa_boxplot_gradient", ROOT / "output" / "y280_center_plane_kappa_boxplot_gradient_interval_tail5"),
        ("summary_300_373", ROOT / "output" / "y280nm_300_373_gradient_summary"),
        ("summary_matrix", ROOT / "output" / "y280nm_gradient_matrix_summary"),
    ]
    for name, target in summary_dirs:
        result_specs.append(
            {
                "name": name,
                "category": "summary_result",
                "target": target,
                "note": "derived summary directory",
            }
        )
    link_items(results_dir, result_specs, study_root / "result_manifest.csv")

    script_specs = [
        {"name": "run_323K_sweep.sh", "category": "shell_script", "target": ROOT / "run_y280nm_323K_gradient_sweep.sh", "note": "323K gradient runner"},
        {"name": "run_300_373_no_pm50.sh", "category": "shell_script", "target": ROOT / "run_y280nm_300_373_gradient_no_pm50.sh", "note": "300/373K runner"},
        {"name": "run_overnight.sh", "category": "shell_script", "target": ROOT / "run_y280nm_gradient_overnight.sh", "note": "overnight matrix runner"},
        {"name": "plot_kappa_vs_gradient.py", "category": "python_script", "target": ROOT / "scripts" / "plot_y280_center_plane_kappa_vs_gradient.py", "note": "gradient trend plot"},
        {"name": "plot_kappa_boxplot_vs_gradient.py", "category": "python_script", "target": ROOT / "scripts" / "plot_y280_center_plane_kappa_boxplot_vs_gradient.py", "note": "gradient boxplot"},
        {"name": "export_heat_flux.py", "category": "python_script", "target": ROOT / "scripts" / "export_heat_flux.py", "note": "heat-flux extractor"},
    ]
    link_items(scripts_dir, script_specs, study_root / "script_manifest.csv")


def main() -> None:
    ensure_clean_dir(OUT_ROOT)
    length_temperature_layout()
    gradient_layout()
    print(f"[ok] organized studies written under {OUT_ROOT}")


if __name__ == "__main__":
    main()
