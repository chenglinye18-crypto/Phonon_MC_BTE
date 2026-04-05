from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


TEMPS_K = (300.0, 323.0, 373.0)
DELTA_HALF_K = 5.0
TABLE_MARGIN_K = 6.0


CASE_SPECS: tuple[dict[str, object], ...] = (
    {"width_um": 0.002, "width_nm": 2.0, "dir_token": "x002nm", "label": "2 nm"},
    {"width_um": 0.005, "width_nm": 5.0, "dir_token": "x005nm", "label": "5 nm"},
    {"width_um": 0.010, "width_nm": 10.0, "dir_token": "x010nm", "label": "10 nm"},
    {"width_um": 0.020, "width_nm": 20.0, "dir_token": "x020nm", "label": "20 nm"},
    {"width_um": 0.050, "width_nm": 50.0, "dir_token": "x0p05um", "label": "0.05 um"},
    {"width_um": 0.070, "width_nm": 70.0, "dir_token": "x0p07um", "label": "0.07 um"},
    {"width_um": 0.100, "width_nm": 100.0, "dir_token": "x0p10um", "label": "0.10 um"},
    {"width_um": 0.200, "width_nm": 200.0, "dir_token": "x0p20um", "label": "0.20 um"},
    {"width_um": 0.400, "width_nm": 400.0, "dir_token": "x0p40um", "label": "0.40 um"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare in-plane SCATTER test input directories.")
    parser.add_argument(
        "--template-dir",
        default="input_in_plane_x5nm_10nm_Eeff5e-19_T300K",
        help="Template input directory with SCATTER rules already defined.",
    )
    parser.add_argument(
        "--output-root",
        default=".",
        help="Directory under which generated input folders are created. Default: repo root.",
    )
    parser.add_argument(
        "--width-token",
        nargs="*",
        default=[],
        help="Optional subset of width tokens, e.g. x002nm x0p05um x0p20um. Default: all predefined widths.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        nargs="*",
        default=[],
        help="Optional subset of temperatures in K. Default: 300 323 373.",
    )
    parser.add_argument(
        "--scatter-probabilities",
        type=float,
        nargs=3,
        metavar=("P_DIFFUSE", "P_SPECULAR", "P_PASS"),
        default=(0.5, 0.5, 0.0),
        help="SCATTER probabilities written into ldg.txt. Default: 0.5 0.5 0.0",
    )
    parser.add_argument(
        "--fixed-initial-particles",
        type=int,
        default=0,
        help="If > 0, write initial_particles_fixed into solver_params.toml. Default: 0",
    )
    parser.add_argument(
        "--case-suffix",
        default="",
        help="Optional suffix inserted into generated case names before the temperature tag.",
    )
    return parser.parse_args()


def case_name(case_spec: dict[str, object], temp_k: float, case_suffix: str) -> str:
    suffix = f"_{case_suffix}" if case_suffix else ""
    return f"input_in_plane_{case_spec['dir_token']}_10nm{suffix}_T{int(round(temp_k))}K"


def replace_lx_in_ldg(text: str, width_um: float) -> str:
    out_lines: list[str] = []
    replaced = False
    for raw in text.splitlines():
        if raw.strip().startswith("$Lx$"):
            out_lines.append(f"$Lx$ {width_um:.6f}".rstrip("0").rstrip("."))
            replaced = True
        else:
            out_lines.append(raw)
    if not replaced:
        raise RuntimeError("failed to locate $Lx$ definition in ldg.txt")
    return "\n".join(out_lines) + "\n"


def replace_scatter_probabilities_in_ldg(text: str, probs: tuple[float, float, float]) -> str:
    out_lines: list[str] = []
    p_diff, p_spec, p_pass = (float(v) for v in probs)
    for raw in text.splitlines():
        stripped = raw.strip()
        if " SCATTER " in f" {stripped} ":
            head, _sep, _tail = stripped.partition("SCATTER")
            updated = f"{head.rstrip()} SCATTER {p_diff:g} {p_spec:g} {p_pass:g}"
            out_lines.append(updated)
        else:
            out_lines.append(raw)
    return "\n".join(out_lines) + "\n"


def build_lgrid(width_um: float) -> str:
    return f"X 2:\n{{0,{width_um:.6f}}}\nY 115:\n{{0,0.01,1.14}}\nZ 2:\n{{0,0.3}}\n"


def build_monitors(width_um: float) -> str:
    y_coords = (0.01, 0.29, 0.54, 0.57, 0.60, 0.85, 1.13)
    lines = ["# x0 x1 y0 y1 z0 z1 direction label"]
    for i, y in enumerate(y_coords, start=1):
        lines.append(f"0 {width_um:.6f} {y:.2f} {y:.2f} 0 0.3 +Y flux_plane_{i:03d}")
    return "\n".join(lines) + "\n"


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_temperature_rows(template_rows: list[dict[str, str]], temp_k: float) -> list[dict[str, object]]:
    n = len(template_rows)
    t_lo = temp_k - DELTA_HALF_K
    t_hi = temp_k + DELTA_HALF_K
    rows: list[dict[str, object]] = []
    for i, row in enumerate(template_rows):
        frac = 0.0 if n <= 1 else i / float(n - 1)
        rows.append(
            {
                "idxcell": int(row["idxcell"]),
                "idycell": int(row["idycell"]),
                "idzcell": int(row["idzcell"]),
                "Temperature": t_lo + (t_hi - t_lo) * frac,
            }
        )
    return rows


def build_tref_rows(template_rows: list[dict[str, str]], temp_k: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in template_rows:
        rows.append(
            {
                "idxcell": int(row["idxcell"]),
                "idycell": int(row["idycell"]),
                "idzcell": int(row["idzcell"]),
                "Tref": temp_k,
            }
        )
    return rows


def update_solver_params(text: str, temp_k: float, fixed_initial_particles: int) -> str:
    replacements = {
        "max_steps": "40000",
        "initial_particles_fixed": str(max(0, int(fixed_initial_particles))),
        "ET_table_T_min": f"{temp_k - TABLE_MARGIN_K:.6f}",
        "ET_table_T_max": f"{temp_k + TABLE_MARGIN_K:.6f}",
        "ET_table_nT": "2001",
        "Tloc_table_T_min": f"{temp_k - TABLE_MARGIN_K:.6f}",
        "Tloc_table_T_max": f"{temp_k + TABLE_MARGIN_K:.6f}",
        "Tloc_table_nT": "2001",
    }
    out_lines: list[str] = []
    seen_keys: set[str] = set()
    for raw in text.splitlines():
        stripped = raw.strip()
        replaced = False
        for key, value in replacements.items():
            if stripped.startswith(f"{key} ="):
                suffix = ""
                if "#" in raw:
                    suffix = "  #" + raw.split("#", 1)[1]
                out_lines.append(f"{key} = {value}{suffix}")
                seen_keys.add(key)
                replaced = True
                break
        if not replaced:
            out_lines.append(raw)
            if stripped.startswith("E_eff =") and "initial_particles_fixed" not in seen_keys:
                out_lines.append(f"initial_particles_fixed = {replacements['initial_particles_fixed']}")
                seen_keys.add("initial_particles_fixed")
    if "initial_particles_fixed" not in seen_keys:
        out_lines.append(f"initial_particles_fixed = {replacements['initial_particles_fixed']}")
    return "\n".join(out_lines) + "\n"


def write_manifest(path: Path, case_spec: dict[str, object], temp_k: float) -> None:
    l_transport_m = (1.14 - 0.02) * 1e-6
    delta_t_k = 2.0 * DELTA_HALF_K
    rows = [
        ["width_um", f"{float(case_spec['width_um']):.6f}"],
        ["width_nm", f"{float(case_spec['width_nm']):.6f}"],
        ["width_label", str(case_spec["label"])],
        ["temperature_K", f"{temp_k:.6f}"],
        ["delta_T_K", f"{delta_t_k:.6f}"],
        ["transport_length_m", f"{l_transport_m:.12e}"],
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)


def prepare_case(
    template_dir: Path,
    output_root: Path,
    case_spec: dict[str, object],
    temp_k: float,
    scatter_probs: tuple[float, float, float],
    fixed_initial_particles: int,
    case_suffix: str,
) -> Path:
    width_um = float(case_spec["width_um"])
    target_dir = output_root / case_name(case_spec, temp_k, case_suffix)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(template_dir, target_dir)

    ldg_text = (target_dir / "ldg.txt").read_text(encoding="utf-8")
    ldg_text = replace_lx_in_ldg(ldg_text, width_um)
    ldg_text = replace_scatter_probabilities_in_ldg(ldg_text, scatter_probs)
    (target_dir / "ldg.txt").write_text(ldg_text, encoding="utf-8")
    (target_dir / "lgrid.txt").write_text(build_lgrid(width_um), encoding="utf-8")
    (target_dir / "heat_flux_monitors.txt").write_text(build_monitors(width_um), encoding="utf-8")

    initial_rows = load_csv_rows(template_dir / "initial_temperature.csv")
    reference_rows = load_csv_rows(template_dir / "reference_temperature.txt")
    write_csv_rows(
        target_dir / "initial_temperature.csv",
        build_temperature_rows(initial_rows, temp_k),
        ["idxcell", "idycell", "idzcell", "Temperature"],
    )
    write_csv_rows(
        target_dir / "reference_temperature.txt",
        build_tref_rows(reference_rows, temp_k),
        ["idxcell", "idycell", "idzcell", "Tref"],
    )
    (target_dir / "solver_params.toml").write_text(
        update_solver_params((target_dir / "solver_params.toml").read_text(encoding="utf-8"), temp_k, fixed_initial_particles),
        encoding="utf-8",
    )
    write_manifest(target_dir / "input_manifest.txt", case_spec, temp_k)
    return target_dir


def resolve_case_specs(selected_tokens: list[str]) -> list[dict[str, object]]:
    if not selected_tokens:
        return list(CASE_SPECS)
    token_map = {str(spec["dir_token"]): spec for spec in CASE_SPECS}
    missing = [token for token in selected_tokens if token not in token_map]
    if missing:
        raise ValueError(f"unknown width token(s): {missing}")
    return [token_map[token] for token in selected_tokens]


def main() -> None:
    args = parse_args()
    root = Path(args.output_root).expanduser().resolve()
    template_dir = (root / args.template_dir).resolve()
    if not template_dir.is_dir():
        raise FileNotFoundError(template_dir)
    scatter_probs = tuple(float(v) for v in args.scatter_probabilities)
    if abs(sum(scatter_probs) - 1.0) > 1e-9:
        raise ValueError(f"scatter probabilities must sum to 1.0, got {scatter_probs}")
    case_specs = resolve_case_specs(list(args.width_token))
    temperatures = [float(v) for v in (args.temperature or TEMPS_K)]
    created: list[Path] = []
    for case_spec in case_specs:
        for temp_k in temperatures:
            created.append(
                prepare_case(
                    template_dir,
                    root,
                    case_spec,
                    temp_k,
                    scatter_probs,
                    int(args.fixed_initial_particles),
                    str(args.case_suffix).strip(),
                )
            )
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
