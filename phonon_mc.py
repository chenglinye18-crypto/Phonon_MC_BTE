from __future__ import annotations

import csv
import math
import os
import re
import shutil
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numba import njit, prange, set_num_threads
from scipy.interpolate import PchipInterpolator


K_B = 1.380649e-23
HBAR = 1.054571817e-34
REALMIN = np.finfo(np.float64).tiny
REPO_DIR = Path(__file__).resolve().parent
MATLAB_DIR = REPO_DIR / "Matlab"
INPUT_DIR = REPO_DIR / "input"
LEGACY_INPUT_DIR = MATLAB_DIR / "input"


def ensure_num_threads(n_threads: int | None = None) -> int:
    if n_threads is None or n_threads <= 0:
        n_threads = max(1, os.cpu_count() or 1)
    try:
        set_num_threads(int(n_threads))
    except Exception:
        pass
    return int(n_threads)


@dataclass
class ParticleBlock:
    id: np.ndarray
    par_id: np.ndarray
    cell: np.ndarray
    material_id: np.ndarray  # 0-based index into the multi-material specs list; -1 = unassigned
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    b: np.ndarray
    m: np.ndarray
    w: np.ndarray
    q: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    vabs: np.ndarray
    E: np.ndarray
    sgn: np.ndarray
    n_ph: np.ndarray
    seed: np.ndarray
    t_left: np.ndarray

    @classmethod
    def empty(cls) -> "ParticleBlock":
        return cls(
            id=np.zeros(0, dtype=np.int64),
            par_id=np.zeros(0, dtype=np.int64),
            cell=np.zeros(0, dtype=np.int32),
            material_id=np.zeros(0, dtype=np.int32),
            x=np.zeros(0, dtype=np.float64),
            y=np.zeros(0, dtype=np.float64),
            z=np.zeros(0, dtype=np.float64),
            b=np.zeros(0, dtype=np.int32),
            m=np.zeros(0, dtype=np.int32),
            w=np.zeros(0, dtype=np.float64),
            q=np.zeros(0, dtype=np.float64),
            vx=np.zeros(0, dtype=np.float64),
            vy=np.zeros(0, dtype=np.float64),
            vz=np.zeros(0, dtype=np.float64),
            vabs=np.zeros(0, dtype=np.float64),
            E=np.zeros(0, dtype=np.float64),
            sgn=np.zeros(0, dtype=np.int8),
            n_ph=np.zeros(0, dtype=np.float64),
            seed=np.zeros(0, dtype=np.int64),
            t_left=np.zeros(0, dtype=np.float64),
        )

    def copy(self) -> "ParticleBlock":
        return ParticleBlock(
            id=self.id.copy(),
            par_id=self.par_id.copy(),
            cell=self.cell.copy(),
            material_id=self.material_id.copy(),
            x=self.x.copy(),
            y=self.y.copy(),
            z=self.z.copy(),
            b=self.b.copy(),
            m=self.m.copy(),
            w=self.w.copy(),
            q=self.q.copy(),
            vx=self.vx.copy(),
            vy=self.vy.copy(),
            vz=self.vz.copy(),
            vabs=self.vabs.copy(),
            E=self.E.copy(),
            sgn=self.sgn.copy(),
            n_ph=self.n_ph.copy(),
            seed=self.seed.copy(),
            t_left=self.t_left.copy(),
        )

    def __len__(self) -> int:
        return int(self.id.size)

    @property
    def v(self) -> np.ndarray:
        if len(self) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return np.column_stack((self.vx, self.vy, self.vz))

    @staticmethod
    def _take_array(arr: np.ndarray, idx: np.ndarray | slice) -> np.ndarray:
        out = arr[idx]
        if isinstance(idx, slice):
            return out.copy()
        return out

    def take(self, idx: np.ndarray | slice) -> "ParticleBlock":
        return ParticleBlock(
            id=self._take_array(self.id, idx),
            par_id=self._take_array(self.par_id, idx),
            cell=self._take_array(self.cell, idx),
            material_id=self._take_array(self.material_id, idx),
            x=self._take_array(self.x, idx),
            y=self._take_array(self.y, idx),
            z=self._take_array(self.z, idx),
            b=self._take_array(self.b, idx),
            m=self._take_array(self.m, idx),
            w=self._take_array(self.w, idx),
            q=self._take_array(self.q, idx),
            vx=self._take_array(self.vx, idx),
            vy=self._take_array(self.vy, idx),
            vz=self._take_array(self.vz, idx),
            vabs=self._take_array(self.vabs, idx),
            E=self._take_array(self.E, idx),
            sgn=self._take_array(self.sgn, idx),
            n_ph=self._take_array(self.n_ph, idx),
            seed=self._take_array(self.seed, idx),
            t_left=self._take_array(self.t_left, idx),
        )

    def append(self, other: "ParticleBlock") -> "ParticleBlock":
        if len(self) == 0:
            return other.copy()
        if len(other) == 0:
            return self.copy()
        return ParticleBlock(
            id=np.concatenate((self.id, other.id)),
            par_id=np.concatenate((self.par_id, other.par_id)),
            cell=np.concatenate((self.cell, other.cell)),
            material_id=np.concatenate((self.material_id, other.material_id)),
            x=np.concatenate((self.x, other.x)),
            y=np.concatenate((self.y, other.y)),
            z=np.concatenate((self.z, other.z)),
            b=np.concatenate((self.b, other.b)),
            m=np.concatenate((self.m, other.m)),
            w=np.concatenate((self.w, other.w)),
            q=np.concatenate((self.q, other.q)),
            vx=np.concatenate((self.vx, other.vx)),
            vy=np.concatenate((self.vy, other.vy)),
            vz=np.concatenate((self.vz, other.vz)),
            vabs=np.concatenate((self.vabs, other.vabs)),
            E=np.concatenate((self.E, other.E)),
            sgn=np.concatenate((self.sgn, other.sgn)),
            n_ph=np.concatenate((self.n_ph, other.n_ph)),
            seed=np.concatenate((self.seed, other.seed)),
            t_left=np.concatenate((self.t_left, other.t_left)),
        )


@dataclass
class SimulationState:
    p: ParticleBlock
    WE: float
    Wp: float
    Nsp_cell: np.ndarray
    enhance_factor: np.ndarray
    info: dict[str, Any]


def get_or(s: Any, name: str, default_v: Any) -> Any:
    if isinstance(s, dict) and name in s and s[name] is not None:
        return s[name]
    return default_v


def as_path(path_like: Any) -> Path:
    return Path(path_like).expanduser().resolve()


def resolve_base_dir(base_dir: str | Path | None = None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    return REPO_DIR


def resolve_input_dir(base_dir: str | Path | None = None) -> Path:
    if base_dir is not None:
        base = Path(base_dir)
        if base.name == "input" and base.is_dir():
            return base
        candidate = base / "input"
        if candidate.is_dir():
            return candidate
        return base
    if INPUT_DIR.is_dir():
        return INPUT_DIR
    return LEGACY_INPUT_DIR


def load_solver_param_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def apply_solver_param_config(opts: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if not config:
        return opts
    if "E_eff" in config:
        opts["E_eff"] = float(config["E_eff"])
    for key in ("dt", "dt_min", "dt_max", "ET_table_T_min", "ET_table_T_max", "Tloc_table_T_min", "Tloc_table_T_max", "p_target", "volume_heat_source_length_scale"):
        if key in config and config[key] is not None:
            opts[key] = float(config[key])
    if "ET_table_nT" in config and config["ET_table_nT"] is not None:
        opts["ET_table_nT"] = int(config["ET_table_nT"])
    if "Tloc_table_nT" in config and config["Tloc_table_nT"] is not None:
        opts["Tloc_table_nT"] = int(config["Tloc_table_nT"])
    if "max_steps" in config and config["max_steps"] is not None:
        opts["max_steps"] = int(config["max_steps"])
    if "initial_particles_fixed" in config and config["initial_particles_fixed"] is not None:
        opts["initial_particles_fixed"] = max(0, int(config["initial_particles_fixed"]))
    output_cfg = dict(config.get("output", {}))
    if output_cfg:
        out = dict(get_or(opts, "output", {}))
        if "every_n_steps" in output_cfg and output_cfg["every_n_steps"] is not None:
            out["every_n_steps"] = max(1, int(round(output_cfg["every_n_steps"])))
        opts["output"] = out
    reservoir_cfg = dict(config.get("reservoir", {}))
    if reservoir_cfg:
        res = dict(get_or(opts, "reservoir", {}))
        if "refresh_every_n_steps" in reservoir_cfg and reservoir_cfg["refresh_every_n_steps"] is not None:
            res["refresh_every_n_steps"] = max(1, int(round(reservoir_cfg["refresh_every_n_steps"])))
        if "refresh_at_step1" in reservoir_cfg and reservoir_cfg["refresh_at_step1"] is not None:
            res["refresh_at_step1"] = bool(reservoir_cfg["refresh_at_step1"])
        opts["reservoir"] = res
    # -- material aliases ---------------------------------------------------
    aliases = config.get("material_aliases", {})
    if isinstance(aliases, dict):
        for alias, canonical in aliases.items():
            register_material_alias(str(alias), str(canonical))
    # -- scattering parameters ----------------------------------------------
    scattering = dict(config.get("scattering", {}))
    _SCATTER_KEYS = ("BL", "BTN", "BTU", "tau_LTO_ps", "A_imp", "B_imp", "C_imp",
                     "PB_Tsi", "PB_bulk_L", "PB_bulk_F", "PB_Delta", "transport_n")
    if scattering:
        # Detect format: if every value is a dict (i.e. has no scalar leaves),
        # treat as per-material.  Otherwise treat as flat global scattering.
        all_tables = all(isinstance(v, dict) for v in scattering.values())
        if all_tables:
            # Per-material format:  [scattering.SILICON]  etc.
            per_mat: dict[str, dict[str, Any]] = {}
            for mat_key, params in scattering.items():
                per_mat[str(mat_key).strip().upper()] = {k: params[k] for k in _SCATTER_KEYS if k in params}
            opts["material_scattering"] = per_mat
            # Also set global scattering keys from the first material for backward compat.
            if per_mat:
                first = next(iter(per_mat.values()))
                for key in _SCATTER_KEYS:
                    if key in first:
                        opts[key] = first[key]
        else:
            # Flat format (legacy): apply to all materials.
            opts["material_scattering"] = {}  # empty = use global fallback
            for key in _SCATTER_KEYS:
                if key in scattering:
                    opts[key] = scattering[key]
    return opts


def clamp_vec(x: np.ndarray | float, lo: float, hi: float) -> np.ndarray | float:
    return np.clip(x, lo, hi)


def safe_pchip_data(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    keep = np.ones_like(x, dtype=bool)
    keep[1:] = np.diff(x) > 0
    return x[keep], y[keep]


def build_clamped_pchip(x: np.ndarray, y: np.ndarray) -> tuple[PchipInterpolator, np.ndarray]:
    x2, y2 = safe_pchip_data(x, y)
    if x2.size == 1:
        x2 = np.array([x2[0], x2[0] + 1.0], dtype=np.float64)
        y2 = np.array([y2[0], y2[0]], dtype=np.float64)
    return PchipInterpolator(x2, y2, extrapolate=True), x2


def eval_clamped(interp: PchipInterpolator, x_support: np.ndarray, xq: np.ndarray | float) -> np.ndarray:
    xq_arr = np.asarray(xq, dtype=np.float64)
    xq_clamped = np.clip(xq_arr, x_support[0], x_support[-1])
    return np.asarray(interp(xq_clamped), dtype=np.float64)


def ensure_2d_dw(spec: dict[str, Any]) -> np.ndarray:
    dw = np.asarray(spec["dw"], dtype=np.float64)
    B, Nw = spec["w_mid"].shape
    if dw.ndim == 1:
        if dw.size != Nw:
            raise ValueError("spec.dw length must match Nw")
        return np.tile(dw.reshape(1, -1), (B, 1))
    if dw.shape != (B, Nw):
        raise ValueError("spec.dw must be 1xNw or BxNw")
    return dw


def bose_occupation(w: np.ndarray, T: float | np.ndarray) -> np.ndarray:
    T_use = np.maximum(np.asarray(T, dtype=np.float64), 1e-12)
    x = HBAR * np.asarray(w, dtype=np.float64) / (K_B * T_use)
    return 1.0 / np.maximum(np.exp(np.minimum(x, 700.0)) - 1.0, REALMIN)


def rand_unit_vec_batch(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    u1 = np.random.random(n)
    u2 = np.random.random(n)
    cz = 2.0 * u1 - 1.0
    sz = np.sqrt(np.maximum(0.0, 1.0 - cz * cz))
    phi = 2.0 * np.pi * u2
    return np.column_stack((sz * np.cos(phi), sz * np.sin(phi), cz))


def rand_hemisphere_vec(normal: str) -> np.ndarray:
    dirs = rand_unit_vec_batch(1)[0]
    key = normal.upper()
    if key == "+X":
        dirs[0] = -abs(dirs[0])
    elif key == "-X":
        dirs[0] = abs(dirs[0])
    elif key == "+Y":
        dirs[1] = -abs(dirs[1])
    elif key == "-Y":
        dirs[1] = abs(dirs[1])
    elif key == "+Z":
        dirs[2] = -abs(dirs[2])
    elif key == "-Z":
        dirs[2] = abs(dirs[2])
    else:
        raise ValueError(f"invalid normal for hemisphere sampling: {normal}")
    return dirs


def parse_scatter_probabilities(tokens: list[str], vars_input: dict[str, float], line: str) -> np.ndarray:
    if len(tokens) < 12:
        raise ValueError(
            f'invalid SCATTER rule: expected "SCATTER p_diffuse p_specular p_pass", got: {line}'
        )
    probs = np.array([eval_expr(tok, vars_input) for tok in tokens[9:12]], dtype=np.float64)
    tol = 1e-12
    if np.any(~np.isfinite(probs)):
        raise ValueError(f"SCATTER probabilities must be finite: {line}")
    if np.any(probs < -tol):
        raise ValueError(f"SCATTER probabilities must be non-negative: {line}")
    probs = np.maximum(probs, 0.0)
    if abs(float(probs.sum()) - 1.0) > 1e-9:
        raise ValueError(
            f"SCATTER probabilities must sum to 1.0 (diffuse/specular/pass), got {probs.tolist()} in: {line}"
        )
    return probs


def sub2ind(nx: int, ny: int, nz: int, ix: np.ndarray | int, iy: np.ndarray | int, iz: np.ndarray | int) -> np.ndarray:
    ix_arr = np.asarray(ix, dtype=np.int64)
    iy_arr = np.asarray(iy, dtype=np.int64)
    iz_arr = np.asarray(iz, dtype=np.int64)
    return ix_arr + (iy_arr - 1) * nx + (iz_arr - 1) * nx * ny


def ind2sub(nx: int, ny: int, nz: int, cid: np.ndarray | int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cid_arr = np.asarray(cid, dtype=np.int64) - 1
    ix = cid_arr % nx + 1
    iy = (cid_arr // nx) % ny + 1
    iz = cid_arr // (nx * ny) + 1
    return ix.astype(np.int64), iy.astype(np.int64), iz.astype(np.int64)


def read_clean_lines(filepath: str | Path) -> list[str]:
    text = Path(filepath).read_text(encoding="utf-8")
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.replace("\ufeff", "")
        line = re.sub(r"[#%].*$", "", line).strip()
        if line:
            lines.append(line)
    return lines


def read_numeric_matrix(filepath: str | Path, delimiter: str | None = ",") -> np.ndarray:
    data = np.genfromtxt(filepath, delimiter=delimiter, comments="#", dtype=np.float64)
    if data.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    valid = np.any(np.isfinite(data), axis=1)
    return np.asarray(data[valid], dtype=np.float64)


def write_csv_rows(filepath: str | Path, rows: Iterable[Iterable[Any]]) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(list(row))


def mc_default_opts(base_dir: str | Path | None = None) -> dict[str, Any]:
    base_dir = resolve_base_dir(base_dir)
    input_dir = resolve_input_dir(base_dir)
    ensure_num_threads()
    now_tag = time.strftime("%Y%m%d_%H%M%S")
    solver_param_file = input_dir / "solver_params.toml"
    opts: dict[str, Any] = {
        "T0": None,
        "initial_temperature_file": str(input_dir / "initial_temperature.csv"),
        "reference_temperature_file": str(input_dir / "reference_temperature.txt"),
        "volume_heat_source_file": str(input_dir / "volume_heat_source.txt"),
        "volume_heat_source_length_scale": 1e-6,
        "solver_param_file": str(solver_param_file),
        "dt": 0.0,
        "dt_min": 1e-15,
        "dt_max": 1e-11,
        "initial_particles_fixed": 0,
        "ET_table_T_min": 1.0,
        "ET_table_T_max": 2000.0,
        "ET_table_nT": 2001,
        "Tloc_table_T_min": 1.0,
        "Tloc_table_T_max": 2000.0,
        "Tloc_table_nT": 2001,
        "fly_mode": "cell",
        "dt_safety_cfl": 0.5,
        "p_target": 0.05,
        "stop_when_steady": True,
        "steady_tol_inf": 1e-2,
        "steady_tol_l2": 1e-2,
        "steady_min_steps": 50000,
        "steady_streak_need": 3,
        "mc_seed": 20240511,
        "n_q": 5000,
        "n_w": 1000,
        "weight_by_Cv_for_Q": True,
        "output": {
            "enable": True,
            "every_n_steps": 100,
            "root_dir": "output",
            "run_tag": now_tag,
            "heat_flux_monitor_file": str(input_dir / "heat_flux_monitors.txt"),
            "monitor_length_scale": 1e-6,
        },
        "reservoir": {
            "enable": True,
            "refresh_every_n_steps": 100,
            "refresh_at_step1": True,
        },
        "scatter_on": True,
        "tau_LTO_ps": 3.5,
        "A_imp": 1.32e-45,
        "B_imp": 0.0,
        "C_imp": 0.0,
        "PB_Tsi": 100e-9,
        "PB_bulk_L": 7.16e-3,
        "PB_bulk_F": 0.68,
        "PB_Delta": 0.0,
        "max_steps": 50000,
        "T_underrelax": 0.5,
        "parallel": {
            "num_threads": max(1, os.cpu_count() or 1),
        },
        "log": {
            "on": True,
            "to_file": False,
            "filename": "mc_log.txt",
            "print_every": 1,
            "fly_chunk": 200000,
        },
        "viz": {
            "enable": False,
        },
        "Tref": 350.0,
        "use_bin_center_w": True,
        "mode": "deviational",
        "enhance_factor": 1.0,
        "max_particles": int(2e8),
    }
    return apply_solver_param_config(opts, load_solver_param_config(solver_param_file))


def et_lookup_cfg_from_opts(opts: dict[str, Any]) -> dict[str, Any]:
    cfg = {
        "T_min": float(get_or(opts, "ET_table_T_min", 1.0)),
        "T_max": float(get_or(opts, "ET_table_T_max", 2000.0)),
        "nT": int(get_or(opts, "ET_table_nT", 2001)),
    }
    if str(get_or(opts, "mode", "absolute")).lower() == "deviational" and np.isfinite(get_or(opts, "Tref", np.nan)):
        cfg["Tref"] = float(opts["Tref"])
    return cfg


def tloc_lookup_cfg_from_opts(opts: dict[str, Any]) -> dict[str, Any]:
    cfg = {
        "T_min": float(get_or(opts, "Tloc_table_T_min", get_or(opts, "ET_table_T_min", 1.0))),
        "T_max": float(get_or(opts, "Tloc_table_T_max", get_or(opts, "ET_table_T_max", 2000.0))),
        "nT": int(get_or(opts, "Tloc_table_nT", get_or(opts, "ET_table_nT", 2001))),
    }
    if str(get_or(opts, "mode", "absolute")).lower() == "deviational" and np.isfinite(get_or(opts, "Tref", np.nan)):
        cfg["Tref"] = float(opts["Tref"])
    return cfg


def setup_case_from_ldg_lgrid(
    ldg_file: str | Path | None = None,
    lgrid_file: str | Path | None = None,
    length_scale: float = 1e-6,
    input_length_unit: str = "um",
    verbose: bool = True,
) -> dict[str, Any]:
    if ldg_file is None:
        ldg_file = resolve_input_dir() / "ldg.txt"
    if lgrid_file is None:
        lgrid_file = resolve_input_dir() / "lgrid.txt"
    layout = parse_ldg(ldg_file, length_scale)
    grid = parse_lgrid(lgrid_file, length_scale)
    validate_layout_vs_grid(layout, grid)
    layout = build_layout_from_rules(layout)
    geom = {
        "shape": "box",
        "origin": np.array([grid["x_edges"][0], grid["y_edges"][0], grid["z_edges"][0]], dtype=np.float64),
        "L": np.array(
            [
                grid["x_edges"][-1] - grid["x_edges"][0],
                grid["y_edges"][-1] - grid["y_edges"][0],
                grid["z_edges"][-1] - grid["z_edges"][0],
            ],
            dtype=np.float64,
        ),
    }
    mesh = {
        "Nx": grid["Nx"],
        "Ny": grid["Ny"],
        "Nz": grid["Nz"],
        "x_edges": grid["x_edges"],
        "y_edges": grid["y_edges"],
        "z_edges": grid["z_edges"],
    }
    cs = {
        "units": {"length": "m", "input_length": input_length_unit, "temp": "K"},
        "geom": geom,
        "mesh": mesh,
        "regions": layout["regions"],
        "materials": layout["materials"],
        "layout": layout,
        "grid": grid,
    }
    if verbose:
        print_case_summary(cs, length_scale)
    return cs


def parse_ldg(filepath: str | Path, length_scale: float) -> dict[str, Any]:
    lines = read_clean_lines(filepath)
    vars_input: dict[str, float] = {}
    vars_si: dict[str, float] = {}
    regions: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    reservoirs: list[dict[str, Any]] = []
    for line in lines:
        tokens = re.split(r"\s+", line)
        head = tokens[0].lower()
        if tokens[0].startswith("$") and tokens[0].endswith("$"):
            var_name = re.sub(r"^\$|\$$", "", tokens[0])
            expr = " ".join(tokens[1:]).strip()
            value_input = eval_expr(expr, vars_input)
            vars_input[var_name] = value_input
            vars_si[var_name] = value_input * length_scale
            continue
        if head == "region":
            if len(tokens) < 8:
                raise ValueError(f"invalid region line: {line}")
            bounds_input = np.array([eval_expr(tokens[1 + k], vars_input) for k in range(6)], dtype=np.float64)
            regions.append(
                {
                    "bounds_input": bounds_input,
                    "bounds": bounds_input * length_scale,
                    "material": tokens[7],
                    "raw": line,
                }
            )
            continue
        if head in {"planerule", "lanerule"}:
            if len(tokens) < 9:
                raise ValueError(f"invalid rule line: {line}")
            bounds_input = np.array([eval_expr(tokens[1 + k], vars_input) for k in range(6)], dtype=np.float64)
            mode_name = tokens[8].upper()
            rule = {
                "kind": head,
                "bounds_input": bounds_input,
                "bounds": bounds_input * length_scale,
                "normal": tokens[7].upper(),
                "mode": mode_name,
                "scatter_probs": parse_scatter_probabilities(tokens, vars_input, line) if mode_name == "SCATTER" else None,
                "raw": line,
            }
            rules.append(finalize_rule(rule, vars_si))
            continue
        if head == "reservoir":
            if len(tokens) < 7:
                raise ValueError(f"invalid reservoir line: {line}")
            bounds_input = np.array([eval_expr(tokens[1 + k], vars_input) for k in range(6)], dtype=np.float64)
            reservoirs.append(
                {
                    "id": len(reservoirs) + 1,
                    "name": f"reservoir_{len(reservoirs) + 1}",
                    "bounds_input": bounds_input,
                    "bounds": bounds_input * length_scale,
                    "raw": line,
                }
            )
            continue
        raise ValueError(f"unsupported ldg entry: {line}")
    materials: list[str] = []
    seen: set[str] = set()
    for reg in regions:
        key = reg["material"].upper()
        if key not in seen:
            materials.append(reg["material"])
            seen.add(key)
    return {
        "source": str(as_path(filepath)),
        "variables_input": vars_input,
        "variables_si": vars_si,
        "regions": regions,
        "materials": materials,
        "rules": rules,
        "reservoirs": reservoirs,
    }


def parse_lgrid(filepath: str | Path, length_scale: float) -> dict[str, Any]:
    lines = read_clean_lines(filepath)
    axes: dict[str, Any] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.match(r"^([XYZxyz])\s+(\d+)\s*:?\s*(.*)$", line)
        if match is None:
            raise ValueError(f"invalid lgrid header: {line}")
        axis_name = match.group(1).upper()
        n_points = int(match.group(2))
        tail = match.group(3).strip()
        if not tail:
            i += 1
            if i >= len(lines):
                raise ValueError(f"missing node list for axis {axis_name}")
            tail = lines[i]
        anchors_input = parse_braced_list(tail)
        edges_input = expand_axis_points(anchors_input, n_points)
        axes[axis_name.lower()] = {
            "n_points": n_points,
            "anchors_input": anchors_input.astype(np.float64),
            "edges_input": edges_input.astype(np.float64),
            "edges": edges_input.astype(np.float64) * length_scale,
        }
        i += 1
    for tag in ("x", "y", "z"):
        if tag not in axes:
            raise ValueError("lgrid must define X, Y and Z")
    return {
        "source": str(as_path(filepath)),
        "axes": axes,
        "x_edges": axes["x"]["edges"],
        "y_edges": axes["y"]["edges"],
        "z_edges": axes["z"]["edges"],
        "Nx": axes["x"]["edges"].size - 1,
        "Ny": axes["y"]["edges"].size - 1,
        "Nz": axes["z"]["edges"].size - 1,
    }


def parse_braced_list(line: str) -> np.ndarray:
    match = re.match(r"^\{(.*)\}$", line.strip())
    if match is None:
        raise ValueError(f"expected braced list, got: {line}")
    parts = [p.strip() for p in match.group(1).split(",")]
    values = [eval_expr(part, {}) for part in parts]
    return np.asarray(values, dtype=np.float64)


def expand_axis_points(anchors: np.ndarray, n_points: int) -> np.ndarray:
    anchors = np.asarray(anchors, dtype=np.float64).reshape(-1)
    if n_points < 2 or anchors.size < 2:
        raise ValueError("axis requires at least two points")
    if np.any(np.diff(anchors) <= 0):
        raise ValueError("axis anchors must be strictly increasing")
    if n_points == anchors.size:
        return anchors.copy()
    uniform_points = np.linspace(anchors[0], anchors[-1], n_points)
    tol = 1e-12 * max(1.0, abs(anchors[-1] - anchors[0]))
    if all(np.any(np.abs(uniform_points - a) <= tol) for a in anchors):
        return uniform_points
    n_seg = anchors.size - 1
    n_intervals = n_points - 1
    if n_intervals < n_seg:
        raise ValueError("grid point count is too small to include all anchors")
    lengths = np.diff(anchors)
    exact = lengths / lengths.sum() * n_intervals
    counts = np.floor(exact).astype(np.int64)
    remainder = n_intervals - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(exact - counts))
        counts[order[:remainder]] += 1
    while np.any(counts == 0):
        zero_idx = int(np.flatnonzero(counts == 0)[0])
        donors = np.flatnonzero(counts > 1)
        if donors.size == 0:
            raise ValueError("failed to allocate intervals for all anchors")
        donor = int(donors[np.argmax(counts[donors] - exact[donors])])
        counts[donor] -= 1
        counts[zero_idx] = 1
    pieces = [np.array([anchors[0]], dtype=np.float64)]
    for i in range(n_seg):
        seg = np.linspace(anchors[i], anchors[i + 1], int(counts[i]) + 1)
        pieces.append(seg[1:])
    return np.concatenate(pieces)


def eval_expr(expr: str, vars_dict: dict[str, float]) -> float:
    expr_use = expr.strip()
    for token in re.findall(r"\$([A-Za-z]\w*)\$", expr_use):
        if token not in vars_dict:
            raise ValueError(f"undefined variable ${token}$ in expression {expr!r}")
        expr_use = re.sub(rf"\${token}\$", f"{vars_dict[token]:.17g}", expr_use)
    if re.match(r"^[0-9eE\+\-\*\/\.\(\)\s]+$", expr_use) is None:
        raise ValueError(f"unsafe expression {expr!r}")
    value = eval(expr_use, {"__builtins__": {}}, {})
    if not np.isfinite(value):
        raise ValueError(f"expression is not finite: {expr!r}")
    return float(value)


def finalize_rule(rule: dict[str, Any], vars_si: dict[str, float]) -> dict[str, Any]:
    values = np.array(list(vars_si.values()), dtype=np.float64)
    tol = 1e-12 * max(1.0, float(np.max(np.abs(values))) if values.size else 1.0)
    axis_name, coord = rule_axis_and_coord(rule)
    rule["axis"] = axis_name
    rule["coord"] = coord
    rule["patch_area"] = rule_patch_area(rule)
    location, face_tag = rule_location(rule, vars_si, tol)
    rule["location"] = location
    rule["face_tag"] = face_tag
    return rule


def rule_axis_and_coord(rule: dict[str, Any]) -> tuple[str, float]:
    normal = rule["normal"].upper()
    b = np.asarray(rule["bounds"], dtype=np.float64)
    if normal in {"+X", "-X"}:
        return "x", float(0.5 * (b[0] + b[1]))
    if normal in {"+Y", "-Y"}:
        return "y", float(0.5 * (b[2] + b[3]))
    if normal in {"+Z", "-Z"}:
        return "z", float(0.5 * (b[4] + b[5]))
    raise ValueError(f"invalid rule normal {normal}")


def rule_patch_area(rule: dict[str, Any]) -> float:
    b = np.asarray(rule["bounds"], dtype=np.float64)
    normal = rule["normal"].upper()
    if normal in {"+X", "-X"}:
        return max(b[3] - b[2], 0.0) * max(b[5] - b[4], 0.0)
    if normal in {"+Y", "-Y"}:
        return max(b[1] - b[0], 0.0) * max(b[5] - b[4], 0.0)
    if normal in {"+Z", "-Z"}:
        return max(b[1] - b[0], 0.0) * max(b[3] - b[2], 0.0)
    return 0.0


def rule_location(rule: dict[str, Any], vars_si: dict[str, float], tol: float) -> tuple[str, str]:
    if not {"Lx", "Ly", "Lz"}.issubset(vars_si):
        return "internal", ""
    coord = float(rule["coord"])
    normal = rule["normal"].upper()
    if normal == "-X" and abs(coord - 0.0) <= tol:
        return "boundary", "x_min"
    if normal == "+X" and abs(coord - vars_si["Lx"]) <= tol:
        return "boundary", "x_max"
    if normal == "-Y" and abs(coord - 0.0) <= tol:
        return "boundary", "y_min"
    if normal == "+Y" and abs(coord - vars_si["Ly"]) <= tol:
        return "boundary", "y_max"
    if normal == "-Z" and abs(coord - 0.0) <= tol:
        return "boundary", "z_min"
    if normal == "+Z" and abs(coord - vars_si["Lz"]) <= tol:
        return "boundary", "z_max"
    return "internal", ""


def build_layout_from_rules(layout: dict[str, Any]) -> dict[str, Any]:
    face_tags = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
    boundary_patches = {tag: [] for tag in face_tags}
    for rule in layout.get("rules", []):
        if rule.get("location") != "boundary":
            continue
        boundary_patches[rule["face_tag"]].append(
            {
                "face_tag": rule["face_tag"],
                "mode": rule["mode"].upper(),
                "normal": rule["normal"].upper(),
                "scatter_probs": None if rule.get("scatter_probs") is None else np.asarray(rule["scatter_probs"], dtype=np.float64),
                "bounds": np.asarray(rule["bounds"], dtype=np.float64),
                "bounds_input": np.asarray(rule["bounds_input"], dtype=np.float64),
                "patch_area": float(rule["patch_area"]),
                "raw": rule["raw"],
            }
        )
    layout["boundary_patches"] = boundary_patches
    layout["warnings"] = []
    return layout


def validate_layout_vs_grid(layout: dict[str, Any], grid: dict[str, Any]) -> None:
    vars_si = layout.get("variables_si", {})
    if {"Lx", "Ly", "Lz"}.issubset(vars_si):
        declared = np.array([vars_si["Lx"], vars_si["Ly"], vars_si["Lz"]], dtype=np.float64)
        gridded = np.array(
            [
                grid["x_edges"][-1] - grid["x_edges"][0],
                grid["y_edges"][-1] - grid["y_edges"][0],
                grid["z_edges"][-1] - grid["z_edges"][0],
            ],
            dtype=np.float64,
        )
        tol = 1e-12 * max(1.0, float(np.max(np.abs(np.concatenate((declared, gridded))))))
        if np.any(np.abs(declared - gridded) > tol):
            raise ValueError("ldg dimensions [Lx Ly Lz] and lgrid extents do not match")


def print_case_summary(cs: dict[str, Any], length_scale: float) -> None:
    geom = cs["geom"]
    Lin = np.asarray(geom["L"], dtype=np.float64) / length_scale
    print(f"[case] loaded {cs['layout']['source']} and {cs['grid']['source']}")
    print(f"[geom] box | L = ({Lin[0]:.6g}, {Lin[1]:.6g}, {Lin[2]:.6g}) {cs['units']['input_length']}")
    print(
        "[mesh] cells = (%d, %d, %d) | nodes = (%d, %d, %d)"
        % (
            cs["grid"]["Nx"],
            cs["grid"]["Ny"],
            cs["grid"]["Nz"],
            cs["grid"]["axes"]["x"]["n_points"],
            cs["grid"]["axes"]["y"]["n_points"],
            cs["grid"]["axes"]["z"]["n_points"],
        )
    )
    if cs.get("materials"):
        print(f"[mat] regions = {len(cs['regions'])} | materials = {', '.join(cs['materials'])}")
    if cs["layout"].get("reservoirs"):
        print(f"[reservoir] count = {len(cs['layout']['reservoirs'])}")
    for tag in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        patches = cs["layout"]["boundary_patches"].get(tag, [])
        total_area = sum(float(p["patch_area"]) for p in patches)
        face_area = face_area_from_geom(geom, tag)
        coverage = total_area / max(face_area, np.finfo(np.float64).eps)
        print(f"[face] {tag:<5} patches={len(patches)} coverage={100.0 * coverage:.1f}%")


def face_area_from_geom(geom: dict[str, Any], face_tag: str) -> float:
    L = np.asarray(geom["L"], dtype=np.float64)
    if face_tag in {"x_min", "x_max"}:
        return float(L[1] * L[2])
    if face_tag in {"y_min", "y_max"}:
        return float(L[0] * L[2])
    if face_tag in {"z_min", "z_max"}:
        return float(L[0] * L[1])
    raise ValueError(f"invalid face tag {face_tag}")


def init_mesh_from_geom(cs: dict[str, Any]) -> dict[str, Any]:
    X = np.asarray(cs["mesh"]["x_edges"], dtype=np.float64).reshape(-1)
    Y = np.asarray(cs["mesh"]["y_edges"], dtype=np.float64).reshape(-1)
    Z = np.asarray(cs["mesh"]["z_edges"], dtype=np.float64).reshape(-1)
    if np.any(np.diff(X) <= 0) or np.any(np.diff(Y) <= 0) or np.any(np.diff(Z) <= 0):
        raise ValueError("mesh edges must be strictly increasing")
    Nx, Ny, Nz = X.size - 1, Y.size - 1, Z.size - 1
    Nc = Nx * Ny * Nz
    dx, dy, dz = np.diff(X), np.diff(Y), np.diff(Z)
    xc = 0.5 * (X[:-1] + X[1:])
    yc = 0.5 * (Y[:-1] + Y[1:])
    zc = 0.5 * (Z[:-1] + Z[1:])
    Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing="ij")
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), np.arange(1, Nz + 1), indexing="ij")
    xmin = X[I - 1]
    xmax = X[I]
    ymin = Y[J - 1]
    ymax = Y[J]
    zmin = Z[K - 1]
    zmax = Z[K]
    centers = np.column_stack((Xc.ravel(order="F"), Yc.ravel(order="F"), Zc.ravel(order="F")))
    boxes = np.column_stack(
        (
            xmin.ravel(order="F"),
            xmax.ravel(order="F"),
            ymin.ravel(order="F"),
            ymax.ravel(order="F"),
            zmin.ravel(order="F"),
            zmax.ravel(order="F"),
        )
    )
    vol = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
    mesh = {
        "L": np.array([X[-1] - X[0], Y[-1] - Y[0], Z[-1] - Z[0]], dtype=np.float64),
        "Lx": float(X[-1] - X[0]),
        "Ly": float(Y[-1] - Y[0]),
        "Lz": float(Z[-1] - Z[0]),
        "origin": np.array([X[0], Y[0], Z[0]], dtype=np.float64),
        "Nx": Nx,
        "Ny": Ny,
        "Nz": Nz,
        "Nc": Nc,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "dx_min": float(dx.min()),
        "dy_min": float(dy.min()),
        "dz_min": float(dz.min()),
        "hmin": float(min(dx.min(), dy.min(), dz.min())),
        "centers": centers,
        "vol": vol.ravel(order="F"),
        "cell_vol": vol.ravel(order="F"),
        "boxes": boxes,
        "domain_box": np.array([X[0], X[-1], Y[0], Y[-1], Z[0], Z[-1]], dtype=np.float64),
        "x_edges": X,
        "y_edges": Y,
        "z_edges": Z,
        "xc": xc,
        "yc": yc,
        "zc": zc,
        "Xc": Xc,
        "Yc": Yc,
        "Zc": Zc,
        "Ax": float((Y[-1] - Y[0]) * (Z[-1] - Z[0])),
        "Ay": float((X[-1] - X[0]) * (Z[-1] - Z[0])),
        "Az": float((X[-1] - X[0]) * (Y[-1] - Y[0])),
        "V": float(np.sum(vol)),
        "regions": cs.get("regions", []),
        "materials": cs.get("materials", []),
        "layout": cs.get("layout", {}),
        "grid": cs.get("grid", {}),
    }
    attach_cell_materials(mesh)
    attach_reservoirs(mesh)
    if mesh.get("layout"):
        build_layout_behavior(mesh, mesh["layout"])
    else:
        mesh["boundary"] = {"by_face": {}}
        mesh["face_rules"] = {"by_normal": {k: [] for k in ("xp", "xn", "yp", "yn", "zp", "zn")}, "all": []}
    return mesh


def attach_cell_materials(mesh: dict[str, Any]) -> None:
    regions = mesh.get("regions", [])
    Nc = mesh["Nc"]
    centers = mesh["centers"]
    cell_region_index = np.zeros(Nc, dtype=np.int32)
    for ir, region in enumerate(regions, start=1):
        b = np.asarray(region["bounds"], dtype=np.float64)
        in_region = (
            (centers[:, 0] >= b[0])
            & (centers[:, 0] <= b[1])
            & (centers[:, 1] >= b[2])
            & (centers[:, 1] <= b[3])
            & (centers[:, 2] >= b[4])
            & (centers[:, 2] <= b[5])
        )
        cell_region_index[in_region] = ir
    cell_material_name = [""] * Nc
    for cid in range(Nc):
        ir = int(cell_region_index[cid])
        if ir > 0:
            cell_material_name[cid] = regions[ir - 1]["material"]
    assigned = [name for name in cell_material_name if name]
    material_keys: list[str] = []
    seen: set[str] = set()
    for name in assigned:
        key = material_key(name)
        if key not in seen:
            material_keys.append(key)
            seen.add(key)
    cell_material_index = np.zeros(Nc, dtype=np.int32)
    for im, key in enumerate(material_keys, start=1):
        for cid, name in enumerate(cell_material_name):
            if material_key(name) == key:
                cell_material_index[cid] = im
    mesh["cell_region_index"] = cell_region_index
    mesh["cell_material_name"] = cell_material_name
    mesh["cell_material_index"] = cell_material_index
    mesh["material_keys"] = material_keys


def attach_reservoirs(mesh: dict[str, Any]) -> None:
    reservoirs_src = mesh.get("layout", {}).get("reservoirs", [])
    centers = mesh["centers"]
    tol = 1e-12 * max(1.0, float(np.max(np.abs(centers))) if centers.size else 1.0)
    reservoirs: list[dict[str, Any]] = []
    cell_mask = np.zeros(mesh["Nc"], dtype=bool)
    for src in reservoirs_src:
        b = np.asarray(src["bounds"], dtype=np.float64)
        in_res = (
            (centers[:, 0] >= b[0] - tol)
            & (centers[:, 0] <= b[1] + tol)
            & (centers[:, 1] >= b[2] - tol)
            & (centers[:, 1] <= b[3] + tol)
            & (centers[:, 2] >= b[4] - tol)
            & (centers[:, 2] <= b[5] + tol)
        )
        cells = np.flatnonzero(in_res).astype(np.int32) + 1
        reservoirs.append(
            {
                "id": int(src["id"]),
                "name": src["name"],
                "bounds_input": np.asarray(src["bounds_input"], dtype=np.float64),
                "bounds": np.asarray(src["bounds"], dtype=np.float64),
                "cell_ids": cells,
                "raw": src["raw"],
            }
        )
        if cells.size:
            cell_mask[cells - 1] = True
    mesh["reservoirs"] = reservoirs
    mesh["reservoir_cell_mask"] = cell_mask


def build_layout_behavior(mesh: dict[str, Any], layout: dict[str, Any]) -> None:
    mesh["boundary"] = {"by_face": {}}
    face_rules = {"by_normal": {k: [] for k in ("xp", "xn", "yp", "yn", "zp", "zn")}, "all": []}
    for rule in layout.get("rules", []):
        entry = {
            "normal": rule["normal"].upper(),
            "mode": rule["mode"].upper(),
            "scatter_probs": None if rule.get("scatter_probs") is None else np.asarray(rule["scatter_probs"], dtype=np.float64),
            "axis": rule["axis"],
            "coord": float(rule["coord"]),
            "bounds": np.asarray(rule["bounds"], dtype=np.float64),
            "bounds_input": np.asarray(rule["bounds_input"], dtype=np.float64),
            "location": rule["location"],
            "face_tag": rule["face_tag"],
            "raw": rule["raw"],
        }
        face_rules["all"].append(entry)
        face_rules["by_normal"][normal_key(entry["normal"])].append(entry)
    mesh["face_rules"] = face_rules
    for tag in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        mesh["boundary"]["by_face"][tag] = list(layout.get("boundary_patches", {}).get(tag, []))


# Material name aliases — maps common variant names to canonical keys.
# Populated at runtime from solver_params.toml [material_aliases] or kept
# as a built-in baseline. Keys are upper-case.
_MATERIAL_ALIASES: dict[str, str] = {
    "SI": "SILICON",
    "SI_100": "SILICON",
    "SILICON_100": "SILICON",
}


def register_material_alias(alias: str, canonical: str) -> None:
    """Register a material name alias (both are uppercased)."""
    _MATERIAL_ALIASES[str(alias).strip().upper()] = str(canonical).strip().upper()


def material_key(name: str) -> str:
    """Canonicalize a material name to its uppercase key, resolving aliases."""
    key = str(name).strip().upper()
    return _MATERIAL_ALIASES.get(key, key)


def resolve_case_material(cs: dict[str, Any], input_dir: str | Path | None = None) -> dict[str, Any]:
    """Resolve the *primary* material for a single-material simulation.

    For multi-material simulations prefer ``resolve_case_materials()`` directly.
    """
    materials = resolve_case_materials(cs, input_dir=input_dir)
    primary = materials["list"][materials["primary_index"] - 1]["mat"].copy()
    primary["case_material"] = materials["primary_key"]
    primary["case_material_label"] = materials["primary_name"]
    primary["material_library"] = materials
    return primary


def resolve_case_materials(cs: dict[str, Any], input_dir: str | Path | None = None) -> dict[str, Any]:
    """Discover and load every distinct material referenced by the case.

    Materials are resolved from ``region`` lines in the LDG file.  If no
    regions are present the ``materials`` list from the case spec is used as a
    fallback.  When both are empty a ``ValueError`` is raised — the old
    silent-Silicon default is removed.
    """
    if input_dir is None:
        input_dir = resolve_input_dir()
    raw_names = [reg["material"] for reg in cs.get("regions", [])] or cs.get("materials", [])
    requested_names: list[str] = []
    seen: set[str] = set()
    for name in raw_names:
        key = material_key(str(name))
        if key not in seen:
            requested_names.append(str(name))
            seen.add(key)
    if not requested_names:
        raise ValueError(
            "No materials specified in the case. "
            "Add at least one 'region ... MATERIAL' line to ldg.txt."
        )
    entries: list[dict[str, Any]] = []
    for raw_name in requested_names:
        key = material_key(raw_name)
        entries.append({"name": raw_name, "key": key, "mat": load_material(key, raw_name, input_dir=input_dir)})
    primary_name = requested_names[0]
    primary_key = material_key(primary_name)
    primary_index = 1
    for i, entry in enumerate(entries, start=1):
        if entry["key"] == primary_key:
            primary_index = i
            break
    by_key = {entry["key"]: entry["mat"] for entry in entries}
    region_names = [reg["material"] for reg in cs.get("regions", [])]
    region_index = np.zeros(len(region_names), dtype=np.int32)
    for i, region_name in enumerate(region_names):
        key = material_key(region_name)
        for j, entry in enumerate(entries, start=1):
            if entry["key"] == key:
                region_index[i] = j
                break
    return {
        "names": [entry["name"] for entry in entries],
        "keys": [entry["key"] for entry in entries],
        "list": entries,
        "by_key": by_key,
        "primary_name": primary_name,
        "primary_key": primary_key,
        "primary_index": primary_index,
        "region_material_name": region_names,
        "region_material_index": region_index,
    }


def load_material(key: str, raw_name: str, input_dir: str | Path | None = None) -> dict[str, Any]:
    """Load a material by canonical key from a phonon-dispersion file.

    Searches the input directory for ``phonon_dispersion_{key}.txt`` (case-
    sensitive first, then case-insensitive fallback).  Also tries the original
    *raw_name* and common aliases so that a file named
    ``phonon_dispersion_Si.txt`` is found when the canonical key is
    ``SILICON``.
    """
    if input_dir is None:
        input_dir = resolve_input_dir()
    base = Path(input_dir)
    # Collect all candidate file-name stems to try (without the .txt suffix).
    # We try: the canonical key, the raw name, the uppercase raw name, and
    # both key and raw name with common capitalisation.
    search_names: set[str] = set()
    for name in (key, raw_name):
        search_names.add(f"phonon_dispersion_{name}")
        search_names.add(f"phonon_dispersion_{name.upper()}")
        search_names.add(f"phonon_dispersion_{name.lower()}")
        search_names.add(f"phonon_dispersion_{name.capitalize()}")
    # Build a short-list of likely full paths.
    candidates: list[Path] = []
    for stem in sorted(search_names):
        p = base / f"{stem}.txt"
        if p not in candidates:
            candidates.append(p)
    # Also scan the directory for any file whose name starts with
    # "phonon_dispersion_" and whose stem (without extension) matches one
    # of our search names case-insensitively.
    search_lower = {s.lower() for s in search_names}
    if base.is_dir():
        for child in base.iterdir():
            if not child.is_file():
                continue
            cname = child.name
            if not cname.lower().startswith("phonon_dispersion_"):
                continue
            stem_lower = cname.rsplit(".", 1)[0].lower() if "." in cname else cname.lower()
            if stem_lower in search_lower:
                if child not in candidates:
                    candidates.append(child)
    for path in candidates:
        if path.is_file():
            mat = mat_from_phonon_dispersion_file(file_path=path, material_name=raw_name)
            mat["case_material"] = key
            mat["case_material_label"] = raw_name
            return mat
    raise FileNotFoundError(
        f"No dispersion file found for material key={key!r} (raw name={raw_name!r}) "
        f"in {str(base)!r}. Expected a file named phonon_dispersion_{key}.txt"
    )


def parse_header_metadata(filepath: str | Path) -> dict[str, Any]:
    meta: dict[str, Any] = {"branch_names": [], "degeneracy": []}
    for line in Path(filepath).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        if not text.startswith("#"):
            break
        body = text[1:].strip()
        m_names = re.search(r"branch_names\s*=\s*([^;#]+)", body)
        if m_names:
            meta["branch_names"] = [p.strip() for p in m_names.group(1).split(",") if p.strip()]
        m_deg = re.search(r"degeneracy\s*=\s*([^;#]+)", body)
        if m_deg:
            meta["degeneracy"] = [float(p.strip()) for p in m_deg.group(1).split(",") if p.strip()]
    return meta


def unique_samples(q: np.ndarray, f: np.ndarray, vg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_q, inverse = np.unique(q, return_inverse=True)
    if unique_q.size == q.size:
        return q, f, vg
    f_u = np.zeros(unique_q.size, dtype=np.float64)
    vg_u = np.zeros(unique_q.size, dtype=np.float64)
    counts = np.zeros(unique_q.size, dtype=np.int64)
    np.add.at(f_u, inverse, f)
    np.add.at(vg_u, inverse, vg)
    np.add.at(counts, inverse, 1)
    return unique_q, f_u / counts, vg_u / counts


def mat_from_phonon_dispersion_file(
    file_path: str | Path,
    material_name: str = "TableDriven",
    branch_names: list[str] | None = None,
    degeneracy: list[float] | None = None,
) -> dict[str, Any]:
    file_path = as_path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    header_meta = parse_header_metadata(file_path)
    data = np.genfromtxt(file_path, comments="#", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"{file_path} must contain at least four numeric columns")
    data = data[:, :4]
    data = data[np.all(np.isfinite(data), axis=1)]
    branch_id = np.round(data[:, 0]).astype(np.int64)
    q_raw = data[:, 1].astype(np.float64)
    f_thz = data[:, 2].astype(np.float64)
    vg_raw = data[:, 3].astype(np.float64)
    branch_ids: list[int] = []
    seen: set[int] = set()
    for bid in branch_id.tolist():
        if bid not in seen:
            branch_ids.append(int(bid))
            seen.add(int(bid))
    B = len(branch_ids)
    q_common = np.unique(q_raw)
    q_common.sort()
    M = q_common.size
    omega_tab = np.zeros((B, M), dtype=np.float64)
    vg_tab = np.zeros((B, M), dtype=np.float64)
    omega_interp: list[PchipInterpolator] = []
    omega_support: list[np.ndarray] = []
    vg_interp: list[PchipInterpolator] = []
    vg_support: list[np.ndarray] = []
    n_negative_freq = 0
    for ib, bid in enumerate(branch_ids):
        mask = branch_id == bid
        q_b = q_raw[mask]
        f_b = f_thz[mask]
        vg_b = vg_raw[mask]
        order = np.argsort(q_b, kind="stable")
        q_b = q_b[order]
        f_b = f_b[order]
        vg_b = vg_b[order]
        q_b, f_b, vg_b = unique_samples(q_b, f_b, vg_b)
        n_negative_freq += int(np.count_nonzero(f_b < 0))
        omega_b = 2.0 * np.pi * 1e12 * np.maximum(f_b, 0.0)
        om_interp, om_support = build_clamped_pchip(q_b, omega_b)
        vg_i, vg_s = build_clamped_pchip(q_b, vg_b)
        omega_tab[ib] = eval_clamped(om_interp, om_support, q_common)
        vg_tab[ib] = eval_clamped(vg_i, vg_s, q_common)
        omega_interp.append(om_interp)
        omega_support.append(om_support)
        vg_interp.append(vg_i)
        vg_support.append(vg_s)
    if branch_names is None:
        branch_names = header_meta["branch_names"] or [f"B{bid}" for bid in branch_ids]
    if degeneracy is None:
        degeneracy = header_meta["degeneracy"] or [1.0] * B
    if len(branch_names) != B or len(degeneracy) != B:
        raise ValueError("branch_names/degeneracy count does not match inferred branch count")
    mat = {
        "name": material_name,
        "source_file": str(file_path),
        "branch_ids": np.asarray(branch_ids, dtype=np.int64),
        "branch_names": list(branch_names),
        "degeneracy": np.asarray(degeneracy, dtype=np.float64),
        "B": B,
        "q": q_common,
        "qmax": float(q_common.max()),
        "omega_tab": omega_tab,
        "vg_tab": vg_tab,
        "frequency_THz_tab": omega_tab / (2.0 * np.pi * 1e12),
        "n_negative_freq_entries": n_negative_freq,
        "omega_interp": omega_interp,
        "omega_support": omega_support,
        "vg_interp": vg_interp,
        "vg_support": vg_support,
    }
    return mat


def mat_silicon_100(file_path: str | Path | None = None) -> dict[str, Any]:
    """Convenience wrapper — returns Silicon (100) material with hard-coded branch metadata."""
    if file_path is None:
        file_path = resolve_input_dir() / "phonon_dispersion_Si.txt"
    mat = mat_from_phonon_dispersion_file(
        file_path=file_path,
        material_name="Silicon (100)",
        branch_names=["LA", "TA", "LO", "TO"],
        degeneracy=[1, 2, 1, 2],
    )
    mat["a0"] = 5.431e-10
    mat["crystal_orientation"] = "100"
    return mat


def mat_igzo(file_path: str | Path | None = None) -> dict[str, Any]:
    """Convenience wrapper — returns IGZO material.

    Branch metadata is read from the dispersion-file header when available;
    otherwise the defaults in ``mat_from_phonon_dispersion_file`` apply.
    """
    if file_path is None:
        file_path = resolve_input_dir() / "phonon_dispersion_IGZO.txt"
    return mat_from_phonon_dispersion_file(file_path=file_path, material_name="IGZO")


def material_eval_omega(mat: dict[str, Any], branch_index: int, q_values: np.ndarray) -> np.ndarray:
    return eval_clamped(mat["omega_interp"][branch_index], mat["omega_support"][branch_index], q_values)


def material_eval_vg(mat: dict[str, Any], branch_index: int, q_values: np.ndarray) -> np.ndarray:
    return eval_clamped(mat["vg_interp"][branch_index], mat["vg_support"][branch_index], q_values)


def build_branch_lookup(spec: dict[str, Any], branch_index: int) -> dict[str, Any]:
    qv = np.asarray(spec["si"]["q"], dtype=np.float64)
    wv = np.maximum(np.asarray(spec["si"]["omega_tab"][branch_index], dtype=np.float64), 0.0)
    gv = np.asarray(spec["si"]["vg_tab"][branch_index], dtype=np.float64)
    order = np.argsort(wv, kind="stable")
    w_sorted = wv[order]
    q_sorted = qv[order]
    v_sorted = gv[order]
    keep = np.ones(w_sorted.size, dtype=bool)
    keep[1:] = np.diff(w_sorted) > 0
    ws = w_sorted[keep]
    qs = q_sorted[keep]
    vs = v_sorted[keep]
    w_to_q, ws_support = build_clamped_pchip(ws, qs)
    q_to_v, qs_support = build_clamped_pchip(qs, vs)
    return {
        "ws": ws_support,
        "qs": qs_support,
        "vs": vs,
        "w_to_q": w_to_q,
        "q_to_v": q_to_v,
    }


def build_spectral_grid(mat: dict[str, Any], opts: dict[str, Any], global_w_edges: np.ndarray | None = None) -> dict[str, Any]:
    if opts.get("T0") is None:
        raise ValueError("opts.T0 must be provided before build_spectral_grid")
    T0 = float(opts["T0"])
    Mq = int(get_or(opts, "n_q", 5000))
    Nw = int(get_or(opts, "n_w", 1000))
    weight_by_cv = bool(get_or(opts, "weight_by_Cv_for_Q", True))
    q_edges = np.linspace(0.0, float(mat["qmax"]), Mq + 1)
    q_mid = 0.5 * (q_edges[:-1] + q_edges[1:])
    dq = np.diff(q_edges)
    branches = list(mat["branch_names"])
    deg = np.asarray(mat["degeneracy"], dtype=np.float64)
    B = len(branches)
    omega = np.zeros((B, Mq), dtype=np.float64)
    vg = np.zeros((B, Mq), dtype=np.float64)
    Cv = np.zeros((B, Mq), dtype=np.float64)
    for b in range(B):
        w_b = np.maximum(material_eval_omega(mat, b, q_mid), 0.0)
        vg_b = material_eval_vg(mat, b, q_mid)
        x = HBAR * w_b / (K_B * T0)
        ex = np.exp(np.minimum(x, 700.0))
        nbar = 1.0 / np.maximum(ex - 1.0, REALMIN)
        dndT = (HBAR * w_b / (K_B * (T0**2))) * nbar * (nbar + 1.0)
        dos_q = (deg[b] / (2.0 * np.pi**2)) * (q_mid**2) * dq
        omega[b] = w_b
        vg[b] = vg_b
        Cv[b] = (HBAR * w_b) * dndT * dos_q
    face_weight = np.abs(vg) * Cv
    vol_weight = Cv if weight_by_cv else np.ones_like(Cv)
    wf = face_weight.reshape(-1, order="F")
    wv = vol_weight.reshape(-1, order="F")
    cdf_face = np.cumsum(wf)
    if cdf_face.size and cdf_face[-1] > 0:
        cdf_face /= cdf_face[-1]
    else:
        cdf_face[:] = 0.0
    cdf_vol = np.cumsum(wv)
    if cdf_vol.size and cdf_vol[-1] > 0:
        cdf_vol /= cdf_vol[-1]
    else:
        cdf_vol[:] = 0.0
    wtab_all = np.maximum(np.asarray(mat["omega_tab"], dtype=np.float64), 0.0)
    if global_w_edges is not None:
        w_edges = np.asarray(global_w_edges, dtype=np.float64)
        Nw = w_edges.size - 1
    else:
        wmin = max(0.0, float(np.min(wtab_all)))
        wmax = float(np.max(wtab_all))
        if not np.isfinite(wmin) or not np.isfinite(wmax) or wmax <= wmin:
            wmin, wmax = 0.0, 1.0
        w_edges = np.linspace(wmin, wmax, Nw + 1)
    w_mid_1 = 0.5 * (w_edges[:-1] + w_edges[1:])
    dw = np.diff(w_edges)
    w_mid = np.tile(w_mid_1.reshape(1, -1), (B, 1))
    DOS_w_b = np.zeros((B, Nw), dtype=np.float64)
    vg_w = np.zeros((B, Nw), dtype=np.float64)
    qtab = np.asarray(mat["q"], dtype=np.float64)
    for b in range(B):
        wtab = np.maximum(np.asarray(mat["omega_tab"][b], dtype=np.float64), 0.0)
        vgtab = np.abs(np.asarray(mat["vg_tab"][b], dtype=np.float64))
        d = np.diff(wtab)
        brk = [0]
        for i in range(1, d.size):
            if d[i - 1] * d[i] < 0 or d[i - 1] == 0.0 or d[i] == 0.0:
                brk.append(i)
        brk.append(wtab.size - 1)
        Wsum = np.zeros(Nw, dtype=np.float64)
        Vsum = np.zeros(Nw, dtype=np.float64)
        for s in range(len(brk) - 1):
            i1, i2 = brk[s], brk[s + 1]
            if i2 - i1 < 2:
                continue
            qseg = qtab[i1 : i2 + 1]
            wseg = wtab[i1 : i2 + 1]
            unique_w, idx = np.unique(wseg, return_index=True)
            if unique_w.size < 2:
                continue
            qseg_u = qseg[idx]
            wlo, whi = float(unique_w.min()), float(unique_w.max())
            mask = (w_mid_1 >= wlo) & (w_mid_1 <= whi)
            if not np.any(mask):
                continue
            wq = w_mid_1[mask]
            q_of_w = np.interp(wq, unique_w, qseg_u)
            vg_at_q = np.abs(np.interp(q_of_w, qtab, vgtab))
            vg_at_q = np.maximum(vg_at_q, 1e-6)
            q2 = q_of_w * q_of_w
            Wsum[mask] += q2 / vg_at_q
            Vsum[mask] += q2
        DOS_w_b[b] = (deg[b] / (2.0 * np.pi**2)) * Wsum
        nz = Wsum > 0
        vg_w[b, nz] = Vsum[nz] / Wsum[nz]
    DOS_w = DOS_w_b.sum(axis=0)
    xw = HBAR * w_mid_1 / (K_B * T0)
    n_w1 = 1.0 / np.maximum(np.exp(np.minimum(xw, 700.0)) - 1.0, REALMIN)
    N_w = DOS_w_b * (n_w1 * dw).reshape(1, -1)
    spec = {
        "si": mat,
        "T0": T0,
        "q_edges": q_edges,
        "q_mid": q_mid,
        "dq": dq,
        "qmax": float(mat["qmax"]),
        "branches": branches,
        "deg": deg,
        "B": B,
        "M": Mq,
        "omega": omega,
        "vg": vg,
        "Cv": Cv,
        "Cv_tot": float(Cv.sum()),
        "face_weight": face_weight,
        "vol_weight": vol_weight,
        "cdf_face": cdf_face,
        "cdf_vol": cdf_vol,
        "vg_max": float(np.max(np.abs(vg))) if vg.size else 0.0,
        "w_edges": w_edges,
        "w_mid": w_mid,
        "dw": dw,
        "DOS_w": DOS_w,
        "DOS_w_b": DOS_w_b,
        "N_w": N_w,
        "N_w_tot": float(N_w.sum()),
        "vg_w": vg_w,
    }
    spec["branch_lookups"] = [build_branch_lookup(spec, b) for b in range(B)]
    spec["branch_is_la"] = np.array([("LA" in name.upper().replace(" ", "")) for name in branches], dtype=np.bool_)
    spec["branch_is_ta"] = np.array([("TA" in name.upper().replace(" ", "")) for name in branches], dtype=np.bool_)
    spec["branch_is_loto"] = np.array(
        [(("LO" in name.upper().replace(" ", "")) or ("TO" in name.upper().replace(" ", ""))) for name in branches],
        dtype=np.bool_,
    )
    b_ta = next((i for i, name in enumerate(branches) if "TA" in name.upper().replace(" ", "")), 1 if B > 1 else 0)
    spec["omega_cut_ta"] = float(material_eval_omega(mat, b_ta, np.array([0.5 * spec["qmax"]], dtype=np.float64))[0])
    return spec


def build_multimaterial_specs(
    material_library: dict[str, Any],
    opts: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build a spectral-grid spec for every material in the library.

    All materials share the same global frequency grid (*w_edges*) so that
    DMM interface lookups can use a common frequency axis.
    """
    entries = material_library.get("list", [])
    if not entries:
        raise ValueError("material_library contains no materials")
    # Phase 1 — determine global frequency range across all materials.
    Nw = int(get_or(opts, "n_w", 1000))
    w_min_global = np.inf
    w_max_global = -np.inf
    for entry in entries:
        mat = entry["mat"]
        wtab = np.maximum(np.asarray(mat["omega_tab"], dtype=np.float64), 0.0)
        w_min_global = min(w_min_global, float(np.min(wtab)))
        w_max_global = max(w_max_global, float(np.max(wtab)))
    if not np.isfinite(w_min_global) or not np.isfinite(w_max_global) or w_max_global <= w_min_global:
        w_min_global, w_max_global = 0.0, 1.0
    global_w_edges = np.linspace(max(0.0, w_min_global), w_max_global, Nw + 1)
    # Phase 2 — build one spec per material.
    specs: list[dict[str, Any]] = []
    for entry in entries:
        spec = build_spectral_grid(entry["mat"], opts, global_w_edges=global_w_edges)
        spec["material_key"] = entry["key"]
        spec["material_name"] = entry["name"]
        specs.append(spec)
    return specs


def build_E_T_lookup(spec: dict[str, Any], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or {}
    T_min = float(get_or(cfg, "T_min", 1.0))
    T_max = float(get_or(cfg, "T_max", 2000.0))
    nT = int(get_or(cfg, "nT", 2001))
    T = np.linspace(T_min, T_max, nT, dtype=np.float64)
    DOS = np.maximum(np.asarray(spec["DOS_w_b"], dtype=np.float64), 0.0)
    w = np.maximum(np.asarray(spec["w_mid"], dtype=np.float64), 0.0)
    dw = ensure_2d_dw(spec)
    x = (HBAR * w[np.newaxis, :, :]) / (K_B * T[:, np.newaxis, np.newaxis])
    nbe = 1.0 / np.maximum(np.exp(np.minimum(x, 700.0)) - 1.0, REALMIN)
    dE = DOS[np.newaxis, :, :] * (HBAR * w[np.newaxis, :, :]) * nbe * dw[np.newaxis, :, :]
    Ub = dE.sum(axis=2)
    U = Ub.sum(axis=1)
    U_mono = np.maximum.accumulate(U)
    keep = np.ones(U_mono.size, dtype=bool)
    keep[1:] = np.diff(U_mono) > 0
    if np.count_nonzero(keep) < 2:
        keep[:] = True
    U_unique = U_mono[keep]
    T_unique = T[keep]
    inv_interp = PchipInterpolator(U_unique, T_unique, extrapolate=True)
    U_interp = PchipInterpolator(T, U, extrapolate=True)
    lut = {
        "T": T,
        "U": U,
        "Ub": Ub,
        "U_mono": U_unique,
        "T_mono": T_unique,
        "inv_interp": inv_interp,
        "U_interp": U_interp,
        "inv": lambda Utarget: np.asarray(inv_interp(np.clip(np.asarray(Utarget, dtype=np.float64), U_unique[0], U_unique[-1]))),
    }
    if np.isfinite(get_or(cfg, "Tref", np.nan)):
        tref = float(cfg["Tref"])
        lut["Uref"] = float(U_interp(np.clip(tref, T[0], T[-1])))
    return lut


def build_q_T_lookup(spec: dict[str, Any], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or {}
    T_min = float(get_or(cfg, "T_min", 1.0))
    T_max = float(get_or(cfg, "T_max", 2000.0))
    nT = int(get_or(cfg, "nT", 2001))
    T = np.linspace(T_min, T_max, nT, dtype=np.float64)
    DOS = np.maximum(np.asarray(spec["DOS_w_b"], dtype=np.float64), 0.0)
    w = np.maximum(np.asarray(spec["w_mid"], dtype=np.float64), 0.0)
    dw = ensure_2d_dw(spec)
    vg = np.maximum(np.asarray(spec["vg_w"], dtype=np.float64), 0.0)
    x = (HBAR * w[np.newaxis, :, :]) / (K_B * T[:, np.newaxis, np.newaxis])
    nbe = 1.0 / np.maximum(np.exp(np.minimum(x, 700.0)) - 1.0, REALMIN)
    dq = 0.25 * DOS[np.newaxis, :, :] * vg[np.newaxis, :, :] * (HBAR * w[np.newaxis, :, :]) * nbe * dw[np.newaxis, :, :]
    qb = dq.sum(axis=2)
    q = qb.sum(axis=1)
    q_mono = np.maximum.accumulate(q)
    keep = np.ones(q_mono.size, dtype=bool)
    keep[1:] = np.diff(q_mono) > 0
    if np.count_nonzero(keep) < 2:
        keep[:] = True
    q_unique = q_mono[keep]
    T_unique = T[keep]
    inv_interp = PchipInterpolator(q_unique, T_unique, extrapolate=True)
    q_interp = PchipInterpolator(T, q, extrapolate=True)
    return {
        "T": T,
        "q": q,
        "qb": qb,
        "inv_interp": inv_interp,
        "q_interp": q_interp,
        "inv": lambda qtar: np.asarray(inv_interp(np.clip(np.asarray(qtar, dtype=np.float64), q_unique[0], q_unique[-1]))),
    }


def scattering_rate_table_formula(spec: dict[str, Any], T: float, opts: dict[str, Any]) -> np.ndarray:
    T = max(float(T), 1e-12)
    w = np.maximum(np.asarray(spec["w_mid"], dtype=np.float64), 0.0)
    Gamma = np.zeros_like(w, dtype=np.float64)
    branch_is_la = np.asarray(spec["branch_is_la"], dtype=np.bool_).reshape(-1)
    branch_is_ta = np.asarray(spec["branch_is_ta"], dtype=np.bool_).reshape(-1)
    branch_is_loto = np.asarray(spec["branch_is_loto"], dtype=np.bool_).reshape(-1)
    omega_cut = float(spec["omega_cut_ta"])
    BL = float(get_or(opts, "BL", 1.18e-24))
    BTN = float(get_or(opts, "BTN", 10.5e-13))
    BTU = float(get_or(opts, "BTU", 6.95e-18))
    tau_LTO_ps = float(get_or(opts, "tau_LTO_ps", 3.5))
    A_imp = float(get_or(opts, "A_imp", 0.0))
    B_imp = float(get_or(opts, "B_imp", 0.0))
    C_imp = float(get_or(opts, "C_imp", 0.0))
    if np.any(branch_is_la):
        Gamma[branch_is_la, :] = BL * w[branch_is_la, :] ** 2 * T**3
    if np.any(branch_is_ta):
        Gamma[branch_is_ta, :] = BTN * w[branch_is_ta, :] * T**4
        w_ta = w[branch_is_ta, :]
        ta_mask = w_ta > omega_cut
        if np.any(ta_mask):
            x = np.minimum((HBAR * w_ta) / (K_B * T), 700.0)
            add = np.zeros_like(w_ta)
            add[ta_mask] = BTU * w_ta[ta_mask] ** 2 / np.maximum(np.sinh(x[ta_mask]), 1e-12)
            Gamma[branch_is_ta, :] += add
    if np.any(branch_is_loto):
        Gamma[branch_is_loto, :] = 1.0 / max(tau_LTO_ps * 1e-12, 1e-30)
    Gamma += A_imp * w**4 + B_imp * w**2 + C_imp
    if bool(get_or(get_or(opts, "scatter", {}), "pb_on", False)):
        B, Nw = w.shape
        branch_grid = np.repeat(np.arange(1, B + 1, dtype=np.int64), Nw)
        q_flat, vabs_flat = q_vabs_from_w_table(spec, w.reshape(-1), branch_grid)
        q_grid = q_flat.reshape(B, Nw)
        vabs_grid = vabs_flat.reshape(B, Nw)
        Tsi = float(get_or(opts, "PB_Tsi", 0.0))
        if Tsi > 0.0:
            delta = float(get_or(opts, "PB_Delta", 0.0))
            cos2 = 1.0 / 3.0
            p_spec = np.exp(-4.0 * (np.maximum(q_grid, 0.0) * delta) ** 2 * cos2)
            ff = (1.0 - p_spec) / (1.0 + p_spec)
            Gamma += vabs_grid / max(Tsi, 1e-12) * ff
        else:
            bulk_L = float(get_or(opts, "PB_bulk_L", 0.0))
            bulk_F = float(get_or(opts, "PB_bulk_F", 0.0))
            if bulk_L > 0.0:
                Gamma += vabs_grid / max(bulk_L * bulk_F, 1e-12)
    return np.maximum(np.nan_to_num(Gamma, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def build_pp_scattering_T_lookup(spec: dict[str, Any], opts: dict[str, Any], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or {}
    T_min = float(get_or(cfg, "T_min", 1.0))
    T_max = float(get_or(cfg, "T_max", 2000.0))
    nT = int(get_or(cfg, "nT", 2001))
    T = np.linspace(T_min, T_max, nT, dtype=np.float64)
    DOS = np.maximum(np.asarray(spec["DOS_w_b"], dtype=np.float64), 0.0)
    w = np.maximum(np.asarray(spec["w_mid"], dtype=np.float64), 0.0)
    dw = ensure_2d_dw(spec)
    S = np.zeros(T.size, dtype=np.float64)
    Sb = np.zeros((T.size, DOS.shape[0]), dtype=np.float64)
    for i, Ti in enumerate(T):
        Gamma = scattering_rate_table_at_T(spec, float(Ti), opts, None)
        nbe = bose_occupation(w, float(Ti))
        dS = DOS * (HBAR * w) * Gamma * nbe * dw
        Sb[i, :] = dS.sum(axis=1)
        S[i] = Sb[i, :].sum()
    S_mono = np.maximum.accumulate(S)
    keep = np.ones(S_mono.size, dtype=bool)
    keep[1:] = np.diff(S_mono) > 0
    if np.count_nonzero(keep) < 2:
        keep[:] = True
    S_unique = S_mono[keep]
    T_unique = T[keep]
    inv_interp = PchipInterpolator(S_unique, T_unique, extrapolate=True)
    S_interp = PchipInterpolator(T, S, extrapolate=True)
    return {
        "T": T,
        "S": S,
        "Sb": Sb,
        "S_mono": S_unique,
        "T_mono": T_unique,
        "inv_interp": inv_interp,
        "S_interp": S_interp,
        "inv": lambda Starget: np.asarray(inv_interp(np.clip(np.asarray(Starget, dtype=np.float64), S_unique[0], S_unique[-1]))),
    }


def infer_Nc(mesh: dict[str, Any]) -> int:
    if mesh.get("Nc") is not None:
        return int(mesh["Nc"])
    if all(k in mesh for k in ("Nx", "Ny", "Nz")):
        return int(mesh["Nx"] * mesh["Ny"] * mesh["Nz"])
    if mesh.get("boxes") is not None:
        return int(np.asarray(mesh["boxes"]).shape[0])
    raise ValueError("cannot infer Nc from mesh")


def cell_volumes(mesh: dict[str, Any]) -> np.ndarray:
    if "cell_vol" in mesh and mesh["cell_vol"] is not None:
        return np.asarray(mesh["cell_vol"], dtype=np.float64).reshape(-1)
    if all(k in mesh for k in ("x_edges", "y_edges", "z_edges", "Nx", "Ny", "Nz")):
        dx = np.diff(np.asarray(mesh["x_edges"], dtype=np.float64))
        dy = np.diff(np.asarray(mesh["y_edges"], dtype=np.float64))
        dz = np.diff(np.asarray(mesh["z_edges"], dtype=np.float64))
        DX, DY, DZ = np.meshgrid(dx, dy, dz, indexing="ij")
        return (DX * DY * DZ).ravel(order="F")
    if "boxes" in mesh and mesh["boxes"] is not None:
        boxes = np.asarray(mesh["boxes"], dtype=np.float64)
        return (boxes[:, 1] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 2]) * (boxes[:, 5] - boxes[:, 4])
    raise ValueError("mesh lacks cell volume information")


def enhance_factor_array(opts: dict[str, Any], Nc: int) -> np.ndarray:
    af = get_or(opts, "enhance_factor", 1.0)
    if np.isscalar(af):
        return np.full(Nc, float(af), dtype=np.float64)
    af_arr = np.asarray(af, dtype=np.float64).reshape(-1)
    if af_arr.size != Nc:
        raise ValueError("enhance_factor must be scalar or Nc-by-1")
    return af_arr


def q_vabs_from_w_table(spec: dict[str, Any], w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.int64).reshape(-1)
    q = np.zeros_like(w)
    vabs = np.zeros_like(w)
    if w.size == 0:
        return q, vabs
    for branch in np.unique(b):
        mask = b == branch
        lookup = spec["branch_lookups"][int(branch) - 1]
        w_branch = np.clip(w[mask], lookup["ws"][0], lookup["ws"][-1])
        q_branch = np.asarray(lookup["w_to_q"](w_branch), dtype=np.float64)
        q_branch = np.clip(q_branch, lookup["qs"][0], lookup["qs"][-1])
        v_branch = np.asarray(lookup["q_to_v"](q_branch), dtype=np.float64)
        q[mask] = q_branch
        vabs[mask] = np.abs(v_branch)
    return q, vabs


def uniform_positions_in_cells(mesh: dict[str, Any], cell_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell_ids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
    if cell_ids.size == 0:
        z = np.zeros(0, dtype=np.float64)
        return z.copy(), z.copy(), z.copy()
    epsl = 1e-12
    if all(k in mesh for k in ("Nx", "Ny", "Nz", "x_edges", "y_edges", "z_edges")):
        ix, iy, iz = ind2sub(mesh["Nx"], mesh["Ny"], mesh["Nz"], cell_ids)
        X = np.asarray(mesh["x_edges"], dtype=np.float64)
        Y = np.asarray(mesh["y_edges"], dtype=np.float64)
        Z = np.asarray(mesh["z_edges"], dtype=np.float64)
        rx = np.random.random(cell_ids.size)
        ry = np.random.random(cell_ids.size)
        rz = np.random.random(cell_ids.size)
        x = X[ix - 1] + (X[ix] - X[ix - 1]) * rx
        y = Y[iy - 1] + (Y[iy] - Y[iy - 1]) * ry
        z = Z[iz - 1] + (Z[iz] - Z[iz - 1]) * rz
        x = np.clip(x, X[ix - 1] + epsl, X[ix] - epsl)
        y = np.clip(y, Y[iy - 1] + epsl, Y[iy] - epsl)
        z = np.clip(z, Z[iz - 1] + epsl, Z[iz] - epsl)
        return x, y, z
    if "boxes" in mesh:
        boxes = np.asarray(mesh["boxes"], dtype=np.float64)[cell_ids - 1]
        rnd = np.random.random((cell_ids.size, 3))
        x = boxes[:, 0] + (boxes[:, 1] - boxes[:, 0]) * rnd[:, 0]
        y = boxes[:, 2] + (boxes[:, 3] - boxes[:, 2]) * rnd[:, 1]
        z = boxes[:, 4] + (boxes[:, 5] - boxes[:, 4]) * rnd[:, 2]
        x = np.clip(x, boxes[:, 0] + epsl, boxes[:, 1] - epsl)
        y = np.clip(y, boxes[:, 2] + epsl, boxes[:, 3] - epsl)
        z = np.clip(z, boxes[:, 4] + epsl, boxes[:, 5] - epsl)
        return x, y, z
    raise ValueError("mesh does not provide usable cell geometry")


def load_initial_temperature_field(mesh: dict[str, Any], opts: dict[str, Any], default_T: float | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    Nc = infer_Nc(mesh)
    if default_T is None or not np.isfinite(default_T):
        Tcell = np.full(Nc, np.nan, dtype=np.float64)
    else:
        Tcell = np.full(Nc, float(default_T), dtype=np.float64)
    meta = {"source": "", "used_file": False, "T_min": np.nan, "T_mean": np.nan, "T_max": np.nan}
    filepath = get_or(opts, "initial_temperature_file", "")
    if not filepath:
        if np.any(~np.isfinite(Tcell)):
            raise ValueError("missing initial_temperature_file and no default_T provided")
        meta["T_min"] = float(Tcell.min())
        meta["T_mean"] = float(Tcell.mean())
        meta["T_max"] = float(Tcell.max())
        return Tcell, meta
    data = read_numeric_matrix(filepath, delimiter=",")
    if data.shape[1] < 4:
        raise ValueError(f"{filepath} must have four columns")
    idx = np.round(data[:, 0]).astype(np.int64)
    idy = np.round(data[:, 1]).astype(np.int64)
    idz = np.round(data[:, 2]).astype(np.int64)
    temp = data[:, 3].astype(np.float64)
    if np.any((idx < 1) | (idx > mesh["Nx"]) | (idy < 1) | (idy > mesh["Ny"]) | (idz < 1) | (idz > mesh["Nz"])):
        raise ValueError(f"indices in {filepath} exceed mesh bounds")
    lin = sub2ind(mesh["Nx"], mesh["Ny"], mesh["Nz"], idx, idy, idz)
    if np.unique(lin).size != lin.size:
        raise ValueError(f"duplicate cell indices found in {filepath}")
    Tcell[lin - 1] = temp
    if np.any(~np.isfinite(Tcell)):
        raise ValueError(f"{filepath} does not cover all mesh cells")
    meta["source"] = str(as_path(filepath))
    meta["used_file"] = True
    meta["T_min"] = float(Tcell.min())
    meta["T_mean"] = float(Tcell.mean())
    meta["T_max"] = float(Tcell.max())
    return Tcell, meta


def load_reference_temperature_field(mesh: dict[str, Any], opts: dict[str, Any], default_Tref: float | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    Nc = infer_Nc(mesh)
    if default_Tref is None or not np.isfinite(default_Tref):
        Tref = np.full(Nc, np.nan, dtype=np.float64)
    else:
        Tref = np.full(Nc, float(default_Tref), dtype=np.float64)
    meta = {"source": "", "used_file": False, "T_min": np.nan, "T_mean": np.nan, "T_max": np.nan}
    filepath = get_or(opts, "reference_temperature_file", "")
    if not filepath:
        if np.any(~np.isfinite(Tref)):
            raise ValueError("missing reference_temperature_file and no default_Tref provided")
        meta["T_min"] = float(Tref.min())
        meta["T_mean"] = float(Tref.mean())
        meta["T_max"] = float(Tref.max())
        return Tref, meta
    data = read_numeric_matrix(filepath, delimiter=",")
    if data.shape[1] < 4:
        raise ValueError(f"{filepath} must have four columns")
    idx = np.round(data[:, 0]).astype(np.int64)
    idy = np.round(data[:, 1]).astype(np.int64)
    idz = np.round(data[:, 2]).astype(np.int64)
    tref = data[:, 3].astype(np.float64)
    if idx.size == 0:
        raise ValueError(f"{filepath} contains no valid numeric rows")
    if np.any((idx < 1) | (idx > mesh["Nx"]) | (idy < 1) | (idy > mesh["Ny"]) | (idz < 1) | (idz > mesh["Nz"])):
        raise ValueError(f"indices in {filepath} exceed mesh bounds")
    lin = sub2ind(mesh["Nx"], mesh["Ny"], mesh["Nz"], idx, idy, idz)
    if np.unique(lin).size != lin.size:
        raise ValueError(f"duplicate cell indices found in {filepath}")
    Tref[lin - 1] = tref
    if np.any(~np.isfinite(Tref)):
        raise ValueError(f"{filepath} does not cover all mesh cells")
    meta["source"] = str(as_path(filepath))
    meta["used_file"] = True
    meta["T_min"] = float(Tref.min())
    meta["T_mean"] = float(Tref.mean())
    meta["T_max"] = float(Tref.max())
    return Tref, meta


def load_volume_heat_source_field(mesh: dict[str, Any], opts: dict[str, Any], default_qvol: float | np.ndarray = 0.0) -> tuple[np.ndarray, dict[str, Any]]:
    Nc = infer_Nc(mesh)
    Vc = cell_volumes(mesh)
    if np.isscalar(default_qvol):
        qvol = np.full(Nc, float(default_qvol), dtype=np.float64)
    else:
        qvol = np.asarray(default_qvol, dtype=np.float64).reshape(-1)
        if qvol.size != Nc:
            raise ValueError("default_qvol must be scalar or Nc-by-1")
    meta = {
        "source": "",
        "used_file": False,
        "format": "",
        "n_entries": 0,
        "n_nonzero_cells": int(np.count_nonzero(qvol)),
        "q_min": float(qvol.min()),
        "q_mean": float(qvol.mean()),
        "q_max": float(qvol.max()),
        "q_total_W": float(np.sum(qvol * Vc)),
    }
    filepath = get_or(opts, "volume_heat_source_file", "")
    if not filepath:
        return qvol, meta
    rows: list[list[float]] = []
    for iline, line in enumerate(read_clean_lines(filepath), start=1):
        tokens = [tok for tok in re.split(r"[\s,]+", line) if tok]
        try:
            values = [float(tok) for tok in tokens]
        except ValueError as exc:
            raise ValueError(f"non-numeric value found in {filepath} line {iline}: {line}") from exc
        rows.append(values)
    if not rows:
        return qvol, meta
    widths = {len(row) for row in rows}
    if len(widths) != 1 or next(iter(widths)) not in {4, 7}:
        raise ValueError(
            f"{filepath} must use either 4 columns (idx,idy,idz,qvol) or 7 columns "
            "(Xmin,Xmax,Ymin,Ymax,Zmin,Zmax,P_W_m3)"
        )
    ncol = next(iter(widths))
    if ncol == 4:
        data = np.asarray(rows, dtype=np.float64)
        idx = np.round(data[:, 0]).astype(np.int64)
        idy = np.round(data[:, 1]).astype(np.int64)
        idz = np.round(data[:, 2]).astype(np.int64)
        qsrc = data[:, 3].astype(np.float64)
        if np.any((idx < 1) | (idx > mesh["Nx"]) | (idy < 1) | (idy > mesh["Ny"]) | (idz < 1) | (idz > mesh["Nz"])):
            raise ValueError(f"indices in {filepath} exceed mesh bounds")
        lin = sub2ind(mesh["Nx"], mesh["Ny"], mesh["Nz"], idx, idy, idz).astype(np.int64)
        np.add.at(qvol, lin - 1, qsrc)
        meta["format"] = "cell_list"
        meta["n_entries"] = int(data.shape[0])
    else:
        X = np.asarray(mesh["x_edges"], dtype=np.float64)
        Y = np.asarray(mesh["y_edges"], dtype=np.float64)
        Z = np.asarray(mesh["z_edges"], dtype=np.float64)
        length_scale = float(get_or(opts, "volume_heat_source_length_scale", 1e-6))
        domain = np.array([X[0], X[-1], Y[0], Y[-1], Z[0], Z[-1]], dtype=np.float64)
        tol = 1e-12 * max(1.0, float(np.max(np.abs(domain))))
        for iline, row in enumerate(rows, start=1):
            bounds_input = np.asarray(row[:6], dtype=np.float64)
            qsrc = float(row[6])
            if np.any(~np.isfinite(bounds_input)) or not np.isfinite(qsrc):
                raise ValueError(f"invalid region source row in {filepath} line {iline}")
            if np.any(bounds_input[1::2] < bounds_input[0::2]):
                raise ValueError(f"volume heat source bounds must satisfy min<=max in {filepath} line {iline}")
            bounds = bounds_input * length_scale
            if (
                bounds[0] < domain[0] - tol
                or bounds[1] > domain[1] + tol
                or bounds[2] < domain[2] - tol
                or bounds[3] > domain[3] + tol
                or bounds[4] < domain[4] - tol
                or bounds[5] > domain[5] + tol
            ):
                raise ValueError(f"volume heat source region exceeds mesh bounds in {filepath} line {iline}")
            bounds[0] = max(bounds[0], domain[0])
            bounds[1] = min(bounds[1], domain[1])
            bounds[2] = max(bounds[2], domain[2])
            bounds[3] = min(bounds[3], domain[3])
            bounds[4] = max(bounds[4], domain[4])
            bounds[5] = min(bounds[5], domain[5])
            ox = np.maximum(np.minimum(X[1:], bounds[1]) - np.maximum(X[:-1], bounds[0]), 0.0)
            oy = np.maximum(np.minimum(Y[1:], bounds[3]) - np.maximum(Y[:-1], bounds[2]), 0.0)
            oz = np.maximum(np.minimum(Z[1:], bounds[5]) - np.maximum(Z[:-1], bounds[4]), 0.0)
            ix = np.flatnonzero(ox > 0.0)
            iy = np.flatnonzero(oy > 0.0)
            iz = np.flatnonzero(oz > 0.0)
            if ix.size == 0 or iy.size == 0 or iz.size == 0:
                continue
            for kz in iz:
                for ky in iy:
                    for kx in ix:
                        cid = int(sub2ind(mesh["Nx"], mesh["Ny"], mesh["Nz"], kx + 1, ky + 1, kz + 1))
                        overlap = float(ox[kx] * oy[ky] * oz[kz])
                        if overlap <= 0.0:
                            continue
                        qvol[cid - 1] += qsrc * overlap / max(Vc[cid - 1], REALMIN)
        meta["format"] = "box_regions"
        meta["n_entries"] = len(rows)
    meta["source"] = str(as_path(filepath))
    meta["used_file"] = True
    meta["n_nonzero_cells"] = int(np.count_nonzero(qvol))
    meta["q_min"] = float(qvol.min())
    meta["q_mean"] = float(qvol.mean())
    meta["q_max"] = float(qvol.max())
    meta["q_total_W"] = float(np.sum(qvol * Vc))
    return qvol, meta


def monitor_plane(bounds: np.ndarray) -> tuple[str, float, float]:
    tol = 1e-15 * max(1.0, float(np.max(np.abs(bounds))))
    fixed = np.array([abs(bounds[1] - bounds[0]) <= tol, abs(bounds[3] - bounds[2]) <= tol, abs(bounds[5] - bounds[4]) <= tol])
    if np.count_nonzero(fixed) != 1:
        raise ValueError("each monitor must define exactly one plane coordinate")
    if fixed[0]:
        return "x", float(0.5 * (bounds[0] + bounds[1])), float(max(bounds[3] - bounds[2], 0.0) * max(bounds[5] - bounds[4], 0.0))
    if fixed[1]:
        return "y", float(0.5 * (bounds[2] + bounds[3])), float(max(bounds[1] - bounds[0], 0.0) * max(bounds[5] - bounds[4], 0.0))
    return "z", float(0.5 * (bounds[4] + bounds[5])), float(max(bounds[1] - bounds[0], 0.0) * max(bounds[3] - bounds[2], 0.0))


def load_heat_flux_monitors(mesh: dict[str, Any], output_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    filepath = output_cfg.get("heat_flux_monitor_file", "")
    warnings: list[str] = []
    monitors: list[dict[str, Any]] = []
    if not filepath:
        return monitors, warnings
    path = Path(filepath)
    if not path.is_file():
        warnings.append(f"heat flux monitor file not found: {filepath}")
        return monitors, warnings
    length_scale = float(get_or(output_cfg, "monitor_length_scale", 1.0))
    for i, line in enumerate(read_clean_lines(path), start=1):
        tokens = [tok for tok in re.split(r"[\s,]+", line) if tok]
        if len(tokens) < 7:
            raise ValueError(f"invalid monitor line: {line}")
        bounds_input = np.asarray([float(v) for v in tokens[:6]], dtype=np.float64)
        requested_direction = tokens[6].upper()
        label = tokens[7] if len(tokens) >= 8 else f"monitor_{i:03d}"
        bounds = bounds_input * length_scale
        axis, coord, area = monitor_plane(bounds)
        sign_char = "-" if requested_direction.startswith("-") else "+"
        effective_normal = f"{sign_char}{axis.upper()}"
        warning_msg = ""
        if len(requested_direction) >= 2 and requested_direction[-1].upper() != axis.upper():
            warning_msg = (
                f'monitor "{label}": requested direction {requested_direction} does not match plane normal axis '
                f'{axis.upper()}; using {sign_char}{axis.upper()}.'
            )
            warnings.append(warning_msg)
        domain_box = np.asarray(mesh.get("domain_box", np.zeros(6)), dtype=np.float64)
        if domain_box.size == 6:
            inside = (
                bounds[0] >= domain_box[0]
                and bounds[1] <= domain_box[1]
                and bounds[2] >= domain_box[2]
                and bounds[3] <= domain_box[3]
                and bounds[4] >= domain_box[4]
                and bounds[5] <= domain_box[5]
            )
            if not inside:
                extra = f'monitor "{label}" extends outside domain bounds.'
                warnings.append(extra)
                warning_msg = extra if not warning_msg else f"{warning_msg} | {extra}"
        monitors.append(
            {
                "id": i,
                "label": label,
                "bounds_input": bounds_input,
                "bounds": bounds,
                "requested_direction": requested_direction,
                "axis": axis,
                "coord": coord,
                "area": area,
                "effective_normal": effective_normal,
                "warning": warning_msg,
                "raw": line,
            }
        )
    return monitors, warnings


def sample_particles_for_cells(
    mesh: dict[str, Any],
    spec: dict[str, Any],
    opts: dict[str, Any],
    cell_ids: np.ndarray,
    Tcell: np.ndarray,
    Tref_cell: np.ndarray | None = None,
    id_start: int = 0,
) -> tuple[ParticleBlock, dict[str, Any]]:
    cell_ids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
    Tcell = np.asarray(Tcell, dtype=np.float64).reshape(-1)
    Tref_cell = np.zeros(0, dtype=np.float64) if Tref_cell is None else np.asarray(Tref_cell, dtype=np.float64).reshape(-1)
    info = {
        "cell_ids": cell_ids.copy(),
        "Nexp_tot": 0.0,
        "Nsp_tot": 0,
        "E_eff_used": float(get_or(opts, "E_eff", 1e-18)),
        "fixed_target_particles": 0,
        "target_temperature": Tcell.copy(),
        "reference_temperature": Tref_cell.copy(),
    }
    if cell_ids.size == 0:
        return ParticleBlock.empty(), info
    mode_name = str(get_or(opts, "mode", "absolute"))
    if Tcell.size != cell_ids.size:
        raise ValueError("Tcell size must match cell_ids")
    if mode_name.lower() == "deviational" and Tref_cell.size != cell_ids.size:
        raise ValueError("Tref_cell size must match cell_ids in deviational mode")
    E_eff = float(get_or(opts, "E_eff", 1e-18))
    initial_particles_fixed = max(0, int(get_or(opts, "initial_particles_fixed", 0)))
    use_fixed_initial_particles = bool(id_start == 0 and initial_particles_fixed > 0)
    use_bin_center_w = bool(get_or(opts, "use_bin_center_w", True))
    max_particles = int(get_or(opts, "max_particles", np.iinfo(np.int64).max))
    Vc_all = cell_volumes(mesh)
    af_all = enhance_factor_array(opts, Vc_all.size)
    Vc = Vc_all[cell_ids - 1]
    af = af_all[cell_ids - 1]
    B, Nw = spec["w_mid"].shape
    dw = ensure_2d_dw(spec)
    pref = HBAR * np.asarray(spec["w_mid"], dtype=np.float64) * np.maximum(np.asarray(spec["DOS_w_b"], dtype=np.float64), 0.0) * dw
    nloc = cell_ids.size
    mode_weight = np.zeros((B * Nw, nloc), dtype=np.float64)
    mode_sign = np.ones((B * Nw, nloc), dtype=np.int8)
    U_density_cell = np.zeros(nloc, dtype=np.float64)
    for i in range(nloc):
        n_cell = bose_occupation(spec["w_mid"], Tcell[i])
        if mode_name.lower() == "deviational":
            n_ref = bose_occupation(spec["w_mid"], Tref_cell[i])
            Wbm = pref * (n_cell - n_ref)
        else:
            Wbm = pref * n_cell
        mode_weight[:, i] = np.abs(Wbm).ravel(order="F")
        if mode_name.lower() == "deviational":
            s = np.sign(Wbm).ravel(order="F")
            s[s == 0] = 1
            mode_sign[:, i] = s.astype(np.int8)
        U_density_cell[i] = float(Wbm.sum())
    mode_energy_abs = mode_weight.sum(axis=0)
    cell_weight = mode_energy_abs * Vc * af
    total_weight = float(cell_weight.sum())
    if total_weight <= 0:
        return ParticleBlock.empty(), info
    if use_fixed_initial_particles:
        Nsp_tot = int(initial_particles_fixed)
        if Nsp_tot <= 0:
            return ParticleBlock.empty(), info
        E_eff = total_weight / max(float(Nsp_tot), REALMIN)
        Nexp_tot = float(Nsp_tot)
        info["fixed_target_particles"] = int(Nsp_tot)
    else:
        Nexp_tot = total_weight / E_eff
        Nsp_tot = int(np.floor(Nexp_tot) + (np.random.random() < (Nexp_tot - np.floor(Nexp_tot))))
    info["Nexp_tot"] = float(Nexp_tot)
    info["Nsp_tot"] = int(Nsp_tot)
    info["E_eff_used"] = float(E_eff)
    if Nsp_tot <= 0:
        return ParticleBlock.empty(), info
    if Nsp_tot > max_particles:
        raise ValueError(f"expected particles {Nsp_tot:g} exceed max_particles {max_particles:g}")
    cdf_cell = np.cumsum(cell_weight) / total_weight
    cell_pick = np.searchsorted(cdf_cell, np.random.random(Nsp_tot), side="left")
    cell_pick = np.clip(cell_pick, 0, nloc - 1)
    idx_pick = np.zeros(Nsp_tot, dtype=np.int64)
    mode_cdf = np.zeros_like(mode_weight)
    positive = mode_energy_abs > 0
    mode_cdf[:, positive] = np.cumsum(mode_weight[:, positive], axis=0) / mode_energy_abs[positive]
    for iloc in np.unique(cell_pick):
        mask = cell_pick == iloc
        draws = np.random.random(np.count_nonzero(mask))
        idx_pick[mask] = np.searchsorted(mode_cdf[:, iloc], draws, side="left")
    idx_pick = np.clip(idx_pick, 0, B * Nw - 1)
    b = (idx_pick % B + 1).astype(np.int32)
    m = (idx_pick // B + 1).astype(np.int32)
    if use_bin_center_w or "w_edges" not in spec or len(spec["w_edges"]) != Nw + 1:
        w = np.asarray(spec["w_mid"], dtype=np.float64)[b - 1, m - 1]
    else:
        w_lo = np.asarray(spec["w_edges"], dtype=np.float64)[m - 1]
        w_hi = np.asarray(spec["w_edges"], dtype=np.float64)[m]
        w = w_lo + (w_hi - w_lo) * np.random.random(Nsp_tot)
    q, vabs = q_vabs_from_w_table(spec, w, b)
    dirs = rand_unit_vec_batch(Nsp_tot)
    E = np.full(Nsp_tot, E_eff, dtype=np.float64)
    sgn = np.ones(Nsp_tot, dtype=np.int8)
    if mode_name.lower() == "deviational":
        sgn = mode_sign[idx_pick, cell_pick].astype(np.int8)
        E = E_eff * sgn.astype(np.float64)
    sampled_cells = cell_ids[cell_pick].astype(np.int32)
    # Compute per-particle material_id (0-based) from the mesh material index.
    cell_mat_idx = np.asarray(mesh.get("cell_material_index", np.zeros(1, dtype=np.int32)), dtype=np.int32)
    mat_id = np.zeros(Nsp_tot, dtype=np.int32)
    valid_cells = (sampled_cells >= 1) & (sampled_cells <= cell_mat_idx.size)
    mat_id[valid_cells] = cell_mat_idx[sampled_cells[valid_cells] - 1] - 1  # 1-based -> 0-based
    mat_id[~valid_cells] = -1
    x, y, z = uniform_positions_in_cells(mesh, sampled_cells)
    ids = np.arange(id_start + 1, id_start + Nsp_tot + 1, dtype=np.int64)
    p = ParticleBlock(
        id=ids,
        par_id=ids.copy(),
        cell=sampled_cells,
        material_id=mat_id,
        x=x,
        y=y,
        z=z,
        b=b,
        m=m,
        w=w,
        q=q,
        vx=vabs * dirs[:, 0],
        vy=vabs * dirs[:, 1],
        vz=vabs * dirs[:, 2],
        vabs=vabs,
        E=E,
        sgn=sgn,
        n_ph=E / (HBAR * np.maximum(w, 1e-30)),
        seed=np.random.randint(1, 2**31 - 1, size=Nsp_tot, dtype=np.int64),
        t_left=np.zeros(Nsp_tot, dtype=np.float64),
    )
    return p, info


def init_state_energy(mesh: dict[str, Any], spec: dict[str, Any], opts: dict[str, Any]) -> SimulationState:
    mode_name = str(get_or(opts, "mode", "absolute"))
    default_T = float(opts["T_init"]) if np.isfinite(get_or(opts, "T_init", np.nan)) else float(get_or(opts, "T0", get_or(opts, "Tref", 300.0)))
    Tcell, Tmeta = load_initial_temperature_field(mesh, opts, default_T)
    default_Tref = float(get_or(opts, "Tref", np.nan))
    if not np.isfinite(default_Tref):
        default_Tref = float(np.mean(Tcell))
    Tref_cell, Tref_meta = load_reference_temperature_field(mesh, opts, default_Tref)
    cell_ids = np.arange(1, infer_Nc(mesh) + 1, dtype=np.int64)
    p, sample_info = sample_particles_for_cells(mesh, spec, opts, cell_ids, Tcell, Tref_cell, 0)
    Nsp_cell = np.bincount(p.cell.astype(np.int64) - 1, minlength=infer_Nc(mesh)).astype(np.int64) if len(p) else np.zeros(infer_Nc(mesh), dtype=np.int64)
    Vc = cell_volumes(mesh)
    Vdom = float(Vc.sum())
    info = {
        "mode": mode_name,
        "Tref": float(get_or(opts, "Tref", get_or(opts, "T0", 300.0))),
        "Tref_cell": Tref_cell,
        "reference_temperature_meta": Tref_meta,
        "T_init_cell": Tcell,
        "initial_temperature_meta": Tmeta,
        "U_density_mean": sample_info["Nexp_tot"] * float(sample_info.get("E_eff_used", get_or(opts, "E_eff", 1e-18))) / max(Vdom, REALMIN),
        "U_total": sample_info["Nexp_tot"] * float(sample_info.get("E_eff_used", get_or(opts, "E_eff", 1e-18))),
        "Nexp_tot": sample_info["Nexp_tot"],
        "Nsp_tot": sample_info["Nsp_tot"],
        "E_eff_used": float(sample_info.get("E_eff_used", get_or(opts, "E_eff", 1e-18))),
        "fixed_target_particles": int(sample_info.get("fixed_target_particles", 0)),
        "Nc": infer_Nc(mesh),
        "Vdom": Vdom,
    }
    return SimulationState(
        p=p,
        WE=float(sample_info.get("E_eff_used", get_or(opts, "E_eff", 1e-18))),
        Wp=float(sample_info.get("E_eff_used", get_or(opts, "E_eff", 1e-18))),
        Nsp_cell=Nsp_cell,
        enhance_factor=enhance_factor_array(opts, infer_Nc(mesh)),
        info=info,
    )


def initial_temperature_from_state_or_file(state: SimulationState, mesh: dict[str, Any], opts: dict[str, Any], default_T: float) -> tuple[np.ndarray, dict[str, Any]]:
    Nc = infer_Nc(mesh)
    T_init = np.asarray(state.info.get("T_init_cell", np.zeros(0)), dtype=np.float64).reshape(-1)
    if T_init.size == Nc:
        meta = dict(state.info.get("initial_temperature_meta", {}))
        meta["T_min"] = float(T_init.min())
        meta["T_mean"] = float(T_init.mean())
        meta["T_max"] = float(T_init.max())
        return T_init.copy(), meta
    return load_initial_temperature_field(mesh, opts, default_T)


def reference_temperature_from_state(state: SimulationState, opts: dict[str, Any], Nc: int) -> np.ndarray:
    Tref = np.asarray(state.info.get("Tref_cell", np.zeros(0)), dtype=np.float64).reshape(-1)
    if Tref.size == Nc:
        return Tref.copy()
    opts_Tref = np.asarray(get_or(opts, "Tref_cell", np.zeros(0)), dtype=np.float64).reshape(-1)
    if opts_Tref.size == Nc:
        return opts_Tref.copy()
    if np.isfinite(get_or(opts, "Tref", np.nan)):
        return np.full(Nc, float(opts["Tref"]), dtype=np.float64)
    raise ValueError("deviational mode requires Tref_cell or opts.Tref")


def update_temperature_from_energy(
    state: SimulationState,
    mesh: dict[str, Any],
    spec: dict[str, Any] | list[dict[str, Any]],
    opts: dict[str, Any],
    lut: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    Vc = cell_volumes(mesh)
    Nc = Vc.size
    if len(state.p) == 0:
        Ecell = np.zeros(Nc, dtype=np.float64)
    else:
        valid = (state.p.cell >= 1) & (state.p.cell <= Nc)
        Ecell = np.bincount(state.p.cell[valid].astype(np.int64) - 1, weights=state.p.E[valid], minlength=Nc).astype(np.float64)
    Ulocal = np.zeros(Nc, dtype=np.float64)
    mask = Vc > 0
    Ulocal[mask] = Ecell[mask] / Vc[mask]
    # Build or normalise LUTs.
    if isinstance(lut, dict) or lut is None:
        _luts: list[dict[str, Any]] = [lut] if lut is not None else [build_E_T_lookup(spec if isinstance(spec, dict) else spec[0], et_lookup_cfg_from_opts(opts))]
    else:
        _luts = list(lut)
    cell_mat = np.asarray(mesh.get("cell_material_index", np.ones(Nc, dtype=np.int32)), dtype=np.int32)
    Tnew = np.zeros(Nc, dtype=np.float64)
    if str(get_or(opts, "mode", "absolute")).lower() == "deviational":
        Tref_cell = reference_temperature_from_state(state, opts, Nc)
    else:
        Tref_cell = np.zeros(Nc, dtype=np.float64)
    for mi, lu in enumerate(_luts):
        cell_mask = (cell_mat == (mi + 1)) & mask
        if not np.any(cell_mask):
            continue
        Ul = Ulocal[cell_mask]
        if str(get_or(opts, "mode", "absolute")).lower() == "deviational":
            Tref = np.clip(Tref_cell[cell_mask], lu["T"][0], lu["T"][-1])
            Uref = np.asarray(lu["U_interp"](Tref), dtype=np.float64)
            Uabs = Ul + Uref
        else:
            Uabs = Ul
        Umin, Umax = float(lu["U_mono"][0]), float(lu["U_mono"][-1])
        Ucl = np.clip(Uabs, Umin, Umax)
        Tnew[cell_mask] = np.asarray(lu["inv_interp"](Ucl), dtype=np.float64)
    Uabs_all = Ulocal.copy()
    if str(get_or(opts, "mode", "absolute")).lower() == "deviational":
        for mi, lu in enumerate(_luts):
            cell_mask = cell_mat == (mi + 1)
            if not np.any(cell_mask):
                continue
            Tref = np.clip(Tref_cell[cell_mask], lu["T"][0], lu["T"][-1])
            Uabs_all[cell_mask] = Ulocal[cell_mask] + np.asarray(lu["U_interp"](Tref), dtype=np.float64)
    Tmin = float(np.min(Tnew)) if Tnew.size else np.nan
    Tmean = float(np.mean(Tnew)) if Tnew.size else np.nan
    Tmax = float(np.max(Tnew)) if Tnew.size else np.nan
    lut0 = _luts[0]
    aux = {
        "Ecell": Ecell,
        "Vcell": Vc,
        "Ulocal": Ulocal,
        "Uabs": Uabs_all,
        "clip_low": 0.0,
        "clip_high": 0.0,
        "clip_low_count": 0,
        "clip_high_count": 0,
        "Tmin": Tmin,
        "Tmean": Tmean,
        "Tmax": Tmax,
        "Tlut_min": float(lut0["T"][0]),
        "Tlut_max": float(lut0["T"][-1]),
        "LUT": lut0,
    }
    return Tnew, aux


def active_total_scattering_rate_for_particles(opts: dict[str, Any], r_tau: np.ndarray) -> np.ndarray:
    rates = np.asarray(r_tau, dtype=np.float64)
    if rates.size == 0:
        return np.zeros(rates.shape[0] if rates.ndim else 0, dtype=np.float64)
    r_all = np.maximum(rates, 0.0).copy()
    if r_all.ndim != 2:
        return np.zeros(0, dtype=np.float64)
    if not bool(get_or(get_or(opts, "scatter", {}), "pb_on", False)) and r_all.shape[1] >= 6:
        r_all[:, 5] = 0.0
    return np.sum(r_all, axis=1)


def _multi_branch_is(branch_id: np.ndarray, material_id: np.ndarray, branch_array_2d: np.ndarray) -> np.ndarray:
    """Index into a (n_materials, max_B) branch-type array per particle.

    Returns a bool array of shape (Np,).  Material indices outside [0, n_mats)
    are treated as material 0.
    """
    b = np.asarray(branch_id, dtype=np.int64).reshape(-1) - 1
    mid = np.asarray(material_id, dtype=np.int64).reshape(-1)
    n_mat = branch_array_2d.shape[0]
    mid = np.clip(mid, 0, n_mat - 1)
    # Fast path: single material
    if n_mat == 1:
        arr = np.asarray(branch_array_2d[0], dtype=np.bool_)
        valid = (b >= 0) & (b < arr.size)
        out = np.zeros(b.size, dtype=bool)
        out[valid] = arr[b[valid]]
        return out
    # General path
    max_b = branch_array_2d.shape[1]
    valid = (b >= 0) & (b < max_b)
    out = np.zeros(b.size, dtype=bool)
    flat_idx = mid[valid] * max_b + b[valid]
    flat = branch_array_2d.ravel()
    out[valid] = flat[flat_idx]
    return out


def pp_only_rate_for_particles(
    branch_id: np.ndarray,
    r_tau: np.ndarray,
    spec: dict[str, Any] | list[dict[str, Any]],
    material_id: np.ndarray | None = None,
) -> np.ndarray:
    b = np.asarray(branch_id, dtype=np.int32).reshape(-1)
    rates = np.asarray(r_tau, dtype=np.float64)
    if b.size == 0 or rates.size == 0:
        return np.zeros(b.size, dtype=np.float64)
    rLA, rTAN, rTAU, rLTO = [np.maximum(rates[:, i], 0.0) for i in range(4)]
    rPP = np.zeros(b.size, dtype=np.float64)
    if material_id is None:
        material_id = np.zeros(b.size, dtype=np.int32)
    # Build padded 2D branch arrays.
    if isinstance(spec, dict):
        specs = [spec]
    else:
        specs = list(spec)
    n_mat = len(specs)
    max_B = max(int(s.get("B", 0)) for s in specs)
    b_la = np.zeros((n_mat, max_B), dtype=np.bool_)
    b_ta = np.zeros((n_mat, max_B), dtype=np.bool_)
    b_loto = np.zeros((n_mat, max_B), dtype=np.bool_)
    for mi, s in enumerate(specs):
        bla = np.asarray(s.get("branch_is_la", np.zeros(0, dtype=np.bool_)), dtype=np.bool_)
        bta = np.asarray(s.get("branch_is_ta", np.zeros(0, dtype=np.bool_)), dtype=np.bool_)
        blo = np.asarray(s.get("branch_is_loto", np.zeros(0, dtype=np.bool_)), dtype=np.bool_)
        Bm = bla.size
        b_la[mi, :Bm] = bla
        b_ta[mi, :Bm] = bta
        b_loto[mi, :Bm] = blo
    is_la = _multi_branch_is(b, material_id, b_la)
    is_ta = _multi_branch_is(b, material_id, b_ta)
    is_loto = _multi_branch_is(b, material_id, b_loto)
    rPP[is_la] = rLA[is_la]
    rPP[is_ta] = rTAN[is_ta] + rTAU[is_ta]
    rPP[is_loto] = rLTO[is_loto]
    other = ~(is_la | is_ta | is_loto)
    if np.any(other):
        # Fall back to the sum of modeled inelastic channels for any branch names
        # that do not match the LA/TA/LO-TO labels.
        rPP[other] = rLA[other] + rTAN[other] + rTAU[other] + rLTO[other]
    return rPP


def update_pp_scattering_temperature_from_energy(
    state: SimulationState,
    mesh: dict[str, Any],
    spec: dict[str, Any] | list[dict[str, Any]],
    opts: dict[str, Any],
    r_tau: np.ndarray,
    lut: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    # --- normalise specs and LUTs ---
    if isinstance(spec, dict):
        _specs_pp: list[dict[str, Any]] = [spec]
    else:
        _specs_pp = list(spec)
    if lut is None:
        _luts_pp: list[dict[str, Any]] = [build_pp_scattering_T_lookup(_specs_pp[0], opts, tloc_lookup_cfg_from_opts(opts))]
    elif isinstance(lut, dict):
        _luts_pp = [lut]
    else:
        _luts_pp = list(lut)
    Vc = cell_volumes(mesh)
    Nc = Vc.size
    if len(state.p) == 0:
        S_cell = np.zeros(Nc, dtype=np.float64)
    else:
        valid = (state.p.cell >= 1) & (state.p.cell <= Nc)
        rTOT = active_total_scattering_rate_for_particles(opts, r_tau)
        S_cell = np.bincount(
            state.p.cell[valid].astype(np.int64) - 1,
            weights=state.p.E[valid] * rTOT[valid],
            minlength=Nc,
        ).astype(np.float64)
    Slocal = np.zeros(Nc, dtype=np.float64)
    mask = Vc > 0
    Slocal[mask] = S_cell[mask] / Vc[mask]
    # --- per-material temperature inversion ---
    cell_mat = np.asarray(mesh.get("cell_material_index", np.ones(Nc, dtype=np.int32)), dtype=np.int32)
    Tnew = np.zeros(Nc, dtype=np.float64)
    single = len(_luts_pp) == 1
    if str(get_or(opts, "mode", "absolute")).lower() == "deviational":
        Tref_cell = reference_temperature_from_state(state, opts, Nc)
    else:
        Tref_cell = np.zeros(Nc, dtype=np.float64)
    for mi, lu in enumerate(_luts_pp):
        cell_mask = (cell_mat == (mi + 1)) if not single else mask
        if not np.any(cell_mask):
            continue
        Sl = Slocal[cell_mask]
        if str(get_or(opts, "mode", "absolute")).lower() == "deviational":
            Tref_clamped = np.clip(Tref_cell[cell_mask], lu["T"][0], lu["T"][-1])
            Sref = np.asarray(lu["S_interp"](Tref_clamped), dtype=np.float64)
            Sabs = Sl + Sref
        else:
            Sabs = Sl
        Smin, Smax = float(lu["S_mono"][0]), float(lu["S_mono"][-1])
        Scl = np.clip(Sabs, Smin, Smax)
        Tnew[cell_mask] = np.asarray(lu["inv_interp"](Scl), dtype=np.float64)
    aux = {
        "S_cell": S_cell,
        "Vcell": Vc,
        "Slocal": Slocal,
        "Sabs": Slocal.copy(),
        "clip_low": 0.0,
        "clip_high": 0.0,
        "LUT": _luts_pp[0],
    }
    # Recompute Sabs for the aux dict (per-material dispatch).
    Sabs_aux = Slocal.copy()
    if str(get_or(opts, "mode", "absolute")).lower() == "deviational":
        for mi, lu in enumerate(_luts_pp):
            cell_mask = (cell_mat == (mi + 1)) if not single else mask
            if not np.any(cell_mask):
                continue
            Tref = np.clip(Tref_cell[cell_mask], lu["T"][0], lu["T"][-1])
            Sabs_aux[cell_mask] = Slocal[cell_mask] + np.asarray(lu["S_interp"](Tref), dtype=np.float64)
    aux["Sabs"] = Sabs_aux
    return Tnew, aux


def prepare_run_output(mesh: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(get_or(opts, "output", {}))
    out_cfg = {
        "enabled": bool(get_or(cfg, "enable", False)),
        "every_n_steps": max(1, int(round(get_or(cfg, "every_n_steps", 100)))),
        "run_dir": "",
        "run_tag": "",
        "inputs_dir": "",
        "steps_dir": "",
        "toc_dir": "",
        "step_history_file": "",
        "run_wallclock_tic": time.perf_counter(),
        "monitors": [],
        "monitor_warnings": [],
        **init_monitor_output_accumulators(0),
        "cum_time": 0.0,
        "interval_time": 0.0,
    }
    if not out_cfg["enabled"]:
        return out_cfg
    root_dir = Path(str(get_or(cfg, "root_dir", "output")))
    run_tag = str(get_or(cfg, "run_tag", time.strftime("%Y%m%d_%H%M%S")))
    run_dir = root_dir / f"run_{run_tag}"
    suffix = 1
    while run_dir.exists():
        run_dir = root_dir / f"run_{run_tag}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = run_dir / "inputs"
    steps_dir = run_dir / "steps"
    toc_dir = run_dir / "Toc"
    inputs_dir.mkdir(exist_ok=True)
    steps_dir.mkdir(exist_ok=True)
    toc_dir.mkdir(exist_ok=True)
    monitors, warnings = load_heat_flux_monitors(mesh, cfg)
    nmon = len(monitors)
    out_cfg.update(
        {
            "run_dir": str(run_dir),
            "run_tag": run_tag,
            "inputs_dir": str(inputs_dir),
            "steps_dir": str(steps_dir),
            "toc_dir": str(toc_dir),
            "step_history_file": str(run_dir / "step_history.txt"),
            "monitors": monitors,
            "monitor_warnings": warnings,
            **init_monitor_output_accumulators(nmon),
        }
    )
    snapshot_specs = [
        ("layout_ldg", mesh.get("layout", {}).get("source", "")),
        ("grid_lgrid", mesh.get("grid", {}).get("source", "")),
        ("initial_temperature", get_or(opts, "initial_temperature_file", "")),
        ("reference_temperature", get_or(opts, "reference_temperature_file", "")),
        ("volume_heat_source", get_or(opts, "volume_heat_source_file", "")),
        ("solver_params", get_or(opts, "solver_param_file", "")),
        ("heat_flux_monitors", get_or(cfg, "heat_flux_monitor_file", "")),
    ]
    manifest = [["kind", "source_path", "snapshot_path"]]
    for kind, src in snapshot_specs:
        if not src:
            continue
        src_path = Path(src)
        if not src_path.is_file():
            continue
        dst = inputs_dir / f"{kind}__{src_path.name}"
        shutil.copyfile(src_path, dst)
        manifest.append([kind, str(src_path.resolve()), str(dst.resolve())])
    write_csv_rows(inputs_dir / "input_manifest.txt", manifest)
    write_csv_rows(
        out_cfg["step_history_file"],
        [[
            "step",
            "dt_s",
            "dt_cfl_s",
            "dt_scat_s",
            "elapsed_time_s",
            "interval_time_s",
            "wall_clock_elapsed_s",
            "Np",
            "T_min_K",
            "T_mean_K",
            "T_max_K",
        ]],
    )
    write_csv_rows(
        run_dir / "run_manifest.txt",
        [
            ["key", "value"],
            ["run_tag", run_tag],
            ["run_dir", str(run_dir.resolve())],
            ["every_n_steps", out_cfg["every_n_steps"]],
            ["toc_dir", str(toc_dir.resolve())],
            ["heat_flux_monitor_file", str(get_or(cfg, "heat_flux_monitor_file", ""))],
            ["monitor_length_scale", get_or(cfg, "monitor_length_scale", np.nan)],
            ["volume_heat_source_length_scale", get_or(opts, "volume_heat_source_length_scale", np.nan)],
            ["created_at", time.strftime("%Y-%m-%d %H:%M:%S")],
        ],
    )
    monitor_rows = [["label", "requested_direction", "effective_normal", "area_m2", "x0_in", "x1_in", "y0_in", "y1_in", "z0_in", "z1_in", "warning"]]
    for mon in monitors:
        b = mon["bounds_input"]
        monitor_rows.append([mon["label"], mon["requested_direction"], mon["effective_normal"], mon["area"], b[0], b[1], b[2], b[3], b[4], b[5], mon["warning"]])
    write_csv_rows(run_dir / "heat_flux_monitors_manifest.txt", monitor_rows)
    if warnings:
        write_csv_rows(run_dir / "heat_flux_monitor_warnings.txt", [[w] for w in warnings])
    return out_cfg


def export_particle_mfp_cdf(
    output_png: str | Path,
    state: SimulationState,
    Tcell: np.ndarray,
    opts: dict[str, Any],
    spec: dict[str, Any],
    x_label: str = "Mean Free Path (nm)",
    plot_title: str = "Particle Mean Free Path CDF",
) -> str:
    output_png = Path(output_png)
    output_csv = output_png.with_suffix(".csv")
    if len(state.p) == 0:
        return ""
    rates = precompute_relax_times(state, np.asarray(Tcell, dtype=np.float64), opts, spec)
    if rates.size == 0:
        return ""
    rate_total = np.sum(np.maximum(rates, 0.0), axis=1)
    vabs = np.asarray(state.p.vabs, dtype=np.float64).reshape(-1)
    n_ph_weight = np.abs(np.asarray(state.p.n_ph, dtype=np.float64).reshape(-1))
    valid = (
        np.isfinite(rate_total)
        & (rate_total > 0.0)
        & np.isfinite(vabs)
        & (vabs > 0.0)
        & np.isfinite(n_ph_weight)
        & (n_ph_weight > 0.0)
    )
    if not np.any(valid):
        return ""
    mfp_m = vabs[valid] / rate_total[valid]
    wgt = n_ph_weight[valid]
    finite = np.isfinite(mfp_m) & (mfp_m > 0.0) & np.isfinite(wgt) & (wgt > 0.0)
    mfp_m = mfp_m[finite]
    wgt = wgt[finite]
    if mfp_m.size == 0 or wgt.size == 0:
        return ""
    order = np.argsort(mfp_m, kind="stable")
    mfp_nm = mfp_m[order] * 1e9
    wgt = wgt[order]
    cdf = np.cumsum(wgt) / np.sum(wgt)
    weighted_mean_nm = float(np.sum(mfp_nm * wgt) / np.sum(wgt))
    median_nm = float(mfp_nm[min(int(np.searchsorted(cdf, 0.5, side="left")), mfp_nm.size - 1)])
    p90_nm = float(mfp_nm[min(int(np.searchsorted(cdf, 0.9, side="left")), mfp_nm.size - 1)])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        output_csv,
        [["mfp_nm", "cdf", "n_ph_weight"], *np.column_stack((mfp_nm, cdf, wgt)).tolist()],
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return str(output_csv)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.plot(mfp_nm, cdf, color="#0f4c81", linewidth=2.2)
    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("CDF")
    ax.set_title(plot_title)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)
    ax.axvline(median_nm, color="#bf360c", linestyle="--", linewidth=1.2, alpha=0.9, label=f"median={median_nm:.3g} nm")
    ax.legend(loc="lower right", frameon=False)
    ax.text(
        0.03,
        0.08,
        "\n".join(
            (
                f"Nsp = {mfp_nm.size}",
                f"sum|n_ph| = {np.sum(wgt):.3g}",
                f"mean = {weighted_mean_nm:.3g} nm",
                f"median = {median_nm:.3g} nm",
                f"p90 = {p90_nm:.3g} nm",
            )
        ),
        transform=ax.transAxes,
        fontsize=9.5,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#f8fafc", "edgecolor": "#d1d5db", "alpha": 0.95},
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(output_png)


def export_particle_total_scattering_rate_cdf(
    output_png: str | Path,
    state: SimulationState,
    Tcell: np.ndarray,
    opts: dict[str, Any],
    spec: dict[str, Any],
    x_label: str = "Total Scattering Rate (s$^{-1}$)",
    plot_title: str = "Total Scattering Rate CDF",
) -> str:
    output_png = Path(output_png)
    if len(state.p) == 0:
        return ""
    rates = precompute_relax_times(state, np.asarray(Tcell, dtype=np.float64), opts, spec)
    if rates.size == 0:
        return ""
    rate_total = active_total_scattering_rate_for_particles(opts, rates)
    valid = np.isfinite(rate_total) & (rate_total > 0.0)
    if not np.any(valid):
        return ""
    rate_sorted = np.sort(rate_total[valid])
    cdf = np.arange(1, rate_sorted.size + 1, dtype=np.float64) / float(rate_sorted.size)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    q_levels = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    q_values = np.quantile(rate_sorted, q_levels)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.plot(rate_sorted, cdf, color="#14532d", linewidth=2.2)
    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("CDF")
    ax.set_title(plot_title)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)
    for q_level, q_value in zip(q_levels, q_values):
        ax.axhline(q_level, color="#64748b", linestyle=":", linewidth=0.9, alpha=0.9)
        ax.axvline(q_value, color="#b45309", linestyle="--", linewidth=1.1, alpha=0.9)
        ax.scatter([q_value], [q_level], color="#b45309", s=18, zorder=3)
        ax.annotate(
            f"{q_level:.1f}: {q_value:.3g}",
            xy=(q_value, q_level),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8.8,
            color="#7c2d12",
        )
    ax.text(
        0.03,
        0.08,
        "\n".join(
            (
                f"N = {rate_sorted.size}",
                f"q0.1 = {q_values[0]:.3g} s^-1",
                f"q0.5 = {q_values[1]:.3g} s^-1",
                f"q0.9 = {q_values[2]:.3g} s^-1",
            )
        ),
        transform=ax.transAxes,
        fontsize=9.2,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#f8fafc", "edgecolor": "#d1d5db", "alpha": 0.95},
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(output_png)


def export_particle_omega_tau_distribution(
    output_png: str | Path,
    state: SimulationState,
    Tcell: np.ndarray,
    opts: dict[str, Any],
    spec: dict[str, Any],
    plot_title: str = "Particle Omega-Tau Distribution",
) -> str:
    output_png = Path(output_png)
    if len(state.p) == 0:
        return ""
    rates = precompute_relax_times(state, np.asarray(Tcell, dtype=np.float64), opts, spec)
    if rates.size == 0:
        return ""
    rate_total = np.sum(np.maximum(rates, 0.0), axis=1)
    omega = np.asarray(state.p.w, dtype=np.float64).reshape(-1)
    valid = np.isfinite(rate_total) & (rate_total > 0.0) & np.isfinite(omega) & (omega > 0.0)
    if not np.any(valid):
        return ""
    tau_s = 1.0 / rate_total[valid]
    omega_valid = omega[valid]
    finite_tau = np.isfinite(tau_s) & (tau_s > 0.0)
    tau_s = tau_s[finite_tau]
    omega_valid = omega_valid[finite_tau]
    if tau_s.size == 0:
        return ""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    hb = ax.hexbin(
        omega_valid,
        tau_s,
        gridsize=70,
        bins="log",
        mincnt=1,
        linewidths=0.0,
        cmap="viridis",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")

    def log_bounds(data: np.ndarray) -> tuple[float, float]:
        vals = np.asarray(data, dtype=np.float64)
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        if vals.size == 0:
            return 1.0, 10.0
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0 or hi <= 0.0:
            return 1.0, 10.0
        if hi <= lo:
            return lo / 1.5, hi * 1.5
        lo_q = float(np.quantile(vals, 0.002))
        hi_q = float(np.quantile(vals, 0.998))
        lo_use = max(lo_q, np.finfo(np.float64).tiny)
        hi_use = max(hi_q, lo_use * 1.01)
        pad = 10.0 ** 0.05
        return lo_use / pad, hi_use * pad

    xlo, xhi = log_bounds(omega_valid)
    ylo, yhi = log_bounds(tau_s)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Angular Frequency $\\omega$ (rad/s)")
    ax.set_ylabel("Relaxation Time $\\tau$ (s)")
    ax.set_title(plot_title)
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Count")
    ax.text(
        0.03,
        0.05,
        "\n".join(
            (
                f"N = {tau_s.size}",
                f"$\\omega_{{med}}$ = {float(np.median(omega_valid)):.3g} rad/s",
                f"$\\tau_{{med}}$ = {float(np.median(tau_s)):.3g} s",
            )
        ),
        transform=ax.transAxes,
        fontsize=9.5,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#f8fafc", "edgecolor": "#d1d5db", "alpha": 0.95},
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(output_png)


def export_initial_mfp_cdf(
    out_cfg: dict[str, Any],
    state: SimulationState,
    Tcell: np.ndarray,
    opts: dict[str, Any],
    spec: dict[str, Any],
) -> str:
    if not out_cfg.get("enabled", False):
        return ""
    run_dir = Path(str(out_cfg.get("run_dir", "")))
    if not run_dir:
        return ""
    return export_particle_mfp_cdf(
        run_dir / "initial_mfp_cdf.png",
        state,
        Tcell,
        opts,
        spec,
        x_label="Initial Mean Free Path (nm)",
        plot_title="Initial Particle Mean Free Path CDF",
    )


def export_initial_scattering_rate_cdf(
    out_cfg: dict[str, Any],
    state: SimulationState,
    Tcell: np.ndarray,
    opts: dict[str, Any],
    spec: dict[str, Any],
) -> str:
    if not out_cfg.get("enabled", False):
        return ""
    run_dir = Path(str(out_cfg.get("run_dir", "")))
    if not run_dir:
        return ""
    return export_particle_total_scattering_rate_cdf(
        run_dir / "initial_total_scattering_rate_cdf.png",
        state,
        Tcell,
        opts,
        spec,
        x_label="Initial Total Scattering Rate (s$^{-1}$)",
        plot_title="Initial Total Scattering Rate CDF",
    )


def write_periodic_output(
    out_cfg: dict[str, Any],
    mesh: dict[str, Any],
    spec: dict[str, Any],
    state: SimulationState,
    Tprime: np.ndarray,
    Toc: np.ndarray | None,
    opts: dict[str, Any],
    dt_info: dict[str, Any] | None,
    step: int,
    elapsed_time: float,
    dt_step: float,
) -> None:
    if not out_cfg.get("enabled", False):
        return

    def fmt_sig5(value: Any) -> Any:
        if isinstance(value, (str, bytes)):
            return value
        if isinstance(value, (int, np.integer)):
            return int(value)
        try:
            x = float(value)
        except Exception:
            return value
        if not np.isfinite(x):
            return str(x)
        return f"{x:.5g}"

    step_dir = Path(out_cfg["steps_dir"]) / f"step_{step:05d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    wall_clock_elapsed = time.perf_counter() - out_cfg["run_wallclock_tic"]
    dt_info = {} if dt_info is None else dict(dt_info)
    I, J, K = np.meshgrid(np.arange(1, mesh["Nx"] + 1), np.arange(1, mesh["Ny"] + 1), np.arange(1, mesh["Nz"] + 1), indexing="ij")
    temp_data = np.column_stack((I.ravel(order="F"), J.ravel(order="F"), K.ravel(order="F"), np.asarray(Tprime, dtype=np.float64).reshape(-1)))
    write_csv_rows(step_dir / "temperature.txt", [["idxcell", "idycell", "idzcell", "Temperature"], *temp_data.tolist()])
    if Toc is not None:
        toc_data = np.column_stack((I.ravel(order="F"), J.ravel(order="F"), K.ravel(order="F"), np.asarray(Toc, dtype=np.float64).reshape(-1)))
        write_csv_rows(Path(out_cfg["toc_dir"]) / f"step_{step:05d}.txt", [["idxcell", "idycell", "idzcell", "Toc"], *toc_data.tolist()])
    # --- branch stats (supports both single-material dict and multi-material list) ---
    if isinstance(spec, dict):
        _specs_out = [spec]
    else:
        _specs_out = list(spec)
    multi_out = len(_specs_out) > 1
    header = ["material_id", "material_key", "branch_id", "branch_name",
              "superparticle_count", "phonon_count_net", "phonon_count_abs",
              "energy_net_J", "energy_abs_J"]
    branch_rows: list[list[Any]] = [header]
    if len(state.p) == 0:
        for mi, sp in enumerate(_specs_out):
            mat_key = sp.get("material_key", f"mat_{mi}")
            for ib, name in enumerate(sp["branches"], start=1):
                branch_rows.append([mi, mat_key, ib, name, 0, 0, 0, 0, 0])
    else:
        n_net = state.p.E / (HBAR * np.maximum(state.p.w, 1e-30))
        n_abs = np.abs(state.p.E) / (HBAR * np.maximum(state.p.w, 1e-30))
        mat_id_arr = state.p.material_id
        for mi, sp in enumerate(_specs_out):
            mat_key = sp.get("material_key", f"mat_{mi}")
            mat_mask = mat_id_arr == mi
            for ib, name in enumerate(sp["branches"], start=1):
                mask = mat_mask & (state.p.b == ib)
                branch_rows.append([
                    mi, mat_key, ib, name,
                    int(np.count_nonzero(mask)),
                    float(n_net[mask].sum()),
                    float(n_abs[mask].sum()),
                    float(state.p.E[mask].sum()),
                    float(np.abs(state.p.E[mask]).sum()),
                ])
    write_csv_rows(step_dir / "branch_stats.txt", branch_rows)

    def flux_from_energy(energy_value: float, area_value: float, time_value: float) -> float:
        if time_value > 0 and area_value > 0:
            return energy_value / (area_value * time_value)
        return np.nan

    def monitor_energy_row(label: str, stat_type: str, elapsed_time_value: float, window_time_value: float, pos_dir_pos: float, pos_dir_neg: float, neg_dir_pos: float, neg_dir_neg: float, warning: str) -> list[Any]:
        total_abs = pos_dir_pos + pos_dir_neg + neg_dir_pos + neg_dir_neg
        net_value = pos_dir_pos - pos_dir_neg - neg_dir_pos + neg_dir_neg
        return [
            label,
            stat_type,
            fmt_sig5(elapsed_time_value),
            fmt_sig5(window_time_value),
            fmt_sig5(pos_dir_pos),
            fmt_sig5(pos_dir_neg),
            fmt_sig5(neg_dir_pos),
            fmt_sig5(neg_dir_neg),
            fmt_sig5(total_abs),
            fmt_sig5(net_value),
            warning,
        ]

    def monitor_flux_row(label: str, stat_type: str, elapsed_time_value: float, window_time_value: float, pos_dir_pos: float, pos_dir_neg: float, neg_dir_pos: float, neg_dir_neg: float, warning: str) -> list[Any]:
        total_abs = pos_dir_pos + pos_dir_neg + neg_dir_pos + neg_dir_neg
        net_value = pos_dir_pos - pos_dir_neg - neg_dir_pos + neg_dir_neg
        return [
            label,
            stat_type,
            fmt_sig5(elapsed_time_value),
            fmt_sig5(window_time_value),
            fmt_sig5(pos_dir_pos),
            fmt_sig5(pos_dir_neg),
            fmt_sig5(neg_dir_pos),
            fmt_sig5(neg_dir_neg),
            fmt_sig5(total_abs),
            fmt_sig5(net_value),
            warning,
        ]

    energy_rows = [[
        "label",
        "stat_type",
        "elapsed_time_s",
        "window_time_s",
        "pos_dir_pos_particle_J",
        "pos_dir_neg_particle_abs_J",
        "neg_dir_pos_particle_J",
        "neg_dir_neg_particle_abs_J",
        "total_abs_J",
        "net_J",
        "warning",
    ]]
    heat_rows = [[
        "label",
        "stat_type",
        "elapsed_time_s",
        "window_time_s",
        "pos_dir_pos_particle_W_m2",
        "pos_dir_neg_particle_abs_W_m2",
        "neg_dir_pos_particle_W_m2",
        "neg_dir_neg_particle_abs_W_m2",
        "total_abs_W_m2",
        "net_W_m2",
        "warning",
    ]]
    for i, mon in enumerate(out_cfg["monitors"]):
        area = float(mon["area"])
        interval_t = float(out_cfg["interval_time"])
        cum_t = float(elapsed_time)
        interval_fwd_pos = float(out_cfg["interval_energy_fwd_pos"][i])
        interval_fwd_neg = float(out_cfg["interval_energy_fwd_neg"][i])
        interval_bwd_pos = float(out_cfg["interval_energy_bwd_pos"][i])
        interval_bwd_neg = float(out_cfg["interval_energy_bwd_neg"][i])
        cumulative_fwd_pos = float(out_cfg["cum_energy_fwd_pos"][i])
        cumulative_fwd_neg = float(out_cfg["cum_energy_fwd_neg"][i])
        cumulative_bwd_pos = float(out_cfg["cum_energy_bwd_pos"][i])
        cumulative_bwd_neg = float(out_cfg["cum_energy_bwd_neg"][i])
        energy_rows.append(
            monitor_energy_row(
                mon["label"],
                "interval",
                cum_t,
                interval_t,
                interval_fwd_pos,
                interval_fwd_neg,
                interval_bwd_pos,
                interval_bwd_neg,
                mon["warning"],
            )
        )
        energy_rows.append(
            monitor_energy_row(
                mon["label"],
                "cumulative",
                cum_t,
                cum_t,
                cumulative_fwd_pos,
                cumulative_fwd_neg,
                cumulative_bwd_pos,
                cumulative_bwd_neg,
                mon["warning"],
            )
        )
        heat_rows.append(
            monitor_flux_row(
                mon["label"],
                "interval",
                cum_t,
                interval_t,
                flux_from_energy(interval_fwd_pos, area, interval_t),
                flux_from_energy(interval_fwd_neg, area, interval_t),
                flux_from_energy(interval_bwd_pos, area, interval_t),
                flux_from_energy(interval_bwd_neg, area, interval_t),
                mon["warning"],
            )
        )
        heat_rows.append(
            monitor_flux_row(
                mon["label"],
                "cumulative",
                cum_t,
                cum_t,
                flux_from_energy(cumulative_fwd_pos, area, cum_t),
                flux_from_energy(cumulative_fwd_neg, area, cum_t),
                flux_from_energy(cumulative_bwd_pos, area, cum_t),
                flux_from_energy(cumulative_bwd_neg, area, cum_t),
                mon["warning"],
            )
        )
    write_csv_rows(step_dir / "heat_energy.txt", energy_rows)
    write_csv_rows(step_dir / "heat_flux.txt", heat_rows)
    write_csv_rows(
        step_dir / "step_info.txt",
        [[
            "step",
            "dt_s",
            "dt_cfl_s",
            "dt_scat_s",
            "elapsed_time_s",
            "interval_time_s",
            "wall_clock_elapsed_s",
            "Np",
            "T_min_K",
            "T_mean_K",
            "T_max_K",
        ], [
            step,
            dt_step,
            float(get_or(dt_info, "dt_cfl", np.nan)),
            float(get_or(dt_info, "dt_prob", np.nan)),
            elapsed_time,
            out_cfg["interval_time"],
            wall_clock_elapsed,
            len(state.p),
            float(np.min(Tprime)),
            float(np.mean(Tprime)),
            float(np.max(Tprime)),
        ]],
    )
    export_particle_mfp_cdf(
        step_dir / "mfp_cdf.png",
        state,
        Tprime,
        opts,
        spec,
        x_label="Mean Free Path (nm)",
        plot_title=f"Particle Mean Free Path CDF @ step {step}",
    )
    export_particle_total_scattering_rate_cdf(
        step_dir / "total_scattering_rate_cdf.png",
        state,
        Tprime,
        opts,
        spec,
        x_label="Total Scattering Rate (s$^{-1}$)",
        plot_title=f"Total Scattering Rate CDF @ step {step}",
    )
    export_particle_omega_tau_distribution(
        step_dir / "omega_tau_distribution.png",
        state,
        Tprime,
        opts,
        spec,
        plot_title=f"Particle Omega-Tau Distribution @ step {step}",
    )
    with Path(out_cfg["step_history_file"]).open("a", encoding="utf-8") as f:
        f.write(
            f"{step},"
            f"{dt_step:.16g},"
            f"{float(get_or(dt_info, 'dt_cfl', np.nan)):.16g},"
            f"{float(get_or(dt_info, 'dt_prob', np.nan)):.16g},"
            f"{elapsed_time:.16g},"
            f"{out_cfg['interval_time']:.16g},"
            f"{wall_clock_elapsed:.16g},"
            f"{len(state.p)},"
            f"{float(np.min(Tprime)):.16g},"
            f"{float(np.mean(Tprime)):.16g},"
            f"{float(np.max(Tprime)):.16g}\n"
        )


def normal_key(normal_name: str) -> str:
    mapping = {"+X": "xp", "-X": "xn", "+Y": "yp", "-Y": "yn", "+Z": "zp", "-Z": "zn"}
    if normal_name.upper() not in mapping:
        raise ValueError(f"invalid normal {normal_name}")
    return mapping[normal_name.upper()]


def opposite_normal(normal_name: str) -> str:
    mapping = {"+X": "-X", "-X": "+X", "+Y": "-Y", "-Y": "+Y", "+Z": "-Z", "-Z": "+Z"}
    return mapping[normal_name.upper()]


def point_hits_rule(pt: np.ndarray, rule: dict[str, Any], tol: float) -> bool:
    b = np.asarray(rule["bounds"], dtype=np.float64)
    if rule["axis"] == "x":
        return abs(pt[0] - rule["coord"]) <= tol and b[2] - tol <= pt[1] <= b[3] + tol and b[4] - tol <= pt[2] <= b[5] + tol
    if rule["axis"] == "y":
        return abs(pt[1] - rule["coord"]) <= tol and b[0] - tol <= pt[0] <= b[1] + tol and b[4] - tol <= pt[2] <= b[5] + tol
    if rule["axis"] == "z":
        return abs(pt[2] - rule["coord"]) <= tol and b[0] - tol <= pt[0] <= b[1] + tol and b[2] - tol <= pt[1] <= b[3] + tol
    return False


def normalize_action(mode_name: str) -> str:
    name = mode_name.lower()
    if name in {"pass", "open"}:
        return "pass"
    if name == "scatter":
        return "scatter"
    if name in {"reflect", "adiabatic"}:
        return "reflect"
    if name in {"catch", "absorb"}:
        return "catch"
    if name == "generate":
        return "generate"
    if name == "periodic":
        return "periodic"
    return "pass"


def face_rule(mesh: dict[str, Any], normal: str, pt: np.ndarray) -> dict[str, Any] | None:
    rules = mesh.get("face_rules", {}).get("by_normal", {}).get(normal_key(normal), [])
    tol = 1e-12 * max(1.0, float(np.max(np.abs(pt))))
    for rule in rules:
        if point_hits_rule(pt, rule, tol):
            return rule
    return None


def face_action(mesh: dict[str, Any], normal: str, pt: np.ndarray) -> str:
    rule = face_rule(mesh, normal, pt)
    if rule is None:
        return "pass"
    return normalize_action(rule["mode"])


def resolve_scatter_face_action(rule: dict[str, Any]) -> str:
    probs = np.asarray(rule.get("scatter_probs", np.array([1.0, 0.0, 0.0], dtype=np.float64)), dtype=np.float64)
    xi = float(np.random.random())
    if xi < probs[0]:
        return "scatter_diffuse"
    if xi < probs[0] + probs[1]:
        return "reflect"
    return "pass"


def point_hits_monitor(pt: np.ndarray, mon: dict[str, Any]) -> bool:
    b = np.asarray(mon["bounds"], dtype=np.float64)
    tol = 1e-12 * max(1.0, float(np.max(np.abs(np.concatenate((pt, b))))))
    if mon["axis"] == "x":
        return abs(pt[0] - mon["coord"]) <= tol and b[2] - tol <= pt[1] <= b[3] + tol and b[4] - tol <= pt[2] <= b[5] + tol
    if mon["axis"] == "y":
        return abs(pt[1] - mon["coord"]) <= tol and b[0] - tol <= pt[0] <= b[1] + tol and b[4] - tol <= pt[2] <= b[5] + tol
    if mon["axis"] == "z":
        return abs(pt[2] - mon["coord"]) <= tol and b[0] - tol <= pt[0] <= b[1] + tol and b[2] - tol <= pt[1] <= b[3] + tol
    return False


MONITOR_ENERGY_BUCKETS = ("fwd_pos", "fwd_neg", "bwd_pos", "bwd_neg")


def init_monitor_output_accumulators(nmon: int) -> dict[str, np.ndarray]:
    acc: dict[str, np.ndarray] = {}
    for prefix in ("cum", "interval"):
        acc[f"{prefix}_energy_net"] = np.zeros(nmon, dtype=np.float64)
        acc[f"{prefix}_crossings_pos"] = np.zeros(nmon, dtype=np.float64)
        acc[f"{prefix}_crossings_neg"] = np.zeros(nmon, dtype=np.float64)
        for bucket in MONITOR_ENERGY_BUCKETS:
            acc[f"{prefix}_energy_{bucket}"] = np.zeros(nmon, dtype=np.float64)
            acc[f"{prefix}_crossings_{bucket}"] = np.zeros(nmon, dtype=np.float64)
    return acc


def empty_heat_flux_stats(mesh: dict[str, Any]) -> dict[str, np.ndarray]:
    nmon = len(mesh.get("heat_flux_monitors", []))
    stats = {
        "net_energy": np.zeros(nmon, dtype=np.float64),
        "crossings_pos": np.zeros(nmon, dtype=np.float64),
        "crossings_neg": np.zeros(nmon, dtype=np.float64),
    }
    for bucket in MONITOR_ENERGY_BUCKETS:
        stats[f"energy_{bucket}"] = np.zeros(nmon, dtype=np.float64)
        stats[f"crossings_{bucket}"] = np.zeros(nmon, dtype=np.float64)
    return stats


def merge_heat_flux_stats(stats: dict[str, np.ndarray], add_stats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not add_stats:
        return stats
    for key in stats:
        stats[key] = stats[key] + add_stats[key]
    return stats


def tally_heat_flux_crossing(stats: dict[str, np.ndarray], mesh: dict[str, Any], pt: np.ndarray, normal: str, packet_E: float) -> None:
    for i, mon in enumerate(mesh.get("heat_flux_monitors", [])):
        if not point_hits_monitor(pt, mon):
            continue
        if mon["effective_normal"].upper() == normal.upper():
            sign = 1.0
        elif opposite_normal(mon["effective_normal"]).upper() == normal.upper():
            sign = -1.0
        else:
            sign = 0.0
        if sign == 0.0:
            continue
        stats["net_energy"][i] += sign * packet_E
        packet_mag = abs(packet_E)
        packet_is_positive = packet_E >= 0.0
        if sign > 0:
            stats["crossings_pos"][i] += 1
            bucket = "fwd_pos" if packet_is_positive else "fwd_neg"
        else:
            stats["crossings_neg"][i] += 1
            bucket = "bwd_pos" if packet_is_positive else "bwd_neg"
        stats[f"energy_{bucket}"][i] += packet_mag
        stats[f"crossings_{bucket}"][i] += 1


def next_particle_id(p: ParticleBlock) -> int:
    return int(p.id.max()) if len(p) else 0


def particles_total_energy(state: SimulationState, opts: dict[str, Any]) -> float:
    if len(state.p):
        return float(state.p.E.sum())
    return float(get_or(opts, "E_eff", 1e-18) * 0.0)


def collect_reservoir_cells(mesh: dict[str, Any]) -> np.ndarray:
    cells = [res["cell_ids"] for res in mesh.get("reservoirs", []) if np.asarray(res["cell_ids"]).size]
    if not cells:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(cells).astype(np.int64))


def refresh_reservoir_particles(state: SimulationState, mesh: dict[str, Any], spec: dict[str, Any] | list[dict[str, Any]], opts: dict[str, Any]) -> tuple[SimulationState, dict[str, Any]]:
    info = {
        "refreshed": False,
        "cell_ids": np.zeros(0, dtype=np.int64),
        "removed_particles": 0,
        "added_particles": 0,
        "target_temperature_cell": np.zeros(0, dtype=np.float64),
        "reference_temperature_cell": np.zeros(0, dtype=np.float64),
    }
    cells = collect_reservoir_cells(mesh)
    if cells.size == 0:
        return state, info
    Nc = infer_Nc(mesh)
    target_T_all = np.asarray(state.info.get("reservoir_target_temperature_cell", state.info.get("T_init_cell", np.zeros(0))), dtype=np.float64).reshape(-1)
    if target_T_all.size != Nc:
        fallback_T = float(get_or(opts, "T0", get_or(opts, "Tref", 300.0)))
        target_T_all, _ = load_initial_temperature_field(mesh, opts, fallback_T)
    Tref_all = np.asarray(state.info.get("reservoir_reference_temperature_cell", state.info.get("Tref_cell", np.zeros(0))), dtype=np.float64).reshape(-1)
    if Tref_all.size != Nc:
        default_Tref = float(get_or(opts, "Tref", np.mean(target_T_all)))
        Tref_all, _ = load_reference_temperature_field(mesh, opts, default_Tref)
    keep_mask = np.ones(len(state.p), dtype=bool)
    if len(state.p):
        keep_mask = ~np.isin(state.p.cell.astype(np.int64), cells)
    kept = state.p.take(keep_mask)
    removed = int(len(state.p) - len(kept))
    refresh_opts = dict(opts)
    refresh_opts["E_eff"] = float(state.WE)
    # Normalise specs.
    if isinstance(spec, dict):
        _specs_r: list[dict[str, Any]] = [spec]
    else:
        _specs_r = list(spec)
    cell_mat = np.asarray(mesh.get("cell_material_index", np.ones(Nc, dtype=np.int32)), dtype=np.int32)
    blocks: list[ParticleBlock] = []
    total_added = 0
    next_id = next_particle_id(kept)
    for mi, sp in enumerate(_specs_r):
        # Reservoir cells belonging to this material.
        mat_cells = cells[cell_mat[cells - 1] == (mi + 1)]
        if mat_cells.size == 0:
            continue
        newp, sinfo = sample_particles_for_cells(
            mesh, sp, refresh_opts, mat_cells,
            target_T_all[mat_cells - 1],
            Tref_all[mat_cells - 1],
            next_id,
        )
        if len(newp):
            blocks.append(newp)
            next_id = int(newp.id.max())
        total_added += sinfo["Nsp_tot"]
    if blocks:
        newp_all = blocks[0]
        for blk in blocks[1:]:
            newp_all = newp_all.append(blk)
    else:
        newp_all = ParticleBlock.empty()
    state.p = kept.append(newp_all)
    state.Nsp_cell = np.bincount(state.p.cell.astype(np.int64) - 1, minlength=Nc).astype(np.int64) if len(state.p) else np.zeros(Nc, dtype=np.int64)
    mask = np.zeros(Nc, dtype=bool)
    mask[cells - 1] = True
    state.info["reservoir_cell_mask"] = mask
    state.info["reservoir_target_temperature_cell"] = target_T_all
    state.info["reservoir_reference_temperature_cell"] = Tref_all
    state.info["reservoir_last_refresh_particles"] = total_added
    info.update(
        {
            "refreshed": True,
            "cell_ids": cells,
            "removed_particles": removed,
            "added_particles": total_added,
            "target_temperature_cell": target_T_all[cells - 1],
            "reference_temperature_cell": Tref_all[cells - 1],
        }
    )
    return state, info


def locate_cell_from_point(mesh: dict[str, Any], pt: np.ndarray) -> int:
    x, y, z = pt
    X = np.asarray(mesh["x_edges"], dtype=np.float64)
    Y = np.asarray(mesh["y_edges"], dtype=np.float64)
    Z = np.asarray(mesh["z_edges"], dtype=np.float64)
    ix = np.searchsorted(X, x, side="right") - 1
    iy = np.searchsorted(Y, y, side="right") - 1
    iz = np.searchsorted(Z, z, side="right") - 1
    if ix < 0 or iy < 0 or iz < 0 or ix >= mesh["Nx"] or iy >= mesh["Ny"] or iz >= mesh["Nz"]:
        return 0
    return int(sub2ind(mesh["Nx"], mesh["Ny"], mesh["Nz"], ix + 1, iy + 1, iz + 1))


def build_local_diff_spectrum(spec: dict[str, Any], Tsrc: float, Tloc: float) -> np.ndarray:
    w = np.maximum(np.asarray(spec["w_mid"], dtype=np.float64), 0.0)
    DOS = np.maximum(np.asarray(spec["DOS_w_b"], dtype=np.float64), 0.0)
    dw = ensure_2d_dw(spec)
    n_src = bose_occupation(w, max(Tsrc, 1e-12))
    n_loc = bose_occupation(w, max(Tloc, 1e-12))
    Wbm = HBAR * w * DOS * (n_src - n_loc) * dw
    Wbm[DOS <= 0] = 0.0
    return Wbm


def region_sampler_volume(mesh: dict[str, Any], src: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    region = src.get("region", {"type": "cells", "id": []})
    if str(region.get("type", "cells")).lower() != "cells":
        raise ValueError("spawn_heat_source currently supports only cell regions")
    all_V = cell_volumes(mesh)
    cells = np.asarray(region.get("id", np.arange(1, infer_Nc(mesh) + 1)), dtype=np.int64).reshape(-1)
    Vc = all_V[cells - 1]
    ctr = np.mean(mesh["centers"][cells - 1], axis=0)
    return cells, Vc, ctr


def emit_particles_from_Wbm_refsample(
    mesh: dict[str, Any],
    spec: dict[str, Any],
    cells: np.ndarray,
    Vc: np.ndarray,
    Wbm: np.ndarray,
    dE_tot: float,
    E_eff: float,
    Tref: float,
    next_id_base: int,
    force_positive: bool,
) -> ParticleBlock:
    B, Nw = Wbm.shape
    Wabs = np.abs(Wbm)
    if not np.any(Wabs > 0):
        return ParticleBlock.empty()
    cdf_ref = np.cumsum(Wabs.ravel(order="F"))
    if cdf_ref[-1] <= 0:
        return ParticleBlock.empty()
    cdf_ref /= cdf_ref[-1]
    Nexp = abs(dE_tot) / E_eff
    Nsp = int(np.floor(Nexp) + (np.random.random() < (Nexp - np.floor(Nexp))))
    if Nsp <= 0:
        return ParticleBlock.empty()
    lin = np.searchsorted(cdf_ref, np.random.random(Nsp), side="left")
    lin = np.clip(lin, 0, B * Nw - 1)
    b = (lin % B + 1).astype(np.int32)
    m = (lin // B + 1).astype(np.int32)
    has_edges = "w_edges" in spec and len(spec["w_edges"]) == Nw + 1
    if has_edges:
        w_lo = np.asarray(spec["w_edges"], dtype=np.float64)[m - 1]
        w_hi = np.asarray(spec["w_edges"], dtype=np.float64)[m]
        w = w_lo + (w_hi - w_lo) * np.random.random(Nsp)
    else:
        w = np.asarray(spec["w_mid"], dtype=np.float64)[b - 1, m - 1]
    q, vabs = q_vabs_from_w_table(spec, w, b)
    dirs = rand_unit_vec_batch(Nsp)
    cdf_cell = np.cumsum(Vc) / Vc.sum()
    picked = np.searchsorted(cdf_cell, np.random.random(Nsp), side="left")
    picked = np.clip(picked, 0, cells.size - 1)
    sampled_cells = cells[picked].astype(np.int32)
    # Per-particle material_id from mesh material index.
    cell_mat_idx = np.asarray(mesh.get("cell_material_index", np.zeros(1, dtype=np.int32)), dtype=np.int32)
    mat_id = np.zeros(Nsp, dtype=np.int32)
    valid_cells = (sampled_cells >= 1) & (sampled_cells <= cell_mat_idx.size)
    mat_id[valid_cells] = cell_mat_idx[sampled_cells[valid_cells] - 1] - 1
    mat_id[~valid_cells] = -1
    x, y, z = uniform_positions_in_cells(mesh, sampled_cells)
    sign_vals = np.sign(Wbm.ravel(order="F")[lin]).astype(np.int8)
    sign_vals[sign_vals == 0] = 1
    if force_positive:
        sign_vals[:] = 1
    E = sign_vals.astype(np.float64) * E_eff
    ids = np.arange(next_id_base + 1, next_id_base + Nsp + 1, dtype=np.int64)
    return ParticleBlock(
        id=ids,
        par_id=ids.copy(),
        cell=sampled_cells,
        material_id=mat_id,
        x=x,
        y=y,
        z=z,
        b=b,
        m=m,
        w=w,
        q=q,
        vx=vabs * dirs[:, 0],
        vy=vabs * dirs[:, 1],
        vz=vabs * dirs[:, 2],
        vabs=vabs,
        E=E,
        sgn=sign_vals,
        n_ph=E / (HBAR * np.maximum(w, 1e-30)),
        seed=np.random.randint(1, 2**31 - 1, size=Nsp, dtype=np.int64),
        t_left=np.zeros(Nsp, dtype=np.float64),
    )


def spawn_heat_source(opts: dict[str, Any], mesh: dict[str, Any], spec: dict[str, Any], state: SimulationState, Tprime: np.ndarray, lut: dict[str, Any], src: dict[str, Any], dt: float) -> ParticleBlock:
    if str(src.get("type", "")).lower() != "volume":
        raise ValueError("only volumetric heat sources are supported")
    E_eff = float(src.get("E_eff", get_or(opts, "E_eff", 1e-18)))
    cells, Vc, ctr = region_sampler_volume(mesh, src)
    V_region = float(Vc.sum())
    if V_region <= 0:
        return ParticleBlock.empty()
    Tref_all = np.asarray(state.info.get("Tref_cell", np.zeros(0)), dtype=np.float64).reshape(-1)
    if Tref_all.size >= int(np.max(cells)):
        Tref_loc = float(np.mean(Tref_all[cells - 1]))
    elif np.isfinite(get_or(opts, "Tref", np.nan)):
        Tref_loc = float(opts["Tref"])
    else:
        cid = locate_cell_from_point(mesh, ctr)
        Tref_loc = float(Tprime[cid - 1]) if 1 <= cid <= Tprime.size else float(np.mean(Tprime))
    dE_tot = float(src.get("dE_tot", float(src["qvol"]) * V_region * dt))
    if abs(dE_tot) <= 0.0:
        return ParticleBlock.empty()
    Uref = float(lut["U_interp"](np.clip(Tref_loc, lut["T"][0], lut["T"][-1])))
    Tsrc = float(lut["inv_interp"](np.clip(Uref + dE_tot / V_region, lut["U_mono"][0], lut["U_mono"][-1])))
    Wbm = build_local_diff_spectrum(spec, Tsrc, Tref_loc)
    if not np.any(Wbm != 0):
        return ParticleBlock.empty()
    return emit_particles_from_Wbm_refsample(mesh, spec, cells, Vc, Wbm, dE_tot, E_eff, Tref_loc, next_particle_id(state.p), bool(src.get("force_positive", False)))


@njit(parallel=True, cache=True)
def _precompute_relax_times_numba(
    cell_id: np.ndarray,
    material_id: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
    vabs: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    q: np.ndarray,
    Tcell: np.ndarray,
    branch_is_la: np.ndarray,       # (n_materials, max_B)  — bool_
    branch_is_ta: np.ndarray,       # (n_materials, max_B)  — bool_
    branch_is_loto: np.ndarray,     # (n_materials, max_B)  — bool_
    omega_cut: np.ndarray,          # (n_materials,)         — float64
    B_L: np.ndarray,                # (n_materials,)
    B_TN: np.ndarray,
    B_TU: np.ndarray,
    tau_LTO_ps: np.ndarray,
    A_imp: np.ndarray,
    B_imp: np.ndarray,
    C_imp: np.ndarray,
    bulk_L: np.ndarray,
    bulk_F: np.ndarray,
    Tsi: np.ndarray,
    Delta: np.ndarray,
    n0: float,
    n1: float,
    n2: float,
) -> np.ndarray:
    Np = w.size
    n_mat = B_L.size
    # r_tau columns:
    #   0 -> LA normal three-phonon rate
    #   1 -> TA normal three-phonon rate
    #   2 -> TA umklapp three-phonon rate
    #   3 -> LO/TO fixed relaxation rate
    #   4 -> impurity scattering rate: A_imp*w^4 + B_imp*w^2 + C_imp
    #   5 -> boundary scattering rate
    out = np.zeros((Np, 6), dtype=np.float64)
    for i in prange(Np):
        mi = material_id[i]
        # Clamp material index to valid range; if out of bounds use material 0.
        if mi < 0 or mi >= n_mat:
            mi = 0
        bi = b[i] - 1
        wi = w[i]
        vg_i = vabs[i]
        cid = cell_id[i] - 1
        T = max(Tcell[cid], 1e-6)
        if branch_is_la[mi, bi]:
            out[i, 0] = B_L[mi] * wi * wi * T**3
        if branch_is_ta[mi, bi]:
            out[i, 1] = B_TN[mi] * wi * T**4
            if wi > omega_cut[mi]:
                x = HBAR * wi / (K_B * T)
                out[i, 2] = B_TU[mi] * wi * wi / max(math.sinh(x), 1e-12)
        if branch_is_loto[mi, bi]:
            out[i, 3] = 1.0 / (tau_LTO_ps[mi] * 1e-12)
        out[i, 4] = A_imp[mi] * wi**4 + B_imp[mi] * wi**2 + C_imp[mi]
        tsi_val = Tsi[mi]
        delta_val = Delta[mi]
        bulk_l_val = bulk_L[mi]
        bulk_f_val = bulk_F[mi]
        if tsi_val > 0.0:
            normv = math.sqrt(vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i])
            if normv > 0.0:
                cosB = abs((vx[i] * n0 + vy[i] * n1 + vz[i] * n2) / normv)
            else:
                cosB = math.sqrt(1.0 / 3.0)
            p_spec = math.exp(-4.0 * (max(q[i], 0.0) * delta_val) ** 2 * cosB * cosB)
            Ffilm = (1.0 - p_spec) / (1.0 + p_spec)
            out[i, 5] = vg_i / max(tsi_val, 1e-12) * Ffilm
        elif bulk_l_val > 0.0:
            out[i, 5] = vg_i / max(bulk_l_val * bulk_f_val, 1e-12)
    return out


def _resolve_scattering_param(opts: dict[str, Any], key: str, material_index: int, specs: list[dict[str, Any]], default: float) -> float:
    """Resolve a scattering parameter for a specific material.

    Priority: material_scattering dict → per-spec override → global opts → hard-coded default.
    """
    mat_scat = opts.get("material_scattering", {})
    spec = specs[material_index] if 0 <= material_index < len(specs) else {}
    mat_key = spec.get("material_key", "")
    if mat_key and mat_key in mat_scat and key in mat_scat[mat_key]:
        return float(mat_scat[mat_key][key])
    return float(get_or(opts, key, default))


def _build_per_material_param_array(opts: dict[str, Any], specs: list[dict[str, Any]], key: str, default: float) -> np.ndarray:
    """Build a (n_materials,) array of a scattering parameter."""
    n = len(specs)
    arr = np.empty(n, dtype=np.float64)
    for i in range(n):
        arr[i] = _resolve_scattering_param(opts, key, i, specs, default)
    return arr


def precompute_relax_times(
    state: SimulationState,
    Tcell: np.ndarray,
    opts: dict[str, Any],
    specs: list[dict[str, Any]] | dict[str, Any],
) -> np.ndarray:
    """Compute per-particle relaxation-time matrix (Np × 6).

    ``specs`` may be a single-material spec dict (legacy) or a list of
    per-material spec dicts (multi-material).
    """
    if len(state.p) == 0:
        return np.zeros((0, 6), dtype=np.float64)
    # Normalise to list form.
    if isinstance(specs, dict):
        _specs: list[dict[str, Any]] = [specs]
    else:
        _specs = list(specs)
    n_materials = len(_specs)
    parallel_cfg = dict(get_or(opts, "parallel", {}))
    ensure_num_threads(int(get_or(parallel_cfg, "num_threads", os.cpu_count() or 1)))
    n_hat = np.asarray(get_or(opts, "transport_n", np.array([0.0, 0.0, 1.0])), dtype=np.float64).reshape(3)
    n_hat /= max(np.linalg.norm(n_hat), np.finfo(np.float64).eps)
    # Pad branch metadata arrays to max_B across all materials.
    max_B = max(int(s.get("B", 0)) for s in _specs)
    branch_la = np.zeros((n_materials, max_B), dtype=np.bool_)
    branch_ta = np.zeros((n_materials, max_B), dtype=np.bool_)
    branch_loto = np.zeros((n_materials, max_B), dtype=np.bool_)
    omega_cut = np.zeros(n_materials, dtype=np.float64)
    for mi, spec in enumerate(_specs):
        b_la = np.asarray(spec.get("branch_is_la", np.zeros(0, dtype=np.bool_)), dtype=np.bool_)
        b_ta = np.asarray(spec.get("branch_is_ta", np.zeros(0, dtype=np.bool_)), dtype=np.bool_)
        b_loto = np.asarray(spec.get("branch_is_loto", np.zeros(0, dtype=np.bool_)), dtype=np.bool_)
        Bm = b_la.size
        branch_la[mi, :Bm] = b_la
        branch_ta[mi, :Bm] = b_ta
        branch_loto[mi, :Bm] = b_loto
        omega_cut[mi] = float(spec.get("omega_cut_ta", 0.0))
    # Build per-material parameter arrays.
    BL_arr = _build_per_material_param_array(opts, _specs, "BL", 1.18e-24)
    BTN_arr = _build_per_material_param_array(opts, _specs, "BTN", 10.5e-13)
    BTU_arr = _build_per_material_param_array(opts, _specs, "BTU", 2.89e-18)
    tau_LTO_arr = _build_per_material_param_array(opts, _specs, "tau_LTO_ps", 3.5)
    A_imp_arr = _build_per_material_param_array(opts, _specs, "A_imp", 1.32e-45)
    B_imp_arr = _build_per_material_param_array(opts, _specs, "B_imp", 0.0)
    C_imp_arr = _build_per_material_param_array(opts, _specs, "C_imp", 0.0)
    bulk_L_arr = _build_per_material_param_array(opts, _specs, "PB_bulk_L", 7.16e-3)
    bulk_F_arr = _build_per_material_param_array(opts, _specs, "PB_bulk_F", 0.68)
    Tsi_arr = _build_per_material_param_array(opts, _specs, "PB_Tsi", 0.0)
    Delta_arr = _build_per_material_param_array(opts, _specs, "PB_Delta", 0.0)
    return _precompute_relax_times_numba(
        state.p.cell.astype(np.int64),
        state.p.material_id.astype(np.int64),
        state.p.b.astype(np.int64),
        state.p.w.astype(np.float64),
        state.p.vabs.astype(np.float64),
        state.p.vx.astype(np.float64),
        state.p.vy.astype(np.float64),
        state.p.vz.astype(np.float64),
        state.p.q.astype(np.float64),
        np.asarray(Tcell, dtype=np.float64),
        branch_la,
        branch_ta,
        branch_loto,
        omega_cut,
        BL_arr,
        BTN_arr,
        BTU_arr,
        tau_LTO_arr,
        A_imp_arr,
        B_imp_arr,
        C_imp_arr,
        bulk_L_arr,
        bulk_F_arr,
        Tsi_arr,
        Delta_arr,
        float(n_hat[0]),
        float(n_hat[1]),
        float(n_hat[2]),
    )


def step_pick_dt(mesh: dict[str, Any], spec: dict[str, Any] | list[dict[str, Any]], opts: dict[str, Any], state: SimulationState, rates_mat: np.ndarray | None) -> tuple[float, dict[str, Any]]:
    dt_min = float(get_or(opts, "dt_min", 1e-15))
    dt_max = float(get_or(opts, "dt_max", 1e-10))
    dt_fixed = float(get_or(opts, "dt", 0.0))
    cfl_safe = float(get_or(opts, "dt_safety_cfl", 0.5))
    p_target = float(get_or(opts, "p_target", 0.01))
    mode = str(get_or(opts, "dt_prob_mode", "max")).lower()
    pctl = float(get_or(opts, "dt_prob_pctl", 95.0))
    # --- vg_max for CFL: use max across all materials (or current particles) ---
    if isinstance(spec, dict):
        _specs_dt = [spec]
    else:
        _specs_dt = list(spec)
    vg_max = 1e-9
    for sp in _specs_dt:
        vg_max = max(vg_max, float(get_or(sp, "vg_max", 0.0)))
    vg_max = max(vg_max, 1e-9)
    if len(state.p):
        vg_max = max(vg_max, float(state.p.vabs.max()))
    hmin = float(get_or(mesh, "hmin", 1.0))
    dt_cfl = cfl_safe * hmin / vg_max
    r_stat = 0.0
    dt_prob = np.inf
    r_tot = np.zeros(0, dtype=np.float64)
    if rates_mat is not None and rates_mat.size:
        r_tot = active_total_scattering_rate_for_particles(opts, rates_mat)
        r_pos = r_tot[r_tot > 0]
        if r_pos.size:
            if mode == "avg":
                r_stat = float(np.mean(r_pos))
            elif mode == "pctl":
                r_stat = float(np.percentile(r_pos, pctl))
            else:
                r_stat = float(np.max(r_pos))
            dt_prob = -math.log(max(1.0 - p_target, 1e-12)) / max(r_stat, 1e-30)
    dt_raw = dt_fixed if dt_fixed > 0.0 else min(dt_cfl, dt_prob)
    dt = min(max(dt_raw, dt_min), dt_max)
    if dt_fixed > 0.0:
        mode = "fixed"
    p_hit = 1.0 - np.exp(-dt * np.maximum(r_tot, 0.0)) if r_tot.size else np.zeros(0, dtype=np.float64)
    return dt, {
        "dt_cfl": dt_cfl,
        "dt_prob": dt_prob,
        "dt_raw": dt_raw,
        "dt": dt,
        "vg_max": vg_max,
        "hmin": hmin,
        "p_target": p_target,
        "mode": mode,
        "r_stat": r_stat,
        "p_scat_max": float(np.max(p_hit)) if p_hit.size else np.nan,
        "hits_expected": float(np.sum(p_hit)) if p_hit.size else 0.0,
    }


def get_w_edges(spec: dict[str, Any]) -> np.ndarray:
    if "w_edges" in spec and len(spec["w_edges"]) == spec["w_mid"].shape[1] + 1:
        return np.asarray(spec["w_edges"], dtype=np.float64)
    wm = np.asarray(spec["w_mid"], dtype=np.float64)[0]
    edges = np.zeros(wm.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (wm[:-1] + wm[1:])
    edges[0] = wm[0] - max(wm[1] - wm[0], 1e-30)
    edges[-1] = wm[-1] + max(wm[-1] - wm[-2], 1e-30)
    return edges


def build_pp_rate_table_from_particles(b_vec: np.ndarray, w_vec: np.ndarray, rpp_vec: np.ndarray, w_edges: np.ndarray, B: int, Nw: int) -> np.ndarray:
    m_vec = np.digitize(w_vec, w_edges) - 1
    m_vec = np.clip(m_vec, 0, Nw - 1)
    lin = m_vec * B + (b_vec.astype(np.int64) - 1)
    sumR = np.bincount(lin, weights=rpp_vec, minlength=B * Nw).reshape(Nw, B).T
    cntR = np.bincount(lin, minlength=B * Nw).reshape(Nw, B).T
    Gamma = np.zeros((B, Nw), dtype=np.float64)
    mask = cntR > 0
    Gamma[mask] = sumR[mask] / cntR[mask]
    global_mean = float(sumR.sum() / max(cntR.sum(), 1))
    if not np.isfinite(global_mean) or global_mean <= 0:
        global_mean = 1e-30
    for bb in range(B):
        if np.any(mask[bb]):
            row_mean = float(sumR[bb].sum() / max(cntR[bb].sum(), 1))
            fill_val = row_mean if np.isfinite(row_mean) and row_mean > 0 else global_mean
        else:
            fill_val = global_mean
        Gamma[bb, ~mask[bb]] = fill_val
    if not np.any(Gamma > 0):
        Gamma[:] = 1.0
    return Gamma


def scattering_rate_table_at_T(spec: dict[str, Any], T: float, opts: dict[str, Any], fallback_table: np.ndarray | None) -> np.ndarray:
    if callable(opts.get("pp_rate_table_fun")):
        Gamma = np.asarray(opts["pp_rate_table_fun"](T, spec), dtype=np.float64)
    elif callable(spec.get("pp_rate_table_fun")):
        Gamma = np.asarray(spec["pp_rate_table_fun"](T), dtype=np.float64)
    elif "Gamma_pp_T" in spec and "T_grid" in spec:
        Tg = np.asarray(spec["T_grid"], dtype=np.float64).reshape(-1)
        G3 = np.asarray(spec["Gamma_pp_T"], dtype=np.float64)
        if T <= Tg[0]:
            Gamma = G3[:, :, 0]
        elif T >= Tg[-1]:
            Gamma = G3[:, :, -1]
        else:
            k = int(np.searchsorted(Tg, T, side="right") - 1)
            a = (T - Tg[k]) / (Tg[k + 1] - Tg[k])
            Gamma = (1.0 - a) * G3[:, :, k] + a * G3[:, :, k + 1]
    else:
        Gamma = scattering_rate_table_formula(spec, T, opts)
    Gamma = np.maximum(np.nan_to_num(Gamma, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    if not np.any(Gamma > 0) and fallback_table is not None:
        Gamma = np.maximum(fallback_table, 0.0)
    if not np.any(Gamma > 0):
        if fallback_table is not None:
            Gamma = np.ones_like(fallback_table)
        else:
            Gamma = np.ones_like(np.asarray(spec["w_mid"], dtype=np.float64))
    return Gamma


def particle_scattering(
    state: SimulationState,
    mesh: dict[str, Any],
    spec: dict[str, Any] | list[dict[str, Any]],
    opts: dict[str, Any],
    dt: float,
    Tstar: np.ndarray,
    r_tau: np.ndarray,
    scatter_lut: dict[str, Any] | None = None,
    Tscatt_cell: np.ndarray | None = None,
) -> tuple[SimulationState, dict[str, Any]]:
    Nc = infer_Nc(mesh)
    if len(state.p) == 0:
        return state, {"hits": 0, "hits_expected": 0.0, "pp": 0, "pi": 0, "pb": 0}
    # Normalise to list form.
    if isinstance(spec, dict):
        _specs: list[dict[str, Any]] = [spec]
    else:
        _specs = list(spec)
    single_mat = len(_specs) == 1
    pb_on = bool(get_or(get_or(opts, "scatter", {}), "pb_on", False))
    b = state.p.b.copy().astype(np.int32)
    w = state.p.w.copy()
    vabs = state.p.vabs.copy()
    vx = state.p.vx.copy()
    vy = state.p.vy.copy()
    vz = state.p.vz.copy()
    cell_id = state.p.cell.astype(np.int64)
    mat_id = state.p.material_id.copy()
    # r_tau[:, i] follows the precomputed order:
    # LA normal, TA normal, TA umklapp, LO/TO, impurity, boundary.
    rLA, rTAN, rTAU, rLTO, rPI, rPB = [np.maximum(r_tau[:, i], 0.0) for i in range(6)]
    if not pb_on:
        rPB[:] = 0.0
    rPP = pp_only_rate_for_particles(b, r_tau, spec, mat_id)
    rTOT = rPP + rPI + rPB
    hit = np.random.random(len(state.p)) < (1.0 - np.exp(-dt * np.maximum(rTOT, 0.0)))
    sel = np.random.random(np.count_nonzero(hit)) * rTOT[hit]
    rPP_hit = rPP[hit]
    rPI_hit = rPI[hit]
    isPP = np.zeros(len(state.p), dtype=bool)
    isPI = np.zeros(len(state.p), dtype=bool)
    isPB = np.zeros(len(state.p), dtype=bool)
    isPP[hit] = sel < rPP_hit
    isPI[hit] = (~isPP[hit]) & (sel < (rPP_hit + rPI_hit))
    isPB[hit] = ~isPP[hit] & ~isPI[hit]
    if np.any(isPI):
        dirs = rand_unit_vec_batch(int(np.count_nonzero(isPI)))
        idx = np.flatnonzero(isPI)
        vx[idx] = vabs[idx] * dirs[:, 0]
        vy[idx] = vabs[idx] * dirs[:, 1]
        vz[idx] = vabs[idx] * dirs[:, 2]
    if np.any(isPP):
        iiPP = np.flatnonzero(isPP)
        cPP = cell_id[iiPP]
        mPP = mat_id[iiPP]
        Tref_cell = reference_temperature_from_state(state, opts, Nc)
        if Tscatt_cell is None:
            Tscatt_cell, _ = update_pp_scattering_temperature_from_energy(state, mesh, spec, opts, r_tau, scatter_lut)
        use_center = bool(get_or(opts, "use_bin_center_w", True))
        b_new = np.zeros(iiPP.size, dtype=np.int32)
        w_new = np.zeros(iiPP.size, dtype=np.float64)
        vabs_new = np.zeros(iiPP.size, dtype=np.float64)
        # Group PP particles by (material_id, cell) and resample within each material's spectral grid.
        # Precompute per-material fallback tables; cache in a dict keyed by material index.
        fallback_cache: dict[int, np.ndarray] = {}
        for mi in range(len(_specs)):
            si = _specs[mi]
            Bi, Nwi = si["w_mid"].shape
            wi_edges = get_w_edges(si)
            fallback_cache[mi] = build_pp_rate_table_from_particles(
                b.astype(np.int64), w, rTOT, wi_edges, Bi, Nwi
            )
        for cid in np.unique(cPP):
            # Determine which material this cell belongs to (most common among particles here).
            cell_mask = cPP == cid
            mat_ids_in_cell = mPP[cell_mask]
            unique_mats, counts = np.unique(mat_ids_in_cell, return_counts=True)
            dominant_mat = unique_mats[np.argmax(counts)]
            dominant_mat = max(0, min(dominant_mat, len(_specs) - 1))
            sp = _specs[dominant_mat]
            Bm, Nwm = sp["w_mid"].shape
            w_edges_m = get_w_edges(sp)
            fallback = fallback_cache.get(dominant_mat,
                build_pp_rate_table_from_particles(b.astype(np.int64), w, rTOT, w_edges_m, Bm, Nwm))
            DOS_m = np.maximum(np.asarray(sp["DOS_w_b"], dtype=np.float64), 0.0)
            dw2_m = ensure_2d_dw(sp)
            wmid_m = np.asarray(sp["w_mid"], dtype=np.float64)
            Tcell_loc = float(max(Tscatt_cell[int(cid) - 1], 1e-12))
            Gamma = scattering_rate_table_at_T(sp, Tcell_loc, opts, fallback)
            if str(get_or(opts, "mode", "absolute")).lower() == "absolute":
                W = HBAR * wmid_m * DOS_m * bose_occupation(wmid_m, Tcell_loc) * Gamma * dw2_m
            else:
                Tref_loc = float(max(Tref_cell[int(cid) - 1], 1e-12))
                dNB = bose_occupation(wmid_m, Tcell_loc) - bose_occupation(wmid_m, Tref_loc)
                W = HBAR * wmid_m * DOS_m * np.abs(dNB) * Gamma * dw2_m
            W = np.maximum(np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
            if not np.any(W > 0):
                W[:] = 1.0
            cdf = np.cumsum(W.ravel(order="F"))
            cdf /= cdf[-1]
            draws = np.random.random(np.count_nonzero(cell_mask))
            lin = np.searchsorted(cdf, draws, side="left")
            lin = np.clip(lin, 0, Bm * Nwm - 1)
            b_pick = (lin % Bm + 1).astype(np.int32)
            m_pick = (lin // Bm + 1).astype(np.int32)
            if use_center or "w_edges" not in sp or len(sp["w_edges"]) != Nwm + 1:
                w_pick = wmid_m[b_pick - 1, m_pick - 1]
            else:
                w_lo = w_edges_m[m_pick - 1]
                w_hi = w_edges_m[m_pick]
                w_pick = w_lo + (w_hi - w_lo) * np.random.random(draws.size)
            _, v_pick = q_vabs_from_w_table(sp, w_pick, b_pick)
            b_new[cell_mask] = b_pick
            w_new[cell_mask] = w_pick
            vabs_new[cell_mask] = v_pick
        dirs = rand_unit_vec_batch(iiPP.size)
        b[iiPP] = b_new
        w[iiPP] = w_new
        vabs[iiPP] = vabs_new
        vx[iiPP] = vabs_new * dirs[:, 0]
        vy[iiPP] = vabs_new * dirs[:, 1]
        vz[iiPP] = vabs_new * dirs[:, 2]
    changed = isPP | isPI | (pb_on & isPB)
    if np.any(changed):
        state.p.b = b
        state.p.w = w
        state.p.vabs = vabs
        state.p.vx = vx
        state.p.vy = vy
        state.p.vz = vz
        # Recompute wavevectors — dispatch to correct material spec per particle.
        if single_mat:
            state.p.q = q_vabs_from_w_table(_specs[0], state.p.w, state.p.b)[0]
        else:
            q_new = np.zeros_like(state.p.w)
            for mi in range(len(_specs)):
                mask = mat_id == mi
                if not np.any(mask):
                    continue
                q_new[mask] = q_vabs_from_w_table(_specs[mi], state.p.w[mask], state.p.b[mask])[0]
            state.p.q = q_new
    hits_expected = float(np.sum(1.0 - np.exp(-dt * np.maximum(rTOT, 0.0))))
    return state, {
        "hits": int(np.count_nonzero(hit)),
        "hits_expected": hits_expected,
        "pp": int(np.count_nonzero(isPP)),
        "pi": int(np.count_nonzero(isPI)),
        "pb": int(np.count_nonzero(isPB)),
    }


def particle_fly(state: SimulationState, mesh: dict[str, Any], dt: float, opts: dict[str, Any], spec: dict[str, Any] | None = None) -> tuple[SimulationState, dict[str, Any]]:
    if len(state.p) == 0:
        return state, {
            "heat_flux": empty_heat_flux_stats(mesh),
            "catch_count": 0,
            "generate_count": 0,
            "pass_count": 0,
            "reflect_count": 0,
        }
    if str(get_or(opts, "fly_mode", "cell")).lower() not in {"cell", "domain"}:
        raise ValueError(f'unknown fly_mode "{opts["fly_mode"]}"')
    Nx, Ny, Nz = mesh["Nx"], mesh["Ny"], mesh["Nz"]
    X = np.asarray(mesh["x_edges"], dtype=np.float64)
    Y = np.asarray(mesh["y_edges"], dtype=np.float64)
    Z = np.asarray(mesh["z_edges"], dtype=np.float64)
    epsl = 1e-11
    x = state.p.x.copy()
    y = state.p.y.copy()
    z = state.p.z.copy()
    pid = state.p.id.copy()
    par_id = state.p.par_id.copy()
    pb = state.p.b.copy()
    pm = state.p.m.copy()
    pw = state.p.w.copy()
    pq = state.p.q.copy()
    vx = state.p.vx.copy()
    vy = state.p.vy.copy()
    vz = state.p.vz.copy()
    vabs = state.p.vabs.copy()
    pE = state.p.E.copy()
    psgn = state.p.sgn.copy()
    pn_ph = state.p.n_ph.copy()
    pseed = state.p.seed.copy()
    cid = state.p.cell.astype(np.int64).copy()
    pmat = state.p.material_id.copy()
    alive = (cid >= 1) & (cid <= Nx * Ny * Nz) & (vabs > 0)
    t_rem = np.full(len(state.p), dt, dtype=np.float64)
    next_id = next_particle_id(state.p)
    cid_safe = np.clip(cid, 1, Nx * Ny * Nz)
    ix, iy, iz = ind2sub(Nx, Ny, Nz, cid_safe)
    # Pre-fetch cell material index array for material_id updates.
    _cell_mat_idx = np.asarray(mesh.get("cell_material_index", np.zeros(1, dtype=np.int32)), dtype=np.int32)
    if np.any(alive):
        idx = np.flatnonzero(alive)
        x[idx] = np.clip(x[idx], X[ix[idx] - 1] + epsl, X[ix[idx]] - epsl)
        y[idx] = np.clip(y[idx], Y[iy[idx] - 1] + epsl, Y[iy[idx]] - epsl)
        z[idx] = np.clip(z[idx], Z[iz[idx] - 1] + epsl, Z[iz[idx]] - epsl)
    flux = empty_heat_flux_stats(mesh)
    catch_count = 0
    generate_count = 0
    pass_count = 0
    reflect_count = 0
    # DMM interface statistics.
    dmm_attempt = 0
    dmm_transmit = 0
    dmm_reflect = 0
    dmm_energy_attempt = 0.0
    dmm_energy_transmit = 0.0
    dmm_energy_reflect = 0.0
    dmm_detail: dict[str, dict[str, int]] = {}  # "i->j" -> {attempt, transmit, reflect}
    # Per-(pair, omega_bin) tracking for empirical transmission comparison.
    dmm_bin_detail: dict[tuple[str, int], dict[str, int]] = {}  # (pair_str, w_bin) -> {att, T, R}
    while True:
        act = np.flatnonzero(alive & (t_rem > 0))
        if act.size == 0:
            break
        INF = np.full(act.size, np.inf, dtype=np.float64)
        tx = INF.copy()
        ty = INF.copy()
        tz = INF.copy()
        xa, ya, za = x[act], y[act], z[act]
        vxa, vya, vza = vx[act], vy[act], vz[act]
        ix_a, iy_a, iz_a = ix[act], iy[act], iz[act]
        xL, xR = X[ix_a - 1], X[ix_a]
        yB, yT = Y[iy_a - 1], Y[iy_a]
        zD, zU = Z[iz_a - 1], Z[iz_a]
        pos = vxa > 0
        tx[pos] = (xR[pos] - xa[pos]) / vxa[pos]
        neg = vxa < 0
        tx[neg] = (xL[neg] - xa[neg]) / vxa[neg]
        pos = vya > 0
        ty[pos] = (yT[pos] - ya[pos]) / vya[pos]
        neg = vya < 0
        ty[neg] = (yB[neg] - ya[neg]) / vya[neg]
        pos = vza > 0
        tz[pos] = (zU[pos] - za[pos]) / vza[pos]
        neg = vza < 0
        tz[neg] = (zD[neg] - za[neg]) / vza[neg]
        tx = np.maximum(tx, 0.0)
        ty = np.maximum(ty, 0.0)
        tz = np.maximum(tz, 0.0)
        tcell = tx.copy()
        axis_ix = np.ones(act.size, dtype=np.int8)
        m = ty < tcell
        tcell[m] = ty[m]
        axis_ix[m] = 2
        m = tz < tcell
        tcell[m] = tz[m]
        axis_ix[m] = 3
        tf = np.minimum(tcell, t_rem[act])
        x[act] += vx[act] * tf
        y[act] += vy[act] * tf
        z[act] += vz[act] * tf
        t_rem[act] -= tf
        hit = np.isfinite(tcell) & (np.abs(tf - tcell) <= np.maximum(1e-15 * tcell, 1e-18))
        for jj in np.flatnonzero(hit):
            k = int(act[jj])
            if axis_ix[jj] == 1:
                normal = "+X" if vx[k] > 0 else "-X"
            elif axis_ix[jj] == 2:
                normal = "+Y" if vy[k] > 0 else "-Y"
            else:
                normal = "+Z" if vz[k] > 0 else "-Z"
            pt = np.array([x[k], y[k], z[k]], dtype=np.float64)
            matched_rule = face_rule(mesh, normal, pt)
            action = "pass" if matched_rule is None else normalize_action(matched_rule["mode"])
            if action == "scatter":
                action = resolve_scatter_face_action(matched_rule)
            packet_E = float(pE[k]) if k < pE.size else 0.0
            if action == "pass":
                pass_count += 1
                tally_heat_flux_crossing(flux, mesh, pt, normal, packet_E)
                if normal == "+X":
                    if ix[k] >= Nx:
                        alive[k] = False
                        cid[k] = -1
                    else:
                        ix[k] += 1
                        x[k] = X[ix[k] - 1] + epsl
                elif normal == "-X":
                    if ix[k] <= 1:
                        alive[k] = False
                        cid[k] = -1
                    else:
                        ix[k] -= 1
                        x[k] = X[ix[k]] - epsl
                elif normal == "+Y":
                    if iy[k] >= Ny:
                        alive[k] = False
                        cid[k] = -1
                    else:
                        iy[k] += 1
                        y[k] = Y[iy[k] - 1] + epsl
                elif normal == "-Y":
                    if iy[k] <= 1:
                        alive[k] = False
                        cid[k] = -1
                    else:
                        iy[k] -= 1
                        y[k] = Y[iy[k]] - epsl
                elif normal == "+Z":
                    if iz[k] >= Nz:
                        alive[k] = False
                        cid[k] = -1
                    else:
                        iz[k] += 1
                        z[k] = Z[iz[k] - 1] + epsl
                else:
                    if iz[k] <= 1:
                        alive[k] = False
                        cid[k] = -1
                    else:
                        iz[k] -= 1
                        z[k] = Z[iz[k]] - epsl
                if alive[k]:
                    cid[k] = int(sub2ind(Nx, Ny, Nz, ix[k], iy[k], iz[k]))
                    # --- DMM interface crossing (Phase 7) -----------------
                    old_mat = int(pmat[k])
                    nc = cid[k]
                    if 1 <= nc <= _cell_mat_idx.size:
                        new_mat = int(_cell_mat_idx[nc - 1] - 1)
                    else:
                        new_mat = -1
                    dmm = mesh.get("dmm_tables", {})
                    if dmm and old_mat >= 0 and new_mat >= 0 and old_mat != new_mat:
                        key = (old_mat, new_mat)
                        T_ab = dmm.get(key)
                        if T_ab is not None:
                            # --- DMM attempt ---
                            dmm_attempt += 1
                            packet_abs = abs(packet_E)
                            dmm_energy_attempt += packet_abs
                            pair_str = f"{old_mat}->{new_mat}"
                            if pair_str not in dmm_detail:
                                dmm_detail[pair_str] = {"attempt": 0, "transmit": 0, "reflect": 0}
                            dmm_detail[pair_str]["attempt"] += 1
                            # Bin frequency to get transmission probability.
                            # w_edges is stored on specs[0] (all materials share the same grid).
                            w_edges_global = np.asarray(mesh.get("specs", [{}])[0].get("w_edges", np.array([0.0, 1e30])), dtype=np.float64)
                            w_idx = np.digitize(pw[k], w_edges_global) - 1
                            w_idx = max(0, min(w_idx, T_ab.size - 1))
                            # Per-bin tracking.
                            bin_key = (pair_str, int(w_idx))
                            if bin_key not in dmm_bin_detail:
                                dmm_bin_detail[bin_key] = {"att": 0, "T": 0, "R": 0}
                            dmm_bin_detail[bin_key]["att"] += 1
                            T_val = float(T_ab[w_idx])
                            if np.random.random() < T_val:
                                # --- Transmitted ---
                                dmm_transmit += 1
                                dmm_energy_transmit += packet_abs
                                dmm_detail[pair_str]["transmit"] += 1
                                dmm_bin_detail[bin_key]["T"] += 1
                                # Update material state from target spec:
                                # Preserve frequency w and energy E; recompute q, vabs from
                                # the destination material dispersion.  Clamp branch id to
                                # the target material's valid range (safe fallback).
                                pmat[k] = new_mat
                                target_spec = mesh.get("specs", [{}])[new_mat] if new_mat < len(mesh.get("specs", [])) else {}
                                if target_spec:
                                    B_target = int(target_spec.get("B", 1))
                                    # --- Branch remapping: resample branch in target material
                                    #     weighted by DOS_w_b * |vg_w| at this omega bin.
                                    DOS_b_target = np.asarray(target_spec.get("DOS_w_b", np.zeros((1,1))), dtype=np.float64)
                                    vg_b_target = np.abs(np.asarray(target_spec.get("vg_w", np.zeros((1,1))), dtype=np.float64))
                                    Nw_tgt = DOS_b_target.shape[1]
                                    wi = max(0, min(w_idx, Nw_tgt - 1))
                                    branch_weights = DOS_b_target[:, wi] * np.maximum(vg_b_target[:, wi], 1e-30)
                                    total_w = float(branch_weights.sum())
                                    if total_w > 0:
                                        # Sample new branch from CDF.
                                        cdf_b = np.cumsum(branch_weights) / total_w
                                        r = np.random.random()
                                        new_b = int(np.searchsorted(cdf_b, r, side="right")) + 1
                                        pb[k] = max(1, min(new_b, B_target))
                                    else:
                                        # Fallback: clamp old branch to valid range.
                                        old_b = int(pb[k])
                                        pb[k] = max(1, min(old_b, B_target))
                                    # Recompute q and vabs from target material's branch_lookups
                                    # using the (possibly resampled) branch.
                                    new_q, new_vabs = q_vabs_from_w_table(target_spec, pw[k], pb[k])
                                    pq[k] = float(new_q[0]) if new_q.size else pq[k]
                                    new_v = float(new_vabs[0]) if new_vabs.size else vabs[k]
                                    # Keep direction unit vector; update speed.
                                    old_vabs = max(vabs[k], 1e-30)
                                    dir_x = vx[k] / old_vabs
                                    dir_y = vy[k] / old_vabs
                                    dir_z = vz[k] / old_vabs
                                    # Renormalise direction.
                                    norm_d = math.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
                                    if norm_d > 0:
                                        dir_x /= norm_d; dir_y /= norm_d; dir_z /= norm_d
                                    vabs[k] = new_v
                                    vx[k] = new_v * dir_x
                                    vy[k] = new_v * dir_y
                                    vz[k] = new_v * dir_z
                            else:
                                # --- Reflected ---
                                dmm_reflect += 1
                                dmm_energy_reflect += packet_abs
                                dmm_detail[pair_str]["reflect"] += 1
                                dmm_bin_detail[bin_key]["R"] += 1
                                # Diffuse backscatter into original cell.
                                # rand_hemisphere_vec(normal) returns a direction *opposite*
                                # to the given normal, i.e. back into the original cell.
                                pmat[k] = old_mat
                                dirs_hemi = rand_hemisphere_vec(normal)
                                vx[k] = vabs[k] * dirs_hemi[0]
                                vy[k] = vabs[k] * dirs_hemi[1]
                                vz[k] = vabs[k] * dirs_hemi[2]
                                # Move back to original cell (the one we came from).
                                if normal == "+X":
                                    ix[k] = max(ix[k] - 1, 1)
                                    x[k] = X[ix[k]] - epsl
                                elif normal == "-X":
                                    ix[k] = min(ix[k] + 1, Nx)
                                    x[k] = X[ix[k] - 1] + epsl
                                elif normal == "+Y":
                                    iy[k] = max(iy[k] - 1, 1)
                                    y[k] = Y[iy[k]] - epsl
                                elif normal == "-Y":
                                    iy[k] = min(iy[k] + 1, Ny)
                                    y[k] = Y[iy[k] - 1] + epsl
                                elif normal == "+Z":
                                    iz[k] = max(iz[k] - 1, 1)
                                    z[k] = Z[iz[k]] - epsl
                                else:  # "-Z"
                                    iz[k] = min(iz[k] + 1, Nz)
                                    z[k] = Z[iz[k] - 1] + epsl
                                cid[k] = int(sub2ind(Nx, Ny, Nz, ix[k], iy[k], iz[k]))
                                # Restore material_id from the original cell.
                                nc_ref = cid[k]
                                if 1 <= nc_ref <= _cell_mat_idx.size:
                                    pmat[k] = _cell_mat_idx[nc_ref - 1] - 1
                                else:
                                    pmat[k] = old_mat
                        else:
                            pmat[k] = new_mat
                    else:
                        pmat[k] = new_mat
            elif action == "reflect":
                reflect_count += 1
                if normal in {"+X", "-X"}:
                    vx[k] = -vx[k]
                    x[k] = X[ix[k]] - epsl if normal == "+X" else X[ix[k] - 1] + epsl
                elif normal in {"+Y", "-Y"}:
                    vy[k] = -vy[k]
                    y[k] = Y[iy[k]] - epsl if normal == "+Y" else Y[iy[k] - 1] + epsl
                else:
                    vz[k] = -vz[k]
                    z[k] = Z[iz[k]] - epsl if normal == "+Z" else Z[iz[k] - 1] + epsl
            elif action == "scatter_diffuse":
                reflect_count += 1
                dirs = rand_hemisphere_vec(normal)
                vx[k] = vabs[k] * dirs[0]
                vy[k] = vabs[k] * dirs[1]
                vz[k] = vabs[k] * dirs[2]
                if normal in {"+X", "-X"}:
                    x[k] = X[ix[k]] - epsl if normal == "+X" else X[ix[k] - 1] + epsl
                elif normal in {"+Y", "-Y"}:
                    y[k] = Y[iy[k]] - epsl if normal == "+Y" else Y[iy[k] - 1] + epsl
                else:
                    z[k] = Z[iz[k]] - epsl if normal == "+Z" else Z[iz[k] - 1] + epsl
            elif action == "catch":
                catch_count += 1
                tally_heat_flux_crossing(flux, mesh, pt, normal, packet_E)
                alive[k] = False
                cid[k] = -1
            elif action == "periodic":
                if normal == "+X":
                    ix[k] = 1
                    x[k] = X[0] + epsl
                elif normal == "-X":
                    ix[k] = Nx
                    x[k] = X[-1] - epsl
                elif normal == "+Y":
                    iy[k] = 1
                    y[k] = Y[0] + epsl
                elif normal == "-Y":
                    iy[k] = Ny
                    y[k] = Y[-1] - epsl
                elif normal == "+Z":
                    iz[k] = 1
                    z[k] = Z[0] + epsl
                else:
                    iz[k] = Nz
                    z[k] = Z[-1] - epsl
                cid[k] = int(sub2ind(Nx, Ny, Nz, ix[k], iy[k], iz[k]))
                nc = cid[k]
                if 1 <= nc <= _cell_mat_idx.size:
                    pmat[k] = _cell_mat_idx[nc - 1] - 1
                else:
                    pmat[k] = -1
            elif action == "generate":
                generate_count += 1
                tally_heat_flux_crossing(flux, mesh, pt, normal, packet_E)
                has_neighbor = True
                child_x, child_y, child_z = x[k], y[k], z[k]
                child_ix, child_iy, child_iz = ix[k], iy[k], iz[k]
                if normal == "+X":
                    if ix[k] >= Nx:
                        has_neighbor = False
                    else:
                        child_ix = ix[k] + 1
                        child_x = X[child_ix - 1] + epsl
                elif normal == "-X":
                    if ix[k] <= 1:
                        has_neighbor = False
                    else:
                        child_ix = ix[k] - 1
                        child_x = X[child_ix] - epsl
                elif normal == "+Y":
                    if iy[k] >= Ny:
                        has_neighbor = False
                    else:
                        child_iy = iy[k] + 1
                        child_y = Y[child_iy - 1] + epsl
                elif normal == "-Y":
                    if iy[k] <= 1:
                        has_neighbor = False
                    else:
                        child_iy = iy[k] - 1
                        child_y = Y[child_iy] - epsl
                elif normal == "+Z":
                    if iz[k] >= Nz:
                        has_neighbor = False
                    else:
                        child_iz = iz[k] + 1
                        child_z = Z[child_iz - 1] + epsl
                else:
                    if iz[k] <= 1:
                        has_neighbor = False
                    else:
                        child_iz = iz[k] - 1
                        child_z = Z[child_iz] - epsl
                if has_neighbor:
                    next_id += 1
                    child_cid = int(sub2ind(Nx, Ny, Nz, child_ix, child_iy, child_iz))
                    pid = np.append(pid, next_id)
                    par_id = np.append(par_id, next_id)
                    cid = np.append(cid, child_cid)
                    x = np.append(x, child_x)
                    y = np.append(y, child_y)
                    z = np.append(z, child_z)
                    pb = np.append(pb, pb[k])
                    pm = np.append(pm, pm[k])
                    pw = np.append(pw, pw[k])
                    pq = np.append(pq, pq[k])
                    vx = np.append(vx, vx[k])
                    vy = np.append(vy, vy[k])
                    vz = np.append(vz, vz[k])
                    vabs = np.append(vabs, vabs[k])
                    pE = np.append(pE, pE[k])
                    psgn = np.append(psgn, psgn[k])
                    pn_ph = np.append(pn_ph, pn_ph[k])
                    pseed = np.append(pseed, np.random.randint(1, 2**31 - 1, dtype=np.int64))
                    t_rem = np.append(t_rem, t_rem[k])
                    alive = np.append(alive, True)
                    ix = np.append(ix, child_ix)
                    iy = np.append(iy, child_iy)
                    iz = np.append(iz, child_iz)
                    # Set child material_id from the destination cell.
                    child_mat = -1
                    if 1 <= child_cid <= _cell_mat_idx.size:
                        child_mat = _cell_mat_idx[child_cid - 1] - 1
                    pmat = np.append(pmat, child_mat)
                if normal in {"+X", "-X"}:
                    vx[k] = -vx[k]
                    x[k] = X[ix[k]] - epsl if normal == "+X" else X[ix[k] - 1] + epsl
                elif normal in {"+Y", "-Y"}:
                    vy[k] = -vy[k]
                    y[k] = Y[iy[k]] - epsl if normal == "+Y" else Y[iy[k] - 1] + epsl
                else:
                    vz[k] = -vz[k]
                    z[k] = Z[iz[k]] - epsl if normal == "+Z" else Z[iz[k] - 1] + epsl
            else:
                raise ValueError(f"unsupported face action {action}")
        alive_idx = np.flatnonzero(alive)
        if alive_idx.size:
            x[alive_idx] = np.clip(x[alive_idx], X[ix[alive_idx] - 1] + epsl, X[ix[alive_idx]] - epsl)
            y[alive_idx] = np.clip(y[alive_idx], Y[iy[alive_idx] - 1] + epsl, Y[iy[alive_idx]] - epsl)
            z[alive_idx] = np.clip(z[alive_idx], Z[iz[alive_idx] - 1] + epsl, Z[iz[alive_idx]] - epsl)
    keep = alive & (cid > 0)
    state.p = ParticleBlock(
        id=pid[keep],
        par_id=par_id[keep],
        cell=cid[keep].astype(np.int32),
        material_id=pmat[keep],
        x=x[keep],
        y=y[keep],
        z=z[keep],
        b=pb[keep],
        m=pm[keep],
        w=pw[keep],
        q=pq[keep],
        vx=vx[keep],
        vy=vy[keep],
        vz=vz[keep],
        vabs=vabs[keep],
        E=pE[keep],
        sgn=psgn[keep],
        n_ph=pn_ph[keep],
        seed=pseed[keep],
        t_left=np.zeros(np.count_nonzero(keep), dtype=np.float64),
    )
    return state, {
        "heat_flux": flux,
        "catch_count": catch_count,
        "generate_count": generate_count,
        "pass_count": pass_count,
        "reflect_count": reflect_count,
        # DMM interface stats
        "dmm_attempt": dmm_attempt,
        "dmm_transmit": dmm_transmit,
        "dmm_reflect": dmm_reflect,
        "dmm_energy_attempt": dmm_energy_attempt,
        "dmm_energy_transmit": dmm_energy_transmit,
        "dmm_energy_reflect": dmm_energy_reflect,
        "dmm_detail": dmm_detail,
        "dmm_bin_detail": dmm_bin_detail,
    }


def accumulate_output(output_cfg: dict[str, Any], fly_stats: dict[str, Any], dt: float) -> dict[str, Any]:
    if not output_cfg.get("enabled", False):
        return output_cfg
    output_cfg["cum_time"] += dt
    output_cfg["interval_time"] += dt
    h = fly_stats.get("heat_flux", {})
    if h:
        output_cfg["cum_energy_net"] += h["net_energy"]
        output_cfg["interval_energy_net"] += h["net_energy"]
        output_cfg["cum_crossings_pos"] += h["crossings_pos"]
        output_cfg["cum_crossings_neg"] += h["crossings_neg"]
        output_cfg["interval_crossings_pos"] += h["crossings_pos"]
        output_cfg["interval_crossings_neg"] += h["crossings_neg"]
        for bucket in MONITOR_ENERGY_BUCKETS:
            output_cfg[f"cum_energy_{bucket}"] += h[f"energy_{bucket}"]
            output_cfg[f"interval_energy_{bucket}"] += h[f"energy_{bucket}"]
            output_cfg[f"cum_crossings_{bucket}"] += h[f"crossings_{bucket}"]
            output_cfg[f"interval_crossings_{bucket}"] += h[f"crossings_{bucket}"]
    return output_cfg


def reset_output_interval(output_cfg: dict[str, Any]) -> dict[str, Any]:
    output_cfg["interval_energy_net"][:] = 0.0
    output_cfg["interval_crossings_pos"][:] = 0.0
    output_cfg["interval_crossings_neg"][:] = 0.0
    for bucket in MONITOR_ENERGY_BUCKETS:
        output_cfg[f"interval_energy_{bucket}"][:] = 0.0
        output_cfg[f"interval_crossings_{bucket}"][:] = 0.0
    output_cfg["interval_time"] = 0.0
    return output_cfg


def spawn_volume_sources_from_map(
    qvol: np.ndarray,
    opts: dict[str, Any],
    mesh: dict[str, Any],
    spec: dict[str, Any] | list[dict[str, Any]],
    state: SimulationState,
    Tprime: np.ndarray,
    lut: dict[str, Any] | list[dict[str, Any]],
    dt: float,
) -> ParticleBlock:
    blocks: list[ParticleBlock] = []
    next_id = next_particle_id(state.p)
    Vc = cell_volumes(mesh)
    Nc = infer_Nc(mesh)
    residual = np.asarray(state.info.get("volume_source_residual_J", np.zeros(Nc, dtype=np.float64)), dtype=np.float64).reshape(-1)
    if residual.size != Nc:
        residual = np.zeros(Nc, dtype=np.float64)
    # Normalise specs and LUTs.
    if isinstance(spec, dict):
        _specs_src: list[dict[str, Any]] = [spec]
        _luts_src: list[dict[str, Any]] = [lut] if isinstance(lut, dict) else [lut[0]]
    else:
        _specs_src = list(spec)
        _luts_src = list(lut) if isinstance(lut, list) else [lut]
    cell_mat = np.asarray(mesh.get("cell_material_index", np.ones(Nc, dtype=np.int32)), dtype=np.int32)
    qsrc_arr = np.asarray(qvol, dtype=np.float64).reshape(-1)
    for cid, qsrc in enumerate(qsrc_arr, start=1):
        residual[cid - 1] += float(qsrc) * float(Vc[cid - 1]) * dt
        dE_cell = float(residual[cid - 1])
        if qsrc > 0.0 and dE_cell <= 0.0:
            continue
        if qsrc < 0.0 and dE_cell >= 0.0:
            continue
        if qsrc == 0.0 and abs(dE_cell) <= 0.0:
            continue
        # Dispatch to the correct material spec and LUT for this cell.
        mi = int(cell_mat[cid - 1]) - 1
        if mi < 0 or mi >= len(_specs_src):
            continue  # unassigned cell
        cell_spec = _specs_src[mi]
        cell_lut = _luts_src[mi] if mi < len(_luts_src) else _luts_src[0]
        src = {
            "type": "volume",
            "qvol": float(qsrc),
            "dE_tot": dE_cell,
            "E_eff": state.WE,
            "region": {"type": "cells", "id": [cid]},
            "force_positive": qsrc > 0.0,
        }
        temp_state = SimulationState(p=state.p, WE=state.WE, Wp=state.Wp, Nsp_cell=state.Nsp_cell, enhance_factor=state.enhance_factor, info=state.info)
        newp = spawn_heat_source(opts, mesh, cell_spec, temp_state, Tprime, cell_lut, src, dt)
        if len(newp):
            residual[cid - 1] -= float(np.sum(newp.E))
            shift = next_id - (int(newp.id[0]) - 1)
            newp.id = newp.id + shift
            newp.par_id = newp.par_id + shift
            next_id = int(newp.id.max())
            blocks.append(newp)
    state.info["volume_source_residual_J"] = residual
    if not blocks:
        return ParticleBlock.empty()
    out = blocks[0]
    for blk in blocks[1:]:
        out = out.append(blk)
    return out


def resolve_linearization_temperature(mesh: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    if np.isfinite(get_or(opts, "T0", np.nan)):
        return opts
    fallback_T = np.nan
    if np.isfinite(get_or(opts, "Tref", np.nan)):
        fallback_T = float(opts["Tref"])
    elif np.isfinite(get_or(opts, "T_init", np.nan)):
        fallback_T = float(opts["T_init"])
    _, meta = load_initial_temperature_field(mesh, opts, fallback_T)
    opts["T0"] = float(meta["T_mean"]) if np.isfinite(meta["T_mean"]) else (fallback_T if np.isfinite(fallback_T) else 300.0)
    return opts


def precompute_dmm_tables(specs: list[dict[str, Any]]) -> dict[tuple[int, int], np.ndarray]:
    """Precompute DMM transmission probability T_{i->j}(omega) for every ordered material pair.

    Returns a dict mapping ``(mat_i, mat_j)`` → T_ij array of shape (Nw,).

    Uses the branch-resolved projected DOS × |vg| formulation::

        M_i(w) = sum_b DOS_{i,b}(w) * |vg_{i,b}(w)|
        T_{i→j}(w) = M_j(w) / (M_i(w) + M_j(w))

    Transmission probability is clamped to [0, 1]; invalid bins (zero denominator)
    default to T = 0.5 (equal-probability fallback).
    """
    n = len(specs)
    tables: dict[tuple[int, int], np.ndarray] = {}
    # Precompute M(w) for every material once.
    M_list: list[np.ndarray] = []
    for i in range(n):
        DOS_b = np.maximum(np.asarray(specs[i]["DOS_w_b"], dtype=np.float64), 0.0)   # (B, Nw)
        vg_b = np.abs(np.asarray(specs[i]["vg_w"], dtype=np.float64))                 # (B, Nw)
        M_b = DOS_b * np.maximum(vg_b, 1e-30)  # branch-wise
        M_list.append(M_b.sum(axis=0))          # sum over branches → (Nw,)
    for i in range(n):
        M_i = M_list[i]
        for j in range(n):
            if i == j:
                continue
            M_j = M_list[j]
            denom = np.maximum(M_i + M_j, 1e-30)
            T = M_j / denom
            # Clamp to physically valid range.
            T = np.clip(T, 0.0, 1.0)
            # For bins where both materials have zero DOS, use equal probability.
            zero_mask = (M_i == 0.0) & (M_j == 0.0)
            T[zero_mask] = 0.5
            tables[(i, j)] = T.astype(np.float64)
    return tables


def init_state_energy_multi(
    mesh: dict[str, Any],
    specs: list[dict[str, Any]],
    opts: dict[str, Any],
) -> SimulationState:
    """Initialise particle state for a multi-material simulation.

    Samples particles per material group and concatenates the blocks.
    When ``initial_particles_fixed > 0`` a global E_eff is computed from the
    total energy weight across *all* materials, then each material samples
    proportionally.
    """
    mode_name = str(get_or(opts, "mode", "absolute"))
    initial_particles_fixed = max(0, int(get_or(opts, "initial_particles_fixed", 0)))
    default_T = float(opts["T_init"]) if np.isfinite(get_or(opts, "T_init", np.nan)) else float(get_or(opts, "T0", get_or(opts, "Tref", 300.0)))
    Tcell, Tmeta = load_initial_temperature_field(mesh, opts, default_T)
    default_Tref = float(get_or(opts, "Tref", np.nan))
    if not np.isfinite(default_Tref):
        default_Tref = float(np.mean(Tcell))
    Tref_cell, Tref_meta = load_reference_temperature_field(mesh, opts, default_Tref)
    cell_mat = np.asarray(mesh.get("cell_material_index", np.ones(1, dtype=np.int32)), dtype=np.int32)
    # --- Phase 1: estimate total energy weight across all materials ---
    # We do a dry-run weight calculation to compute a global E_eff when
    # initial_particles_fixed is requested.
    Vc = cell_volumes(mesh)
    af = enhance_factor_array(opts, Vc.size)
    total_energy_weight = 0.0
    material_weights: list[float] = []
    material_cell_ids: list[np.ndarray] = []
    for mi, sp in enumerate(specs):
        cell_ids = np.flatnonzero(cell_mat == (mi + 1)).astype(np.int64) + 1
        material_cell_ids.append(cell_ids)
        if cell_ids.size == 0:
            material_weights.append(0.0)
            continue
        w = _estimate_energy_weight_for_cells(mesh, sp, opts, cell_ids, Tcell, Tref_cell, mode_name)
        material_weights.append(w)
        total_energy_weight += w
    # --- Phase 2: determine global E_eff ---
    # Handle zero-energy-weight edge case (e.g. deviational mode at uniform T).
    fallback_to_absolute = False
    if total_energy_weight <= 0:
        print("[init] total energy weight is zero (uniform T in deviational mode?), "
              "falling back to absolute mode")
        fallback_to_absolute = True
        global_E_eff = float(get_or(opts, "E_eff", 1e-18))
        use_fixed = False
    elif initial_particles_fixed > 0:
        global_E_eff = total_energy_weight / float(initial_particles_fixed)
        use_fixed = True
    else:
        global_E_eff = float(get_or(opts, "E_eff", 1e-18))
        use_fixed = False
    # --- Phase 3: sample each material ---
    blocks: list[ParticleBlock] = []
    total_Nexp = 0.0
    total_Nsp = 0
    id_offset = 0
    for mi, sp in enumerate(specs):
        cell_ids = material_cell_ids[mi]
        if cell_ids.size == 0:
            continue
        mat_opts = dict(opts)
        mat_opts["E_eff"] = global_E_eff
        mat_opts["initial_particles_fixed"] = 0  # we already handled the fixed count
        if fallback_to_absolute:
            mat_opts["mode"] = "absolute"
        p_blk, info = sample_particles_for_cells(
            mesh, sp, mat_opts, cell_ids,
            Tcell[cell_ids - 1],
            Tref_cell[cell_ids - 1] if mode_name.lower() == "deviational" and not fallback_to_absolute else None,
            id_offset,
        )
        if len(p_blk):
            blocks.append(p_blk)
            id_offset = int(p_blk.id.max())
        total_Nexp += info["Nexp_tot"]
        total_Nsp += info["Nsp_tot"]
    E_eff_used = global_E_eff
    if not blocks:
        raise RuntimeError("No particles generated for any material region")
    p = blocks[0]
    for blk in blocks[1:]:
        p = p.append(blk)
    Nc = infer_Nc(mesh)
    Nsp_cell = np.bincount(p.cell.astype(np.int64) - 1, minlength=Nc).astype(np.int64) if len(p) else np.zeros(Nc, dtype=np.int64)
    Vdom = float(Vc.sum())
    info_full = {
        "mode": mode_name,
        "Tref": float(get_or(opts, "Tref", get_or(opts, "T0", 300.0))),
        "Tref_cell": Tref_cell,
        "reference_temperature_meta": Tref_meta,
        "T_init_cell": Tcell,
        "initial_temperature_meta": Tmeta,
        "U_density_mean": total_Nexp * E_eff_used / max(Vdom, REALMIN),
        "U_total": total_Nexp * E_eff_used,
        "Nexp_tot": total_Nexp,
        "Nsp_tot": total_Nsp,
        "E_eff_used": E_eff_used,
        "fixed_target_particles": initial_particles_fixed if use_fixed else 0,
        "Nc": Nc,
        "Vdom": Vdom,
    }
    return SimulationState(
        p=p,
        WE=E_eff_used,
        Wp=E_eff_used,
        Nsp_cell=Nsp_cell,
        enhance_factor=af,
        info=info_full,
    )


def _estimate_energy_weight_for_cells(
    mesh: dict[str, Any],
    spec: dict[str, Any],
    opts: dict[str, Any],
    cell_ids: np.ndarray,
    Tcell: np.ndarray,
    Tref_cell: np.ndarray,
    mode_name: str,
) -> float:
    """Return the total absolute-mode energy |E| of the equilibrium phonon
    distribution in the given cells (dry-run, no particles created)."""
    cell_ids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
    if cell_ids.size == 0:
        return 0.0
    Vc = cell_volumes(mesh)
    af = enhance_factor_array(opts, Vc.size)
    B, Nw = spec["w_mid"].shape
    dw = ensure_2d_dw(spec)
    pref = HBAR * np.asarray(spec["w_mid"], dtype=np.float64) * np.maximum(np.asarray(spec["DOS_w_b"], dtype=np.float64), 0.0) * dw
    total_w = 0.0
    for ci in cell_ids:
        Tc = float(max(Tcell[ci - 1], 1e-12))
        n_cell = bose_occupation(spec["w_mid"], Tc)
        if mode_name.lower() == "deviational":
            Tr = float(max(Tref_cell[ci - 1], 1e-12))
            n_ref = bose_occupation(spec["w_mid"], Tr)
            Wbm = pref * (n_cell - n_ref)
        else:
            Wbm = pref * n_cell
        total_w += float(np.sum(np.abs(Wbm))) * float(Vc[ci - 1]) * float(af[ci - 1])
    return total_w


def _write_dmm_bin_stats(
    output_cfg: dict[str, Any],
    mesh: dict[str, Any],
    dmm_bin_accum: dict[tuple[str, int], dict[str, int]],
    log_fn,
) -> None:
    """Write per-(pair, omega_bin) DMM statistics and log a pair-wise summary."""
    run_dir = Path(str(output_cfg.get("run_dir", "")))
    if not run_dir:
        return
    specs = mesh.get("specs", [])
    dmm_tables = mesh.get("dmm_tables", {})
    w_edges = np.asarray(specs[0]["w_edges"] if specs else [0.0, 1e30], dtype=np.float64)
    w_centers = 0.5 * (w_edges[:-1] + w_edges[1:])
    Nw = w_centers.size

    # Write per-bin CSV.
    header = ["pair", "omega_bin", "omega_center_rad_s", "attempts", "transmit",
              "reflect", "empirical_T", "table_T"]
    rows: list[list] = [header]
    pair_summary: dict[str, dict[str, float]] = {}  # pair_str -> {att, T, R, emp_T, table_T_mean}

    for (pair_str, w_bin), counts in sorted(dmm_bin_accum.items()):
        att = counts.get("att", 0)
        tr = counts.get("T", 0)
        rf = counts.get("R", 0)
        emp_T = tr / max(att, 1)
        # Look up table_T.
        parts = pair_str.split("->")
        table_T = np.nan
        if len(parts) == 2:
            try:
                i, j = int(parts[0]), int(parts[1])
                tbl = dmm_tables.get((i, j))
                if tbl is not None and 0 <= w_bin < tbl.size:
                    table_T = float(tbl[w_bin])
            except (ValueError, IndexError):
                pass
        wc = w_centers[w_bin] if 0 <= w_bin < Nw else np.nan
        rows.append([pair_str, w_bin, f"{wc:.6e}" if np.isfinite(wc) else "",
                     att, tr, rf, f"{emp_T:.6f}", f"{table_T:.6f}" if np.isfinite(table_T) else ""])

        # Aggregate into pair summary.
        if pair_str not in pair_summary:
            pair_summary[pair_str] = {"att": 0, "T": 0, "R": 0, "table_T_sum": 0.0, "bins": 0}
        ps = pair_summary[pair_str]
        ps["att"] += att
        ps["T"] += tr
        ps["R"] += rf
        if np.isfinite(table_T):
            ps["table_T_sum"] += table_T * att  # weight by attempts
            ps["bins"] += att

    write_csv_rows(run_dir / "dmm_bin_stats.txt", rows)

    # Log pair-wise summary.
    log_fn("\n[DMM bin summary]\n")
    for pair_str in sorted(pair_summary.keys()):
        ps = pair_summary[pair_str]
        emp_T = ps["T"] / max(ps["att"], 1)
        avg_table_T = ps["table_T_sum"] / max(ps["bins"], 1)
        log_fn(f"  {pair_str}: attempts={ps['att']} transmit={ps['T']} reflect={ps['R']} "
               f"empirical_T={emp_T:.4f} table_T_mean={avg_table_T:.4f}\n")


def MC_time_loop_BTE(
    mesh: dict[str, Any],
    spec: dict[str, Any] | list[dict[str, Any]],
    opts: dict[str, Any],
    state: SimulationState,
    luts: list[dict[str, Any]] | None = None,
    pp_luts: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, ParticleBlock, dict[str, Any]]:
    Nc = infer_Nc(mesh)
    T0 = float(get_or(opts, "T0", 300.0))
    dt_min = float(get_or(opts, "dt_min", 1e-15))
    dt_max = float(get_or(opts, "dt_max", 1e-10))
    max_steps = int(get_or(opts, "max_steps", 5000))
    alpha_T = float(np.clip(get_or(opts, "T_underrelax", 1.0), 0.0, 1.0))
    scatter_on = bool(get_or(opts, "scatter_on", True))
    reservoir_cfg = dict(get_or(opts, "reservoir", {}))
    reservoir_on = bool(get_or(reservoir_cfg, "enable", True))
    refresh_every = max(1, int(round(get_or(reservoir_cfg, "refresh_every_n_steps", 100))))
    refresh_at_step1 = bool(get_or(reservoir_cfg, "refresh_at_step1", True))
    qvol, qsrc_meta = load_volume_heat_source_field(mesh, opts, 0.0)
    use_volume_map = bool(np.any(qvol != 0))
    output_cfg = prepare_run_output(mesh, opts)
    if output_cfg["enabled"]:
        mesh["heat_flux_monitors"] = output_cfg["monitors"]
    logcfg = dict(get_or(opts, "log", {}))
    log_on = bool(get_or(logcfg, "on", True))
    print_every = int(get_or(logcfg, "print_every", 1))
    to_file = bool(get_or(logcfg, "to_file", False))
    tee_stdout = bool(get_or(logcfg, "tee_stdout", to_file))
    logfile = str(get_or(logcfg, "filename", "mc_log.txt"))
    if to_file and output_cfg["enabled"]:
        logfile = str(Path(output_cfg["run_dir"]) / Path(logfile).name)
    log_handle = open(logfile, "a", encoding="utf-8") if to_file else None

    def log(msg: str) -> None:
        if not log_on:
            return
        if log_handle is None:
            print(msg, end="")
        else:
            if tee_stdout:
                print(msg, end="")
            log_handle.write(msg)
            log_handle.flush()

    conv_cfg = dict(get_or(opts, "conv", {}))
    conv = {
        "enabled": bool(get_or(conv_cfg, "enable", get_or(opts, "stop_when_steady", True))),
        "min_steps": max(1, int(round(get_or(conv_cfg, "min_steps", get_or(opts, "steady_min_steps", max_steps))))),
        "n_consec": max(1, int(round(get_or(conv_cfg, "n_consec", get_or(opts, "steady_streak_need", 3))))),
        "tol_inf": float(get_or(conv_cfg, "tol_inf", get_or(opts, "steady_tol_inf", 5e-2))),
        "tol_l2": float(get_or(conv_cfg, "tol_l2", get_or(opts, "steady_tol_l2", 5e-2))),
        "tol_Enet": float(get_or(conv_cfg, "tol_Enet", 2e-18)),
    }
    consec_ok = 0
    Tstar, Tmeta = initial_temperature_from_state_or_file(state, mesh, opts, T0)
    Tprime = Tstar.copy()
    out = {
        "dt_hist": [],
        "T_inf_hist": [],
        "T_l2_hist": [],
        "pscat_max_hist": [],
        "E_net_hist": [],
        "dU_cells_hist": [],
        "dU_alive_hist": [],
        "resid_hist": [],
        "iface_hist": [],
        "nsteps": 0,
        "converged": False,
        "Temperature_hist": [],
        "initial_temperature": Tstar.copy(),
        "initial_temperature_meta": Tmeta,
        "output_dir": "",
        "output_steps_dir": "",
        "step_history_file": "",
        "heat_flux_monitor_warnings": [],
        "reservoir_refresh_steps": [],
        "initial_mfp_cdf_file": "",
        "initial_scattering_rate_cdf_file": "",
    }
    init_mfp_file = export_initial_mfp_cdf(output_cfg, state, Tstar, opts, spec)
    if init_mfp_file:
        out["initial_mfp_cdf_file"] = init_mfp_file
        log(f"[init] initial MFP CDF exported to {init_mfp_file}\n")
    init_rate_cdf_file = export_initial_scattering_rate_cdf(output_cfg, state, Tstar, opts, spec)
    if init_rate_cdf_file:
        out["initial_scattering_rate_cdf_file"] = init_rate_cdf_file
        log(f"[init] initial total scattering rate CDF exported to {init_rate_cdf_file}\n")
    # --- multi-material normalisation --------------------------------------
    if isinstance(spec, dict):
        _specs: list[dict[str, Any]] = [spec]
        single_mat = True
    else:
        _specs = list(spec)
        single_mat = len(_specs) == 1
    if luts is None:
        LUTS: list[dict[str, Any]] = [build_E_T_lookup(_specs[0], et_lookup_cfg_from_opts(opts))]
    else:
        LUTS = list(luts)
    if pp_luts is None:
        PP_SCAT_LUTS: list[dict[str, Any]] = [build_pp_scattering_T_lookup(_specs[0], opts, tloc_lookup_cfg_from_opts(opts))]
    else:
        PP_SCAT_LUTS = list(pp_luts)
    LUT = LUTS[0]  # primary LUT for backward-compat single-material code paths
    PP_SCAT_LUT = PP_SCAT_LUTS[0]
    Vc = cell_volumes(mesh)
    Toc_step = Tstar.copy()
    last_dt_info: dict[str, Any] = {}
    if reservoir_on and mesh.get("reservoirs") and refresh_at_step1:
        state, res_info = refresh_reservoir_particles(state, mesh, _specs[0] if single_mat else _specs, opts)
        if res_info["refreshed"]:
            Tstar[res_info["cell_ids"] - 1] = res_info["target_temperature_cell"]
            Tprime[res_info["cell_ids"] - 1] = res_info["target_temperature_cell"]
            out["reservoir_refresh_steps"].append(1)
            log(f"[reservoir] step=1 refreshed {res_info['cell_ids'].size} cells | removed={res_info['removed_particles']} added={res_info['added_particles']}\n")
    U_alive_prev = particles_total_energy(state, opts)
    U_cells_prev = float(np.sum(np.asarray(LUT["U_interp"](np.clip(Tstar, LUT["T"][0], LUT["T"][-1])), dtype=np.float64) * Vc))
    log(f"[{time.strftime('%H:%M:%S')}] MC BTE start. Ncells={Nc}, T0={T0:.2f} K, Tinit=[{Tstar.min():.2f}, {Tstar.mean():.2f}, {Tstar.max():.2f}] K\n")
    if qsrc_meta["used_file"]:
        log(
            f"[source] loaded volume heat source from {qsrc_meta['source']} | "
            f"format={qsrc_meta['format']} entries={qsrc_meta['n_entries']} nonzero_cells={qsrc_meta['n_nonzero_cells']} "
            f"| q[min,mean,max]=[{qsrc_meta['q_min']:+.3e}, {qsrc_meta['q_mean']:+.3e}, {qsrc_meta['q_max']:+.3e}] "
            f"| Ptot={qsrc_meta['q_total_W']:+.3e} W\n"
        )
    for step in range(1, max_steps + 1):
        if reservoir_on and mesh.get("reservoirs") and step > 1 and ((step - 1) % refresh_every == 0):
            state, res_info = refresh_reservoir_particles(state, mesh, _specs[0] if single_mat else _specs, opts)
            if res_info["refreshed"]:
                Tstar[res_info["cell_ids"] - 1] = res_info["target_temperature_cell"]
                Tprime[res_info["cell_ids"] - 1] = res_info["target_temperature_cell"]
                out["reservoir_refresh_steps"].append(step)
                log(f"[reservoir] step={step} refreshed {res_info['cell_ids'].size} cells | removed={res_info['removed_particles']} added={res_info['added_particles']}\n")
        r_tau = precompute_relax_times(state, Tstar, opts, _specs) if scatter_on else np.zeros((len(state.p), 6), dtype=np.float64)
        dt, info_dt = step_pick_dt(mesh, _specs[0], opts, state, r_tau)
        last_dt_info = dict(info_dt)
        dt = min(max(dt, dt_min), dt_max)
        out["dt_hist"].append(dt)
        newpV_count = 0
        if use_volume_map:
            newpV = spawn_volume_sources_from_map(qvol, opts, mesh, _specs, state, Tprime, LUTS if not single_mat else LUT, dt)
            newpV_count = len(newpV)
            if len(newpV):
                state.p = state.p.append(newpV)
        state, fly_stats = particle_fly(state, mesh, dt, opts, _specs[0])
        output_cfg = accumulate_output(output_cfg, fly_stats, dt)
        scatter_stats = {"hits": 0, "hits_expected": 0.0, "pp": 0, "pi": 0, "pb": 0}
        if scatter_on and len(state.p):
            r_tau = precompute_relax_times(state, Tstar, opts, _specs)
            Toc_step = update_pp_scattering_temperature_from_energy(state, mesh, _specs[0], opts, r_tau, PP_SCAT_LUT)[0]
            state, scatter_stats = particle_scattering(state, mesh, _specs, opts, dt, Tstar, r_tau, PP_SCAT_LUT, Toc_step)
        elif scatter_on:
            Toc_step = update_pp_scattering_temperature_from_energy(state, mesh, _specs[0], opts, r_tau, PP_SCAT_LUT)[0]
        else:
            Toc_step = Tstar.copy()
        Tprime, temp_aux = update_temperature_from_energy(state, mesh, _specs, opts, LUTS if not single_mat else LUT)
        E_net_total = 0.0
        U_alive_now = particles_total_energy(state, opts)
        U_cells_now = float(np.sum(np.asarray(LUT["U_interp"](np.clip(Tprime, LUT["T"][0], LUT["T"][-1])), dtype=np.float64) * Vc))
        dU_cells = U_cells_now - U_cells_prev
        dU_alive = U_alive_now - U_alive_prev
        resid = dU_cells - dU_alive
        dT = Tprime - Tstar
        T_inf = float(np.linalg.norm(dT, ord=np.inf))
        T_l2 = float(np.linalg.norm(dT) / math.sqrt(max(Tprime.size, 1)))
        pscat_max = float(info_dt.get("p_scat_max", np.nan))
        out["E_net_hist"].append(E_net_total)
        out["dU_cells_hist"].append(dU_cells)
        out["dU_alive_hist"].append(dU_alive)
        out["resid_hist"].append(resid)
        out["T_inf_hist"].append(T_inf)
        out["T_l2_hist"].append(T_l2)
        out["pscat_max_hist"].append(pscat_max)
        out["Temperature_hist"].append(Tprime.copy())
        out["iface_hist"].append({
            "dmm_attempt": fly_stats.get("dmm_attempt", 0),
            "dmm_transmit": fly_stats.get("dmm_transmit", 0),
            "dmm_reflect": fly_stats.get("dmm_reflect", 0),
            "dmm_energy_attempt": fly_stats.get("dmm_energy_attempt", 0.0),
            "dmm_energy_transmit": fly_stats.get("dmm_energy_transmit", 0.0),
            "dmm_energy_reflect": fly_stats.get("dmm_energy_reflect", 0.0),
            "dmm_detail": fly_stats.get("dmm_detail", {}),
            "dmm_bin_detail": fly_stats.get("dmm_bin_detail", {}),
        })
        if log_on and (step == 1 or step % print_every == 0):
            log(
                f"[step {step:6d}] "
                f"Np={len(state.p):7d} | "
                f"T=[{temp_aux['Tmin']:.2f}, {temp_aux['Tmean']:.2f}, {temp_aux['Tmax']:.2f}] K | "
                f"catch={int(fly_stats.get('catch_count', 0))} "
                f"generate={int(fly_stats.get('generate_count', 0))} "
                f"src_new={int(newpV_count)} | "
                f"scat={int(scatter_stats.get('hits', 0))} "
                f"(PP={int(scatter_stats.get('pp', 0))}, PI={int(scatter_stats.get('pi', 0))}, PB={int(scatter_stats.get('pb', 0))}) | "
                f"dmm=[att={int(fly_stats.get('dmm_attempt', 0))} "
                f"T={int(fly_stats.get('dmm_transmit', 0))} "
                f"R={int(fly_stats.get('dmm_reflect', 0))}] | "
                f"dt={float(info_dt.get('dt', dt)):1.3e}\n"
            )
            if temp_aux["clip_low_count"] or temp_aux["clip_high_count"]:
                log(
                    f"           clip_U_to_T: low={temp_aux['clip_low_count']} "
                    f"high={temp_aux['clip_high_count']} "
                    f"(Tlut={temp_aux['Tlut_min']:.2f}..{temp_aux['Tlut_max']:.2f} K)\n"
                )
            log(
                f"           dT_inf={T_inf:1.3e} dT_L2={T_l2:1.3e} "
                f"dUc={dU_cells:+.3e} dUa={dU_alive:+.3e} resid={resid:+.3e} "
                f"dt_cfl={float(info_dt.get('dt_cfl', np.nan)):1.3e} "
                f"dt_scat={float(info_dt.get('dt_prob', np.nan)):1.3e}\n"
            )
        if output_cfg["enabled"] and (step % output_cfg["every_n_steps"] == 0):
            write_periodic_output(output_cfg, mesh, spec, state, Tprime, Toc_step, opts, info_dt, step, output_cfg["cum_time"], dt)
            output_cfg = reset_output_interval(output_cfg)
        Tstar = (1.0 - alpha_T) * Tstar + alpha_T * Tprime if alpha_T < 1.0 else Tprime.copy()
        U_alive_prev = U_alive_now
        U_cells_prev = U_cells_now
        out["nsteps"] = step
        pass_now = T_inf <= conv["tol_inf"] and T_l2 <= conv["tol_l2"] and abs(E_net_total) <= conv["tol_Enet"]
        consec_ok = consec_ok + 1 if pass_now else 0
        if conv["enabled"] and step >= conv["min_steps"] and consec_ok >= conv["n_consec"]:
            out["converged"] = True
            log(f"[{time.strftime('%H:%M:%S')}] Converged at step {step}: dT_inf={T_inf:1.3e}, dT_L2={T_l2:1.3e}, E_net={E_net_total:+.3e}\n")
            break
    # --- accumulate DMM bin stats across all steps ---
    dmm_bin_accum: dict[tuple[str, int], dict[str, int]] = {}
    for h in out.get("iface_hist", []):
        for (pair_str, w_bin), counts in h.get("dmm_bin_detail", {}).items():
            key = (pair_str, w_bin)
            if key not in dmm_bin_accum:
                dmm_bin_accum[key] = {"att": 0, "T": 0, "R": 0}
            for kk in ("att", "T", "R"):
                dmm_bin_accum[key][kk] += counts.get(kk, 0)
    if output_cfg["enabled"]:
        if output_cfg["interval_time"] > 0 and out["nsteps"] > 0:
            write_periodic_output(output_cfg, mesh, spec, state, Tprime, Toc_step, opts, last_dt_info, out["nsteps"], output_cfg["cum_time"], out["dt_hist"][-1])
        # --- write DMM bin stats ---
        if dmm_bin_accum:
            _write_dmm_bin_stats(output_cfg, mesh, dmm_bin_accum, log)
        out["output_dir"] = output_cfg["run_dir"]
        out["output_steps_dir"] = output_cfg["steps_dir"]
        out["step_history_file"] = output_cfg["step_history_file"]
        out["heat_flux_monitor_warnings"] = output_cfg["monitor_warnings"]
    out["dmm_bin_accum"] = dmm_bin_accum  # make accessible for test scripts
    if log_handle is not None:
        log_handle.close()
    return Tprime, state.p, out


def MC_solve_BTE(cs: dict[str, Any], mat: dict[str, Any], opts: dict[str, Any] | None = None) -> tuple[np.ndarray, ParticleBlock, dict[str, Any]]:
    if opts is None:
        opts = mc_default_opts()
    np.random.seed(int(get_or(opts, "mc_seed", 20240511)))
    mesh = init_mesh_from_geom(cs)
    material_library = mat.get("material_library")
    if material_library is not None:
        mesh["material_library"] = material_library
    opts = resolve_linearization_temperature(mesh, opts)
    # --- multi-material setup ------------------------------------------------
    if material_library is not None and len(material_library.get("list", [])) > 1:
        specs = build_multimaterial_specs(material_library, opts)
        luts = [build_E_T_lookup(s, et_lookup_cfg_from_opts(opts)) for s in specs]
        pp_luts = [build_pp_scattering_T_lookup(s, opts, tloc_lookup_cfg_from_opts(opts)) for s in specs]
        mesh["specs"] = specs
        mesh["n_materials"] = len(specs)
        print(f"[init] T0={float(opts['T0']):.2f} K | {len(specs)} materials: "
              f"{[s['material_key'] for s in specs]} | building initial particles")
        state = init_state_energy_multi(mesh, specs, opts)
        print(
            f"[init] particles={len(state.p)} | mode={state.info.get('mode', 'unknown')} "
            f"| Eeff={float(state.WE):.3e} J"
        )
        # Precompute DMM tables for all material pairs.
        dmm_tables = precompute_dmm_tables(specs)
        mesh["dmm_tables"] = dmm_tables
        return MC_time_loop_BTE(mesh, specs, opts, state, luts, pp_luts)
    # --- single-material path (backward compatible) --------------------------
    spec = build_spectral_grid(mat, opts)
    print(f"[init] T0={float(opts['T0']):.2f} K | building initial particles")
    state = init_state_energy(mesh, spec, opts)
    print(
        f"[init] particles={len(state.p)} | mode={state.info.get('mode', 'unknown')} "
        f"| Eeff={float(state.WE):.3e} J"
    )
    return MC_time_loop_BTE(mesh, spec, opts, state)


def run_current_case(run_tag: str | None = None, base_dir: str | Path | None = None) -> dict[str, Any]:
    base_dir = resolve_base_dir(base_dir)
    input_dir = resolve_input_dir(base_dir)
    if run_tag is None:
        run_tag = time.strftime("%Y%m%d_%H%M%S")
    cs = setup_case_from_ldg_lgrid(input_dir / "ldg.txt", input_dir / "lgrid.txt", length_scale=1e-6, input_length_unit="um", verbose=True)
    mat = resolve_case_material(cs, input_dir=input_dir)
    opts = mc_default_opts(base_dir)
    opts["viz"]["enable"] = False
    opts["log"]["on"] = True
    opts["log"]["to_file"] = True
    opts["log"]["filename"] = "mc_log.txt"
    opts["log"]["print_every"] = 10
    opts["output"]["run_tag"] = run_tag
    Tp, p, out = MC_solve_BTE(cs, mat, opts)
    if out.get("output_dir"):
        write_csv_rows(Path(out["output_dir"]) / "final_summary.txt", [["steps", out["nsteps"]], ["converged", int(bool(out["converged"]))], ["reservoir_refresh_steps", str(out["reservoir_refresh_steps"])], ["Np", len(p)], ["Tmin_K", float(np.min(Tp))], ["Tmean_K", float(np.mean(Tp))], ["Tmax_K", float(np.max(Tp))]])
    print(f"FINAL_OK steps={out['nsteps']} converged={int(bool(out['converged']))} refreshes={out['reservoir_refresh_steps']} Np={len(p)} Tmin={float(np.min(Tp)):.6f} Tmean={float(np.mean(Tp)):.6f} Tmax={float(np.max(Tp)):.6f} output={out.get('output_dir', '')}")
    return {"Tp": Tp, "p": p, "out": out}


def main(base_dir: str | Path | None = None) -> dict[str, Any]:
    base_dir = resolve_base_dir(base_dir)
    input_dir = resolve_input_dir(base_dir)
    print("*****************************************************")
    print("*          BTE Monte Carlo Simulator V1.0           *")
    print("*    Developed by Chenglin Ye, Peking University    *")
    print("*           Python Port for Codex, 2026             *")
    print("*****************************************************")
    cs = setup_case_from_ldg_lgrid(input_dir / "ldg.txt", input_dir / "lgrid.txt", length_scale=1e-6, input_length_unit="um", verbose=True)
    mat = resolve_case_material(cs, input_dir=input_dir)
    opts = mc_default_opts(base_dir)
    Tp, p, out = MC_solve_BTE(cs, mat, opts)
    return {"Tp": Tp, "p": p, "out": out}


__all__ = [
    "ParticleBlock",
    "SimulationState",
    "MC_solve_BTE",
    "MC_time_loop_BTE",
    "build_E_T_lookup",
    "build_pp_scattering_T_lookup",
    "build_q_T_lookup",
    "build_spectral_grid",
    "init_mesh_from_geom",
    "init_state_energy",
    "load_heat_flux_monitors",
    "load_initial_temperature_field",
    "load_reference_temperature_field",
    "load_volume_heat_source_field",
    "main",
    "mat_from_phonon_dispersion_file",
    "mat_igzo",
    "mat_silicon_100",
    "mc_default_opts",
    "particle_fly",
    "particle_scattering",
    "precompute_relax_times",
    "prepare_run_output",
    "refresh_reservoir_particles",
    "resolve_case_material",
    "resolve_case_materials",
    "run_current_case",
    "sample_particles_for_cells",
    "setup_case_from_ldg_lgrid",
    "build_multimaterial_specs",
    "precompute_dmm_tables",
    "init_state_energy_multi",
    "register_material_alias",
    "material_key",
    "load_material",
    "spawn_heat_source",
    "step_pick_dt",
    "update_pp_scattering_temperature_from_energy",
    "update_temperature_from_energy",
]
