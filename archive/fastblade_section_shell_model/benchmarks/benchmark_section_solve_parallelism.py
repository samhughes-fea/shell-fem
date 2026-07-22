"""Benchmark multi-station shell parallelism and optional follow-up profiling.

Compares Path C station solve wall time with ``ProcessPoolExecutor`` workers
=1 vs >1 using picklable tasks (Windows ``spawn`` safe). Profiles topology-cache
churn during simulated design iterations and records deferred perf-item assessment.

Run from repo root::

    python blade_precompute/section_shell_model/benchmarks/benchmark_section_solve_parallelism.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
# Repo root only when run as a script (pytest already sets pythonpath via pyproject).
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from blade_precompute.materials.laminate import LaminateDefinition
from blade_precompute.materials.ply import OrthotropicPly
from blade_precompute.section_definition.types import SectionDefinition, SubcomponentGeometry
from blade_precompute.section_optimisation.core.types import OptimBladeGeometry
from blade_precompute.section_optimisation.engine import shell_fi_eval


def _ply() -> OrthotropicPly:
    return OrthotropicPly(
        name="g",
        E1=30e9,
        E2=8e9,
        G12=3e9,
        nu12=0.3,
        rho=1.5e3,
        t_ply=2e-4,
        Xt=500e6,
        Xc=300e6,
        Yt=50e6,
        Yc=200e6,
        S12=40e6,
        Zt=1e6,
        S13=12e6,
        S23=1e6,
    )


def _multi_station_setup(*, n_stations: int = 4) -> tuple[list[SectionDefinition], np.ndarray, OptimBladeGeometry]:
    from blade_utilities.sdf.airfoil import AirfoilSDF

    z = np.linspace(0.0, 1.0, int(n_stations), dtype=np.float64)
    lam = LaminateDefinition(plies=[(_ply(), 0.0), (_ply(), 45.0)])
    line = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    skin = SubcomponentGeometry(name="skin", midsurface_coords=line, material=lam, thickness=0.002)
    section_defs = [
        SectionDefinition(station_z=float(zi), subcomponents=[skin]) for zi in z
    ]
    prof = np.asarray(AirfoilSDF.from_naca("2412", n_points=60, chord=1.0).vertices)[:, :2]
    bg = OptimBladeGeometry(
        z_stations=z,
        r_ref=np.zeros((n_stations, 3)),
        kappa0=np.zeros((n_stations, 3)),
        chord=np.linspace(1.0, 0.9, n_stations),
        twist=np.zeros(n_stations),
        airfoil_profiles=[prof] * n_stations,
        web_positions=np.array([0.35]),
        subcomponent_materials={},
    )
    r_sec = np.zeros((n_stations, 7))
    r_sec[:, 4] = 5.0e3
    r_sec[:, 5] = 2.0e3
    return section_defs, r_sec, bg


def _clear_shell_caches() -> None:
    from blade_precompute.section_shell_model.engine.periodic_cache import clear_periodic_caches

    clear_periodic_caches()
    shell_fi_eval._PIPELINE_B_CACHE.clear()
    shell_fi_eval._TAPERED_PAIR_CACHE.clear()


def _benchmark_station_solve_task(
    *,
    station_index: int,
    n_elements_per_panel: int,
    load_scale: float,
) -> float:
    """Picklable worker task: one Path C station solve with pre-seeded topology cache."""
    from blade_precompute.section_shell_model.api.pipeline_b import build_pipeline_b_section

    section_defs, _, bg = _multi_station_setup(n_stations=4)
    sdef = section_defs[int(station_index)]
    pb = build_pipeline_b_section("2412", [0.35], chord_m=float(bg.chord[int(station_index)]))
    topo_key = shell_fi_eval._station_topology_key(sdef, bg, int(station_index))
    shell_fi_eval._PIPELINE_B_CACHE[topo_key] = pb

    t0 = time.perf_counter()
    shell_fi_eval._run_section_shell_solve(
        sdef=sdef,
        bg=bg,
        s_idx=int(station_index),
        airfoil=np.asarray(bg.airfoil_profiles[int(station_index)], dtype=np.float64),
        spars=[0.85],
        skin=_ply(),
        loads={
            "N": 0.0,
            "Vy": float(load_scale) * 5.0e3,
            "Vz": float(load_scale) * 2.0e3,
            "My": 0.0,
            "Mz": 0.0,
            "T": 0.0,
        },
        n_elements_per_panel=int(n_elements_per_panel),
        target_element_length_m=None,
        min_elements_per_panel=2,
    )
    return time.perf_counter() - t0


def _timed_parallel_station_solves(
    *,
    n_workers: int,
    n_stations: int,
    n_elements_per_panel: int,
    repeats: int,
) -> dict[str, Any]:
    station_indices = list(range(int(n_stations)))
    times: list[float] = []
    for rep in range(max(1, int(repeats))):
        _clear_shell_caches()
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=max(1, int(n_workers))) as ex:
            futs = [
                ex.submit(
                    _benchmark_station_solve_task,
                    station_index=i,
                    n_elements_per_panel=n_elements_per_panel,
                    load_scale=1.0 + 0.01 * rep,
                )
                for i in station_indices
            ]
            for fut in as_completed(futs):
                fut.result()
        times.append(time.perf_counter() - t0)
    return {
        "n_workers": int(n_workers),
        "n_stations": int(n_stations),
        "n_elements_per_panel": int(n_elements_per_panel),
        "repeats": int(repeats),
        "wall_s_mean": round(float(np.mean(times)), 6),
        "wall_s_min": round(float(np.min(times)), 6),
        "wall_s_max": round(float(np.max(times)), 6),
    }


@dataclass
class TopologyChurnReport:
    n_iterations: int
    full_station_key_changes: int
    single_subcomponent_changes: int
    partial_invalidation_warranted: bool
    notes: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_iterations": self.n_iterations,
            "full_station_key_changes": self.full_station_key_changes,
            "single_subcomponent_changes": self.single_subcomponent_changes,
            "partial_invalidation_warranted": self.partial_invalidation_warranted,
            "notes": self.notes,
            **self.extra,
        }


def profile_station_topology_cache_churn(*, n_iterations: int = 24) -> TopologyChurnReport:
    """Simulate SLSQP-like thickness updates and count topology-key churn."""
    section_defs, _, bg = _multi_station_setup(n_stations=4)
    sdef = section_defs[0]
    keys_seen: set[tuple] = set()
    full_changes = 0
    single_sub_changes = 0

    rng = np.random.default_rng(42)
    base_thickness = {sub.name: float(sub.thickness) for sub in sdef.subcomponents}

    lam_cap = LaminateDefinition(plies=[(_ply(), 0.0)])
    cap = SubcomponentGeometry(
        name="spar_cap",
        midsurface_coords=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        material=lam_cap,
        thickness=0.004,
    )
    multi_sub = SectionDefinition(station_z=sdef.station_z, subcomponents=[*sdef.subcomponents, cap])

    for it in range(int(n_iterations)):
        trial_base = multi_sub if it % 2 == 0 else sdef
        if it % 3 == 0:
            delta = rng.uniform(-0.0002, 0.0002, size=len(trial_base.subcomponents))
            new_thickness = {
                sub.name: max(1e-4, float(sub.thickness) + float(delta[i]))
                for i, sub in enumerate(trial_base.subcomponents)
            }
            mode = "all"
        else:
            pick = int(it % len(trial_base.subcomponents))
            new_thickness = {sub.name: float(sub.thickness) for sub in trial_base.subcomponents}
            sub = trial_base.subcomponents[pick]
            new_thickness[sub.name] = max(1e-4, base_thickness.get(sub.name, sub.thickness) + rng.uniform(-0.0001, 0.0001))
            mode = "single"

        trial = SectionDefinition(
            station_z=trial_base.station_z,
            subcomponents=[
                SubcomponentGeometry(
                    name=sub.name,
                    midsurface_coords=sub.midsurface_coords,
                    material=sub.material,
                    thickness=new_thickness[sub.name],
                )
                for sub in trial_base.subcomponents
            ],
        )
        key = shell_fi_eval._station_topology_key(trial, bg, 0)
        if key not in keys_seen:
            keys_seen.add(key)
            if mode == "all":
                full_changes += 1
            else:
                single_sub_changes += 1

    single_frac = single_sub_changes / max(n_iterations, 1)
    warranted = single_frac > 0.75 and full_changes <= 2
    notes = (
        "Station-level topology keys are sufficient for production optimiser loops: "
        "SLSQP updates multiple thickness DVs per iteration, so partial panel "
        "invalidation would rarely hit. Revisit only if profiling shows "
        "single-subcomponent dirty-station edits dominate wall time."
    )
    if warranted:
        notes = (
            "Single-subcomponent edits dominate this synthetic workload; consider "
            "panel-level partial invalidation if assembly patch cost is verified."
        )

    return TopologyChurnReport(
        n_iterations=int(n_iterations),
        full_station_key_changes=int(full_changes),
        single_subcomponent_changes=int(single_sub_changes),
        partial_invalidation_warranted=bool(warranted),
        notes=notes,
        extra={"unique_topology_keys": len(keys_seen), "single_subcomponent_fraction": round(single_frac, 4)},
    )


def assess_deferred_perf_items() -> dict[str, Any]:
    """Record assessment of open PERF report findings (#8–9, HSDT Gauss)."""
    return {
        "finding_8_cusp_penalty_analytical_derivatives": {
            "status": "closed_deferred",
            "warranted": False,
            "reason": "Full-solve spine + cusp penalty deleted in R3 (2026-07-14); never on production periodic FI path.",
        },
        "finding_9_legacy_panel_local_bc_lil": {
            "status": "deferred",
            "warranted": False,
            "reason": "Legacy panel-local path only (panel_mitc4_model._apply_bcs); production Path C uses SectionLinearStaticRunner.",
        },
        "hsdt_element_gauss_batching": {
            "status": "deferred",
            "warranted": False,
            "reason": "HSDT not on optimiser hot path (FSDT periodic default); MITC4 Gauss already vectorised.",
        },
    }


def benchmark_shell_fi_parallelism(
    *,
    n_stations: int = 4,
    n_elements_per_panel: int = 4,
    serial_workers: int = 1,
    parallel_workers: int | None = None,
    repeats: int = 2,
) -> dict[str, Any]:
    """Compare parallel Path C station solves serial vs parallel worker fan-out."""
    cpu = os.cpu_count() or 2
    par_workers = int(parallel_workers if parallel_workers is not None else min(4, cpu, n_stations))

    serial = _timed_parallel_station_solves(
        n_workers=serial_workers,
        n_stations=n_stations,
        n_elements_per_panel=n_elements_per_panel,
        repeats=repeats,
    )
    parallel = _timed_parallel_station_solves(
        n_workers=par_workers,
        n_stations=n_stations,
        n_elements_per_panel=n_elements_per_panel,
        repeats=repeats,
    )
    speedup = serial["wall_s_mean"] / max(parallel["wall_s_mean"], 1e-12)
    return {
        "cpu_count": cpu,
        "serial": serial,
        "parallel": parallel,
        "speedup": round(float(speedup), 4),
        "parallel_faster": bool(speedup >= 1.05),
        "topology_cache_churn": profile_station_topology_cache_churn().to_dict(),
        "deferred_perf_assessment": assess_deferred_perf_items(),
    }


def write_benchmark_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    manifest = benchmark_shell_fi_parallelism()
    out = Path("blade_precompute/section_shell_model/benchmarks/output/section_solve_parallelism.json")
    write_benchmark_manifest(manifest, out)
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {out.resolve()}")


if __name__ == "__main__":
    main()
