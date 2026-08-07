"""Run production runtime.json A/B for section_solve_n_workers (measurement helper).

Usage (repo root)::

    python blade_precompute/section_shell_model/benchmarks/run_runtime_ab.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
# Repo root only when run as a script (pytest already sets pythonpath via pyproject).
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import main_precompute as mp  # noqa: E402
from blade_precompute.orchestration.precompute import PrecomputeJobConfig, run_precompute_pipeline  # noqa: E402


def _base_config(out_base: Path) -> PrecomputeJobConfig:
    return PrecomputeJobConfig(
        data_dir=mp.DATA_DIR,
        output_base_dir=out_base,
        n_geometry=mp.N_GEOMETRY,
        n_structural=mp.N_STRUCTURAL,
        section_plot_station_spec="root,mid,tip",
        n_beam_nodes=mp.N_BEAM_NODES,
        n_beam_png_span_samples=mp.N_BEAM_PNG_SPAN_SAMPLES,
        run_section_shell_model=mp.RUN_SECTION_SHELL_MODEL,
        section_shell_n_elements_per_panel=mp.SECTION_SHELL_N_ELEMENTS_PER_PANEL,
        section_mesh_target_element_length_frac=mp.SECTION_MESH_TARGET_ELEMENT_LENGTH_FRAC,
        section_mesh_min_elements_per_panel=mp.SECTION_MESH_MIN_ELEMENTS_PER_PANEL,
        section_shell_dpi=mp.SECTION_SHELL_DPI,
        save_section_recovery_cache_npz=mp.SAVE_SECTION_RECOVERY_CACHE_NPZ,
        enable_shell_recovery_enrichment=mp.ENABLE_SHELL_RECOVERY_ENRICHMENT,
        shell_recovery_n_elements_per_panel=mp.SHELL_RECOVERY_N_ELEMENTS_PER_PANEL,
        system_type=mp.SYSTEM_TYPE,
        component_materials=mp.COMPONENT_MATERIALS,
        subcomponent_material_ids=mp.SUBCOMPONENT_MATERIAL_IDS,
        execution_mode=mp.EXECUTION_MODE,
        analysis_t_skin_root_mm=mp.ANALYSIS_T_SKIN_ROOT_MM,
        analysis_t_skin_tip_mm=mp.ANALYSIS_T_SKIN_TIP_MM,
        analysis_t_cap_root_mm=mp.ANALYSIS_T_CAP_ROOT_MM,
        analysis_t_cap_tip_mm=mp.ANALYSIS_T_CAP_TIP_MM,
        analysis_t_web_root_mm=mp.ANALYSIS_T_WEB_ROOT_MM,
        analysis_t_web_tip_mm=mp.ANALYSIS_T_WEB_TIP_MM,
        design_optimise=mp.DESIGN_OPTIMISE,
        design_objective=mp.DESIGN_OBJECTIVE,
        design_max_iter=mp.DESIGN_MAX_ITER,
        ks_rho=mp.KS_RHO,
        optimise_design_bias=mp.OPTIMISE_DESIGN_BIAS,
        section_solve_n_workers_override=None,
        seed_design_from_section_properties=mp.SEED_DESIGN_FROM_SECTION_PROPERTIES,
        log_dump_level="summary",
        section_optim_dump_iteration_npz=False,
        section_optim_iteration_hotspot_k=mp.SECTION_OPTIM_ITERATION_HOTSPOT_K,
        section_optim_emit_iteration_schema=False,
        iteration_pipeline_snapshots=False,
        iteration_snapshot_dpi=mp.ITERATION_SNAPSHOT_DPI,
        iteration_snapshot_pngs=False,
        iteration_snapshot_max=mp.ITERATION_SNAPSHOT_MAX,
        iteration_snapshot_stride=mp.ITERATION_SNAPSHOT_STRIDE,
        iteration_field_history=False,
        enable_panel_buckling=mp.ENABLE_PANEL_BUCKLING,
        ks_rho_buckling=mp.KS_RHO_BUCKLING,
        enable_interlaminar=mp.ENABLE_INTERLAMINAR,
        enable_global_buckling=mp.ENABLE_GLOBAL_BUCKLING,
        global_buckling_lambda_min=mp.GLOBAL_BUCKLING_LAMBDA_MIN,
        n_global_buckling_modes=mp.N_GLOBAL_BUCKLING_MODES,
        orientation_bounds=mp.ORIENTATION_BOUNDS,
        n_half_min_skin=mp.N_HALF_MIN_SKIN,
        n_half_max_skin=mp.N_HALF_MAX_SKIN,
        n_biax_min_skin=mp.N_BIAX_MIN_SKIN,
        stress_driven_ply_addition=False,
        stress_driven_n_ply_max=mp.STRESS_DRIVEN_N_PLY_MAX,
        stress_driven_n_ply_min=mp.STRESS_DRIVEN_N_PLY_MIN,
        stress_driven_refine_thickness=mp.STRESS_DRIVEN_REFINE_THICKNESS,
        enforce_monotone_thickness=mp.ENFORCE_MONOTONE_THICKNESS,
        debug_stress_projection=mp.DEBUG_STRESS_PROJECTION,
        u_inf=mp.U_INF,
        tsr=mp.TSR,
        azimuth_deg=mp.AZIMUTH_DEG,
        gravity_m_s2=mp.GRAVITY_M_S2,
        axial_loading_enabled=mp.AXIAL_LOADING_ENABLED,
        section_loads_pitch_deg=mp.SECTION_LOADS_PITCH_DEG,
        section_loads_inflow_phi_deg=mp.SECTION_LOADS_INFLOW_PHI_DEG,
        section_loads_rho=mp.SECTION_LOADS_RHO,
        xfoil_executable=mp.XFOIL_EXECUTABLE,
        beam_driver=mp.BEAM_DRIVER,
        shell_theory=mp.SHELL_THEORY,
        shell_n_elements_per_panel=mp.SHELL_N_ELEMENTS_PER_PANEL,
        shell_mesh_target_element_length_frac=mp.SHELL_MESH_TARGET_ELEMENT_LENGTH_FRAC,
        shell_mesh_min_elements_per_panel=mp.SHELL_MESH_MIN_ELEMENTS_PER_PANEL,
        design_optimizer_method=mp.DESIGN_OPTIMIZER_METHOD,
        design_optimizer_ftol=mp.DESIGN_OPTIMIZER_FTOL,
        design_optimizer_n_restarts=mp.DESIGN_OPTIMIZER_N_RESTARTS,
        design_optimizer_multistart_seed=mp.DESIGN_OPTIMIZER_MULTISTART_SEED,
    )


def _shell_keys(stage_seconds: dict[str, float]) -> dict[str, float]:
    keys = (
        "shell_runtime_prepass_s",
        "section_shell_model_s",
        "section_optimisation_s",
        "section_geometry_s",
        "section_properties_s",
    )
    return {k: float(stage_seconds.get(k, 0.0)) for k in keys}


def main() -> int:
    out_base = _REPO / "blade_precompute" / "section_shell_model" / "benchmarks" / "output" / "runtime_ab"
    out_base.mkdir(parents=True, exist_ok=True)

    workers_list = (1, 4)
    results: dict[str, object] = {"n_structural": mp.N_STRUCTURAL, "runs": {}}

    for n_workers in workers_list:
        cfg = replace(_base_config(out_base), section_solve_n_workers_override=n_workers)
        print(f"\n=== A/B run: section_solve_n_workers_override={n_workers} ===", flush=True)
        result = run_precompute_pipeline(cfg)
        if result.exit_code != 0:
            print(f"Run failed with exit_code={result.exit_code}", flush=True)
            return result.exit_code
        shell = _shell_keys(result.stage_seconds)
        shell_combined = shell["shell_runtime_prepass_s"] + shell["section_shell_model_s"]
        results["runs"][str(n_workers)] = {
            "job_dir": str(result.job_dir),
            "stage_seconds": dict(result.stage_seconds),
            "shell_combined_s": shell_combined,
        }
        print(f"job={result.job_dir.name} shell_combined={shell_combined:.2f}s", flush=True)

    serial = float(results["runs"]["1"]["shell_combined_s"])  # type: ignore[index]
    parallel = float(results["runs"]["4"]["shell_combined_s"])  # type: ignore[index]
    results["speedup_shell_combined"] = serial / parallel if parallel > 0 else None

    out_json = out_base / "runtime_ab_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")
    print(f"Speedup (shell prepass + section_shell_model): {results['speedup_shell_combined']:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
