# `section_shell_model` vs 1d-fem: computational architecture alignment

**Status:** evaluation (new doc)  
**Last updated:** 2026-07-08

This document evaluates `section_shell_model`'s organisational philosophy against `1d-fem` (`C:\Users\s1834431\Projects\1d-fem`), the reference project whose layer decomposition inspired [PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md](PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md), [PROCESSING_ARCHITECTURE_PROPOSAL.md](PROCESSING_ARCHITECTURE_PROPOSAL.md), and [ELEMENT_AGNOSTIC_ARCHITECTURE.md](ELEMENT_AGNOSTIC_ARCHITECTURE.md). All 1d-fem claims below were verified directly against its source tree, not inferred from documentation, to avoid repeating unverified assumptions from earlier drafts of the sibling proposals.

**Related docs**

| Doc | Role |
|-----|------|
| [PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md](PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md) | Pre-solve package layout, HSDT/CCX quarantine, FSDT registry |
| [PROCESSING_ARCHITECTURE_PROPOSAL.md](PROCESSING_ARCHITECTURE_PROPOSAL.md) | Four-tier results pipeline |
| [ELEMENT_AGNOSTIC_ARCHITECTURE.md](ELEMENT_AGNOSTIC_ARCHITECTURE.md) | Element factory registry pattern |

---

## Principle: 1d-fem is a reference for organisational philosophy, not a contract to mirror line-for-line

1d-fem is a 1-D beam/bar FEM framework (Euler-Bernoulli, Timoshenko, Levinson/Reddy higher-order beams, geometrically-exact large-rotation elements) built for a different problem class than `section_shell_model` (2-D composite shell cross-sections, multi-theory station dispatch across a blade span, tapered/periodic boundary conditions, parallel per-station orchestration). The verdicts below are grouped into four categories, and **category D departures are not defects** — they are documented so future contributors don't try to force-fit blade-specific logic into a 1d-fem shape it was never designed for:

- **(A) Direct analogue** — same role, same relative position; align naming/placement where drift exists.
- **(B) Blade-specific extension** — 1d-fem has no equivalent; the extension is warranted by blade_precompute's problem domain and should stay as its own thing, not be renamed to appear more 1d-fem-like.
- **(C) Drift to correct** — 1d-fem *does* have an equivalent, but `section_shell_model` has it in the wrong place or missing entirely; worth fixing (tracked in the sibling proposals).
- **(D) Deliberate departure** — 1d-fem's approach does not fit blade_precompute's needs even though a nominal analogue exists; diverging is the right call, and forcing parity would make the code worse.

---

## Verified 1d-fem facts (source-checked, not doc-derived)

| Claim | Verified against | Finding |
|-------|-------------------|---------|
| `pre_processing/containers/` exists | `pre_processing/containers/{mesh_container,element_container,element_stiffness_matrix_container,element_force_vector_container}.py` | True, but **zero production modules import from this package** (repo-wide grep for `pre_processing.containers` / `pre_processing\.containers` matches nothing outside itself) |
| Active formulation cache lives elsewhere | `processing/static/results/containers/formulation_results.py` | `FormulationResultSet` (`element_objects`, `force_objects`) imports `ElementObject`/`ForceObject` from `pre_processing/element_library/gauss_point_data.py` — **not** from `pre_processing/containers/` |
| Solve-state DTOs live under `processing/`, not `pre_processing/` | `processing/static/results/containers/global_results.py`, `processing/static/operations/*.py` | `GlobalResults` (`K_global`, `F_global`, `K_mod`, `F_mod`, `K_cond`, `F_cond`, `U_cond`, `U_global`) is constructed and mutated entirely inside `LinearStaticSimulationRunner`'s `primary_results_set`; 1d-fem never puts solve-state fields under `pre_processing/` |
| Element registry is a static class-level map, not a runtime `register()` call | `pre_processing/element_library/element_factory.py` | `ElementFactory.ELEMENT_CLASS_MAP = {**LINEAR_ELEMENT_CLASS_MAP, **NONLINEAR_ELEMENT_CLASS_MAP}` — a dict literal mapping element-type strings to dotted module paths, lazily `importlib.import_module`'d per lookup. It is a registry in the "no if/elif" sense, but the registration mechanism is a static dict, not an imperative `register_element_factory()` call |
| `processing/static/diagnostics/` exists | `runtime_monitor_telemetry.py`, `linear_static_diagnostic.py` (`DiagnoseLinearStaticSystem`) | True — direct analogue to blade's `processing/diagnostics/` |
| Runner owns cache + result sets for the whole lifecycle | `simulation_runner/static/linear_static_simulation.py::LinearStaticSimulationRunner` | `self.formulation_cache = FormulationResultSet(...)` in `__init__`; `self.primary_results_set` populated stage-by-stage; `compute_secondary_results()` / `compute_tertiary_results()` are runner methods injecting `self.formulation_cache` |
| Orchestration layer above the runner | `workflow_orchestrator/{run_job,cli,run_manifest,review_job_logs}.py` | Analogue to `section_shell_model/api/` + blade's own `orchestration/precompute/` — job-level driving, not part of the element/results layering itself |

---

## Verdict table

| `section_shell_model` item | Category | 1d-fem counterpart | Verdict |
|------------------------------|----------|---------------------|---------|
| `pre_processing/containers/` (`types.py`, `formulation_objects.py`, `fem_config.py`) | **A** | `pre_processing/element_library/gauss_point_data.py` (active) — **not** 1d-fem's own dormant `pre_processing/containers/` | In step in *role* (agnostic pre-solve DTOs / formulation objects), but the fidelity claim in earlier drafts of [PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md](PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md) pointed at the wrong 1d-fem module as the analogue — now corrected there and in [ELEMENT_AGNOSTIC_ARCHITECTURE.md](ELEMENT_AGNOSTIC_ARCHITECTURE.md) |
| `pre_processing/containers/solve_states.py` (pre-Phase-7 location) | **C, being fixed** | `processing/static/results/containers/global_results.py` + runner-owned state | Drift — 1d-fem never places solve-state DTOs under `pre_processing/`; relocating to `processing/operations/solve_states.py` (Phase 5/7 of the pre-processing proposal) closes this gap |
| `pre_processing/element_library/` | **A** | `pre_processing/element_library/` | Direct — same placement, same role (kernels, shape functions, formulation factory) |
| `resolve_element_factory` registry (Phase 7 landed) | **A / D (mechanism)** | `ElementFactory.ELEMENT_CLASS_MAP` | Landed — dict lookup via `register_element_factory`. Opt-in gate remains for **CCX dispatch** (must not register at import time); former HSDT opt-in backend was **deleted**. Category **D** for the registration *mechanism*, category **A** for the "no branching" *outcome* |
| `pre_processing/mesh/` | **B** | none | Blade-specific — cross-section strip/panel meshing has no 1-D-beam equivalent. Correctly documented as "convention only" in the pre-processing proposal; no further action needed |
| `pre_processing/loading/` (absorbed `section_loads`) | **B** | none (1d-fem takes point/distributed loads as job-config inputs, no aerodynamic derivation layer) | Blade-specific — aerofoil aerodynamics (XFOIL/synthetic incident flow → surface traction) is entirely outside 1d-fem's problem domain. Correctly kept as its own convention-only home |
| `processing/operations/` (`PrepareLocalShellSystem` … `DisassembleGlobalShellField`) | **A** | `processing/static/operations/*.py` (`assembly.py`, `solver.py`, …) | Direct — both decompose the solve into named stage classes; 1d-fem's stages are function-based, blade's are class-based (`.run()` methods), but the staged-verb decomposition principle is identical |
| `processing/diagnostics/` | **A (corrected)** | `processing/static/diagnostics/` (`DiagnoseLinearStaticSystem`, `RuntimeMonitorTelemetry`) | Direct — corrects the original pre-processing proposal's claim that this had no 1d-fem counterpart |
| `processing/results/` four-tier taxonomy | **B** | `processing/static/results/compute_primary/`, `compute_secondary/`, `compute_tertiary/` (three tiers, no quaternary) | Blade-specific extension — ply-level failure-index evaluation (Hashin, Von Mises) has no 1-D-beam analogue (1d-fem's tertiary is section-force/principal-stress recovery, not material failure criteria). Quaternary is a legitimate fourth tier that 1d-fem's problem domain simply doesn't need |
| `simulation_runner/section_linear_static.py::SectionLinearStaticRunner` | **A** | `simulation_runner/static/linear_static_simulation.py::LinearStaticSimulationRunner` | Direct — both own the formulation cache and result sets for the whole solve lifecycle, expose numbered/ordered stage methods, and inject cache + upstream result set into each results tier method. Confirms the runner-as-sole-orchestrator principle in [ELEMENT_AGNOSTIC_ARCHITECTURE.md](ELEMENT_AGNOSTIC_ARCHITECTURE.md) is correctly modelled, not aspirational |
| `engine/` (assembly, constraints, homogenisation, recovery, dispatch) | **D** | No single counterpart — 1d-fem's assembly lives in `processing/static/operations/assembly.py` (a stage op), with no separate "engine" layer | Deliberate departure, **currently over-extended**. Some of `engine/`'s responsibilities (`section_assembly.py::assemble_section_system`) *should* eventually fold into `processing/operations/` to match 1d-fem's stage-op placement (tracked as long-term direction below), but `engine/`'s homogenisation (`panel_abd.py`, `homogenisation.py`) and per-theory station dispatch (`shell_solve_registry.py`, `station_theory_dispatch.py`) have no 1-D-beam equivalent at all — a beam element doesn't need cross-sectional ABD homogenisation or multi-theory station routing. Keeping `engine/` as a named mixed layer for blade-specific concerns is more honest than either (a) forcing it fully into `processing/operations/` where it doesn't fit, or (b) pretending it doesn't exist |
| Per-theory monolith solvers (legacy MITC4 thin-wall / HSDT) bypassing the runner | **A (fixed)** | `LinearStaticSimulationRunner` is the *only* solve path for every element type 1d-fem supports | **Closed** — legacy MITC4 clamped FE deleted; HSDT package deleted. Production routes through periodic FSDT + `SectionLinearStaticRunner`. CCX remains an external validation subprocess (opt-in), not a second native monolith |
| `orchestration/precompute/` (blade span-wise parallel station orchestration) | **B** | `workflow_orchestrator/` (single-job driving, no per-station parallelism) | Blade-specific extension — 1d-fem solves one job at a time; blade_precompute's need to solve tens-to-hundreds of stations per blade in parallel (`shell_runtime.py` worker pool) is a scale problem 1d-fem's single-job orchestrator was never built for. No action needed; do not attempt to fold blade's orchestration into a 1d-fem-shaped single-job runner |
| `use_hsdt_interlaminar` theory-conditional flag chain | **A (fixed)** | 1d-fem has no theory-conditional result-computation flags | **Closed** — flag chain removed; interlaminar FI is theory-independent quaternary Hashin mode ([PROCESSING_ARCHITECTURE_PROPOSAL.md](PROCESSING_ARCHITECTURE_PROPOSAL.md) Phase 6) |

---

## Where `section_shell_model` should keep diverging (do not "fix" these)

Per the note that prompted this document: alignment with 1d-fem is a means, not an end. The following are correct as blade-specific departures and should **not** be reshaped to look more like 1d-fem in a future pass:

1. **`pre_processing/mesh/` and `pre_processing/loading/`** — no 1-D-beam analogue exists or should be invented. These stay named for what they do (cross-section meshing, aerodynamic load derivation), not for 1d-fem placement parity.
2. **The quaternary results tier** — ply-level composite failure criteria are structurally absent from a 1-D beam framework. Bolting a fourth tier onto the three-tier taxonomy is the right extension, not a deviation to apologise for.
3. **`orchestration/precompute/`'s parallel station orchestration** — solving hundreds of stations per blade concurrently is a different orchestration problem than 1d-fem's single-job `workflow_orchestrator/`; no attempt should be made to converge these.
4. **`engine/`'s homogenisation and multi-theory dispatch responsibilities** — cross-sectional ABD homogenisation and per-theory station routing have no beam-element equivalent; only the *assembly* portion of `engine/` is a long-term candidate to fold toward `processing/operations/` (see below), not homogenisation or dispatch.
5. **Registration mechanism (imperative `register_element_factory()` vs. 1d-fem's static `ELEMENT_CLASS_MAP` dict)** — blade's opt-in gating requirement for **CCX dispatch** (must not register at import time) is a real constraint 1d-fem's single-static-dict pattern doesn't need to solve. FSDT registers unconditionally; the former HSDT opt-in backend was deleted. The imperative registration hook remains the correct shape for validation backends.

---

## Where `section_shell_model` should still converge (the fixes already scoped)

| Gap | Fix | Tracked in |
|-----|-----|------------|
| Solve-state DTOs under `pre_processing/containers/` instead of `processing/` | Relocate `solve_states.py` → `processing/operations/solve_states.py` | [PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md](PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md) Phase 5/7 |
| Hard-coded if/elif element factory resolution | Registry map (`register_element_factory` + dict lookup) | [PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md](PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md) Phase 7, [ELEMENT_AGNOSTIC_ARCHITECTURE.md](ELEMENT_AGNOSTIC_ARCHITECTURE.md) |
| Per-theory monolith solvers bypassing the single runner spine (legacy MITC4, HSDT) | **Done** — legacy MITC4 deleted; HSDT package deleted (2026-07-14) | [PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md](PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md) post-completion amendments |
| Theory-conditional result-computation flag (`use_hsdt_interlaminar`) | **Done** — interlaminar is theory-independent Hashin mode | [PROCESSING_ARCHITECTURE_PROPOSAL.md](PROCESSING_ARCHITECTURE_PROPOSAL.md) Phase 6 |
| `processing/diagnostics` fidelity claim (previously said "no 1d-fem counterpart") | Corrected — direct analogue to `processing/static/diagnostics/` | This doc; [PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md](PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md) |
| `pre_processing/containers` fidelity claim (previously pointed at 1d-fem's dormant `pre_processing/containers/`) | Corrected — the real analogue is `pre_processing/element_library/gauss_point_data.py` (the active formulation object cache) | This doc; [PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md](PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md) |
| `engine/` mixed spine — assembly portion only | **Done (R8, 2026-07-14):** `assemble_section_system` lives in `processing/operations/assemble_global.py`; `engine/section_assembly.py` retains `assemble_section_from_panels` / re-exports. Homogenisation and dispatch stay in `engine/` (category D, not drift). | Landed |

---

## Summary

`section_shell_model` is materially in step with 1d-fem's computational philosophy: layer decomposition (pre_processing → processing → simulation_runner → api), staged verb-named operations, a runner that owns the formulation cache and result sets for the whole solve lifecycle and injects them into each results tier, and a factory/registry seam for element kernels. The gaps that exist are narrow and already scoped ([PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md](PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md), [PROCESSING_ARCHITECTURE_PROPOSAL.md](PROCESSING_ARCHITECTURE_PROPOSAL.md)). Equally important, several of blade's largest structural differences from 1d-fem — the quaternary results tier, `pre_processing/mesh` and `pre_processing/loading`, `engine/`'s homogenisation and dispatch responsibilities, and span-wise parallel station orchestration — are not gaps to close but correct responses to a problem domain 1d-fem was never built for. Future passes should keep using 1d-fem as a philosophy reference for *how to decompose a solve pipeline*, not as a literal target shape for every directory.
