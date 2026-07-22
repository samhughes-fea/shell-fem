# section_shell_model — next steps: performance & maintainability

Work-plan context for the 2026 cleanup/perf/maintainability refactor
(facade removal, periodic assembly/K6 caches, batched CLPT Hashin, orchestration parallelisation).
The planned tracks (A1–A4, B1–B2) are **complete** as of 2026-07-07; later SoC (R1–R15),
vendor housekeeping, and the production FI feature track are also **complete**. This document
records what landed; remaining items are parked research or optional polish only.
Use [PERFORMANCE_OPTIMISATION_REPORT.md](PERFORMANCE_OPTIMISATION_REPORT.md) for ranked
findings and deferred items.

**Last updated:** 2026-07-22 — Phase 0 complete (Channel-A / `section_optimisation` removed).
Full-section buckling landed on `SectionLinearBucklingRunner`. Remaining work is
doc/test hygiene and consumer hardening on the linear-static track; see
[LINEAR_STATIC_MAIN_TRACK_PHASE0.md](LINEAR_STATIC_MAIN_TRACK_PHASE0.md).

Tracks A + B, SoC R1–R15, and Channel-A removal are **complete**. Do not schedule
Channel-A reintroduction.

> **Historical ledger:** sections below the architecture baseline retain the
> original A/B/R implementation narrative (Path C, periodic caches, optimiser
> `shell_fi_batch`). Treat them as archaeology. Live product truth is the Phase 0
> memo + [SHELL_MODEL.md](SHELL_MODEL.md) header sections.

### Production baseline gate (2026-07-22)

| Suite | Command intent | Result |
|-------|----------------|--------|
| Package fast gate | `pytest blade_precompute/section_shell_model -m "not slow and not ccx and not legacy"` | **240 passed**, 1 skipped, 2 deselected |
| Orchestration shell / stage order | `pytest blade_precompute/orchestration -k "shell or pipeline_stage"` | **4 passed** |
| Import boundary | no live `section_shell_model.api.periodic` / `section_optimisation` imports | **OK** |
| CCX E2E | `@pytest.mark.ccx` under `calculix_ccx/tests/` | Environment-gated (`ccx` on PATH); not part of the default gate |

Import-boundary note (same day): Defect K harness routes edge traction through
`resolve_kernel(...).recover_edge_shear_traction` so `processing/` does not import
`mitc4.element` directly.

**Related baseline docs**

| Doc | Role |
|-----|------|
| [PERFORMANCE_OPTIMISATION_REPORT.md](PERFORMANCE_OPTIMISATION_REPORT.md) | Ranked perf findings; update status as work lands |
| [README.md](../README.md) | Canonical import paths and pipeline position |
| [SHELL_MODEL.md](SHELL_MODEL.md) | Periodic section solve policy + multi-scale BCs |
| [DEFERRED_RESEARCH.md](DEFERRED_RESEARCH.md) | Out-of-scope policy and deferred backlog |
| [../orchestration/docs/PERFORMANCE_OPTIMISATION_REPORT.md](../../orchestration/docs/PERFORMANCE_OPTIMISATION_REPORT.md) | Cross-package stage timings |
| [../orchestration/docs/PERFORMANCE_MEASUREMENT_DEFERRED_OPTIMISATIONS.md](../../orchestration/docs/PERFORMANCE_MEASUREMENT_DEFERRED_OPTIMISATIONS.md) | Production A/B measurement notes |
| [../benchmarks/output/section_solve_parallelism.json](../benchmarks/output/section_solve_parallelism.json) | Latest micro-benchmark artefact (regenerate via script) |

---

## Implementation status (2026-07)

Summary of all implementation passes. Parity tests green (2026-07-14 baseline): **263**
`section_shell_model` tests (excl. `ccx`/`slow`/`legacy`; 1 skip, 2 xfail research locks;
includes offline root-segment deck tests), plus optimiser/wiring bundle (**36**),
interface-shear/FRD/batch + orchestration shell (**28**).

| Track | Item | Status | Notes |
|-------|------|--------|-------|
| **B1** | Documentation refresh | **Done** | ~17 doc files + 6 live docstrings updated; stale `thin_wall_bridge` / `lib/*` refs replaced |
| **A2** | Panel ABD threading | **Done** | FSDT periodic + K6 cache; CCX uses pre-solve `collect_panel_abd_stack`; HSDT backend removed 2026-07-14 |
| **A1** | Station FE cache split | **Done** | Periodic solve relies on `periodic_cache.py` geometry caches (load-independent); clamped Path B removed from optimiser |
| **A1** | FI worker-local caches | **Done** | `ProcessPoolExecutor(initializer=_shell_fi_worker_init)`; `_worker_cache_sizes` probe; `get_periodic_geometry_state()` |
| **A3** | Orchestration parallelisation | **Done** | Unified `run_shell_station_loop` parallel when `section_solve_n_workers > 1`; prepass JSON sidecar reload |
| **B2** | `lib.laminate_clpt` migration | **Done** | Production CLPT via `materials/laminate.py` + `materials/clpt_recovery.py`; `panel_abd.py` / `local_clpt_shell.py` shims removed |
| **B2** | Legacy test quarantine | **Done** | Vendor thin-wall `run_section` oracle removed; CI still excludes `legacy`/`ccx`/`slow` markers |
| **B2** | Demo / I/O migration | **Done** | `job_outputs.py` / `run_section_both` gated as regression/demo-only; v2 Path C prepass builds `station_analysis`; examples documented |
| **B2** | Path B removed from optimiser | **Done** | `_run_section_shell_solve` is periodic only; `_STATION_FE_RESULT_CACHE` removed; `use_periodic_bc` later deleted entirely (2026-07-14) |
| **A4** | Smaller perf items | **Done** | `splu` reuse, gated edge-traction diagnostics (`equilibrium_checks`), vectorised MITC4 Gauss |
| **Docs** | Perf report sync | **Done** | [PERFORMANCE_OPTIMISATION_REPORT.md](PERFORMANCE_OPTIMISATION_REPORT.md) finding statuses updated |
| **Follow-up** | Multi-station parallelism benchmark | **Done** | [`benchmarks/benchmark_section_solve_parallelism.py`](../benchmarks/benchmark_section_solve_parallelism.py); sample ~1.06× speedup (4 stations, 4 workers) |
| **Follow-up** | Nested worker oversubscription warning | **Done** | `warn_nested_worker_oversubscription()` in `orchestration/precompute/containers.py`; called from `pipeline.py` |
| **Follow-up** | Topology-cache churn profiling | **Done** | Panel-level partial invalidation **not warranted** for typical SLSQP loops |
| **Follow-up** | Deferred perf assessment (#8–9, HSDT Gauss) | **Done** | All deferred — not on production hot path; recorded in benchmark JSON |
| **Follow-up** | Delete clamped Path B FE | **Done (2026-07-14)** | `legacy_clamped` + B-vs-C tests + `use_periodic_bc` + dead `panel_mitc4_model` removed; docs drop Path C letter |
| **Follow-up** | Extract periodic caches / narrow registry | **Done (2026-07-14)** | `engine/periodic_cache.py`; `solve_global_coupled` dispatcher deleted |
| **Follow-up** | Dual-assembly + secondary recovery + module splits | **Done** | R1–R15 complete (cusp/full-solve spine deleted with R3) |
| **Follow-up** | Vendor housekeeping | **Done** | Thin-wall oracle removed; CCX fixtures on `ClptPlyRow`; docs synced |
| **Follow-up** | Interface shear + four-tier optimiser FI | **Done** | `interlaminar_fi_from_row_fsdt` + `compute_quaternary_fi_from_panel_rows`; theory shear operators; FRD bending coverage stats |

### What landed (by area)

**Panel ABD (A2)**

- New [`engine/panel_abd.py`](../engine/panel_abd.py): `panel_abd_tuple_from_lam()`, `compact_panel_abd_from_stacks()` — prefers `stiffness_bundle()` → `build_ABD()` → `laminate_definition_from_legacy()` → `stiffness_bundle()` (no `lib.laminate_clpt`).
- [`engine/global_shell_assembly.py`](../engine/global_shell_assembly.py): assembly-only return is now 8-tuple `(K, f, node_meta, panel_maps, abd_p, t_p, g_p, lab_p)`; cusp-penalty block reuses `panel_abd_stacks`.
- [`api/periodic.py`](../api/periodic.py): `_PERIODIC_K6_CACHE` stores panel ABD; FSDT periodic + tapered MITC4 paths consume threaded ABD (no post-solve rebuild).
- `calculix_ccx/periodic.py`: single **pre-solve** `collect_panel_abd_stack` (not post-solve). HSDT periodic paths were removed with the HSDT backend (2026-07-14).
- [`section_optimisation/engine/shell_fi_eval.py`](../../section_optimisation/engine/shell_fi_eval.py): `LaminateDefinitionMitc4SkinAdapter` delegates `fingerprint()` / `stiffness_bundle()` to source `LaminateDefinition`.
- Test: [`tests/test_panel_abd_threading.py`](../tests/test_panel_abd_threading.py).

**Optimiser FE caching (A1)**

- Periodic-only: `_run_section_shell_solve` always calls `run_section_periodic_bc`; geometry/K6 hits come from `engine/periodic_cache.py` (re-exported via `api/periodic.py`).
- Clamped Path B FE deleted (2026-07-14); B-vs-C regression suite removed with it.
- `_STATION_FE_RESULT_CACHE` removed; `_PIPELINE_B_CACHE` retains Pipeline B topology only.
- `_station_topology_key` now includes subcomponent **thickness** (fixes isotropic-only invalidation gap).
- Test: [`section_optimisation/tests/test_shell_fi_load_only_cache.py`](../../section_optimisation/tests/test_shell_fi_load_only_cache.py).
- `ProcessPoolExecutor(initializer=_shell_fi_worker_init)` pays import/compile cost up front per worker (parallel across workers).
- Public accessor [`get_periodic_geometry_state()`](../../section_shell_model/api/periodic.py) over `_cached_periodic_k6`.
- Debug probe `_worker_cache_sizes()` for worker-local cache introspection.
- Test: [`section_optimisation/tests/test_shell_fi_worker_cache_persistence.py`](../../section_optimisation/tests/test_shell_fi_worker_cache_persistence.py).

**Orchestration (A3)**

- [`orchestration/precompute/shell_runtime.py`](../../orchestration/precompute/shell_runtime.py): picklable `_shell_station_task` (theory dispatch + PNG/JSON).
- **`run_shell_station_loop`** fans out with `ProcessPoolExecutor` when `section_solve_n_workers > 1`.
- Prepass (`_run_structural_shell_prepass`) parallel workers write JSON manifest + pickle sidecar via [`section_shell_model/io/prepass_station_analysis_io.py`](../io/prepass_station_analysis_io.py); parent reloads with `load_prepass_station_analysis()`.
- `stages_pkg/shell.py` delegates to `run_shell_station_loop`.
- `section_solve_n_workers` threaded through `SectionShellModelParams`, recovery prep, pipeline, and shell stage paths.

**Maintainability (B1 + B2)**

- Package + repo docs refreshed (see B1 checklist below — all items addressed).
- Live docstrings updated in `pipeline_b`, orchestrators, `connected_assembly_core`, `calculix_ccx`, `README.md` tree.
- [`pyproject.toml`](../../../pyproject.toml): `legacy` pytest marker registered.
- [`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml): CI runs `pytest -m "not slow and not ccx and not legacy"`.
- [`docs/architecture/DEPRECATION_REGISTER.md`](../../../docs/architecture/DEPRECATION_REGISTER.md): `panel_station_shell_resultants` row added (status: deprecated/pending).
- [`api/fi_recovery.py`](../api/fi_recovery.py): batch Hashin uses `materials/clpt_recovery.ply_mid_stresses_laminate`; `laminate_definition_from_legacy()` resolves thin-wall panel laminates.
- [`materials/clpt_recovery.py`](../../materials/clpt_recovery.py): canonical CLPT stress/strain recovery; [`materials/laminate.py`](../../materials/laminate.py): `ClptPlyRow`, `build_clpt_ply_rows()`, `laminate_definition_from_legacy()`.
- Migrated off `lib.laminate_clpt`: `engine/homogenisation.py`, `engine/local_clpt_shell.py`, `engine/panel_abd.py`, `interface/recovery_point_set_builder.py`, `shell_fi_eval.LaminateDefinitionMitc4SkinAdapter`, `mitc4_solve_orchestrator.py`.
- `engine/local_clpt_shell.py`: production path uses `materials/clpt_recovery` + `hashin_fi` only (`use_materials_hashin` removed).

**Benchmark harness (follow-up pass)**

- [`benchmarks/benchmark_section_solve_parallelism.py`](../benchmarks/benchmark_section_solve_parallelism.py): serial vs parallel Path C station solves (Windows `spawn`-safe); topology-cache churn profiler; deferred-perf assessment.
- Output artefact: [`benchmarks/output/section_solve_parallelism.json`](../benchmarks/output/section_solve_parallelism.json) (regenerate via script below).
- Tests: [`tests/test_section_solve_parallelism_benchmark.py`](../tests/test_section_solve_parallelism_benchmark.py).
- Nested-pool guard: `warn_nested_worker_oversubscription()` in [`orchestration/precompute/containers.py`](../../orchestration/precompute/containers.py).

**Smaller perf (A4)**

- [`engine/connected_assembly_core.py`](../engine/connected_assembly_core.py): `reduce_propagate_solve(..., K_rr_factor=)` returns/reuses `splu` factor.
- [`engine/periodic_traction_superposition.py`](../engine/periodic_traction_superposition.py): `_KRR_FACTOR_CACHE` (LRU) reuses factor across traction RHS.
- [`processing/static/diagnostics/equilibrium_checks.py`](../processing/static/diagnostics/equilibrium_checks.py): `check_cluster_equilibrium` gated by `SECTION_SHELL_EDGE_TRACTION_DIAG` (default `"1"`; set `"0"` on hot paths).
- [`pre_processing/element_library/mitc4/element.py`](../pre_processing/element_library/mitc4/element.py): vectorised `einsum` Gauss integration on first-build; `_KE_TEMPLATE_CACHE` on hits.

### Remaining optional follow-ups

1. ~~**Production `runtime.json` A/B**~~ — **Done (2026-07-07)** — 2.02× combined prepass+shell at N=16 (workers 1 vs 4); see `benchmarks/run_runtime_ab.py` and orchestration measurement doc.
2. **Deferred backlog** — CCX reaction-K6 MPC redesign, Defect K / `transformed_basis` K6 (parked); see [DEFERRED_RESEARCH.md](DEFERRED_RESEARCH.md) (7th warping, geometric nonlinear, and HSDT revival are **permanently out of scope**).

#### Closed follow-ups (2026-07 third pass + HSDT migration)

| Item | Outcome |
|------|---------|
| Multi-station parallelism benchmark | **Done** — harness + tests; sample 1.06× speedup (4 stations, 4 workers, 20-core machine) |
| Drop `panel_abd.py` / `local_clpt_shell` CLPT shims | **Done** |
| Panel-level partial cache invalidation | **Profiled — not warranted** |
| Cusp penalty (#8), panel-local BC (#9), HSDT Gauss batching | **#9 Closed-obsolete**; #8 Closed-deferred (analytics only); HSDT Gauss deferred |
| Nested worker oversubscription warning | **Done** — `warn_nested_worker_oversubscription()` |
| HSDT oracle off `lib.laminate_clpt` | **Done** — `materials/laminate.py` (`Q_bar_transverse`, `clpt_ply_row`); HSDT tests migrated |
| Legacy MVP / demo `lib.laminate_clpt` | **Done** — tests migrated off `lib.laminate_clpt`; FSDT demos use shared `ensure_legacy_clpt_examples_path`; boundary test guards production imports |

---

## Architecture baseline (post-cleanup)

### Target layering

**Contract:** `pre_processing` prepares inputs for `simulation_runner`; the runner
calls `processing` operations and result computers and **records diagnostic
information** about those subprocesses. Per-domain `*/api/` on prep/results;
no package-level `api/` façade.

| Layer | Role |
|-------|------|
| `pre_processing/` | Mesh, constitutive, kernels, loading, dispatch → **runner inputs** (`*/api/`) |
| `simulation_runner/` | **Orchestrates** ops + results; owns solve state; records diagnostics |
| `processing/` | Ops (`static/` + `buckling/`), results tiers, diagnostics **implementations** |

**Default orchestration track:** `SectionLinearStaticRunner` — classic FE staged
solve + result tiers (1d-fem analogue). See [SECTION_SOLVE_STAGES.md](SECTION_SOLVE_STAGES.md).

**Main FI track:** `SectionLinearStaticRunner` via `api/linear_static` +
`api/station_recovery`. Channel-A (`SectionPeriodicRunner` / `api/k6` /
`operations/periodic/`) is **removed** (Phase 0). See [SHELL_MODEL.md](SHELL_MODEL.md)
and [LINEAR_STATIC_MAIN_TRACK_PHASE0.md](LINEAR_STATIC_MAIN_TRACK_PHASE0.md).

### Canonical modules (import these)

| Need | Module |
|------|--------|
| Pipeline B geometry | `pre_processing/mesh/api/pipeline_b.py` |
| Default orchestration (classic FE) | `simulation_runner/section_linear_static_runner.py` |
| Linear-static FI cascade (authoritative production FI) | `simulation_runner/api/linear_static.py` + `station_recovery.py` |
| Full-section buckling | `simulation_runner/api/linear_buckling.py` |
| Hashin / CLPT FI helpers | `processing/static/results/api/fi_recovery.py` |
| Global assembly | `processing/static/operations/global_shell_assembly.py` + `section_assembly.py` |
| Panel ABD extraction | `pre_processing/constitutive/panel_abd.py` |
| Equilibrium diagnostics | `processing/static/diagnostics/equilibrium_checks.py` |
| Theory registry / station dispatch | `pre_processing/dispatch/` |
| Handoff DTOs | `pre_processing/containers/{station,recovery,buckling,station_analysis}_types.py` |
| Parallelism benchmark | `benchmarks/benchmark_section_solve_parallelism.py` |

### Removed (do not reintroduce)

- `api/thin_wall_bridge.py`, `api/recovery_adapter.py`, `api/mitc4_workflow.py`, `api/diagnostics.py`
- `api/legacy_clamped.py` / `run_section_with_mitc4_shell` (clamped Path B FE — deleted 2026-07-14)
- `pre_processing/element_library/mitc4/utilities/panel_mitc4_model.py` (dead after Path B)
- Top-level `section_shell_model/engine/` (Wave 6 — relocated into the three layers)
- `engine/global_mitc4_assembly.py` (alias → use `assemble_global_shell`)
- `examples/repro_*.py`, `element_library/calculix_ccx/__init__.py`
- `solve_global_coupled_shell` / cusp-penalty diagnostic (R3, 2026-07-14)
- `engine/panel_abd.py`, `io/example_plots.py`, `_utils/_stress_model_path.py` (warn-stage shims)
- `bc_mode="full_clamp"` alias (use `contour_support`)

### Production section solve

- **Periodic unit-cell** (`run_section_periodic_bc`) — sole FE entry for Hashin FI and optimiser `shell_fi_batch`.
- **Clamped global-coupled Path B** — **deleted** (`api/legacy_clamped.py` / `run_section_with_mitc4_shell` removed 2026-07-14). Assembly-only helpers (`assemble_global_shell` / `assemble_for_periodic_k6`) remain for periodic K6 and CCX thin-wall validation.
- Blade-root clamp lives upstream in cantilever resultant integration — see [SHELL_MODEL.md](SHELL_MODEL.md) multi-scale BC section. The section FE does not clamp either cut face.
- Do **not** rename `api/pipeline_b.py` (geometry topology); it is unrelated to the deleted Path B FE path.

---

## Track A — Faster optimiser runs (performance)

**Goal:** Reduce wall-clock of `section_optimisation.engine.shell_fi_eval.shell_fi_batch` and
orchestration shell stages without changing numerical parity (~1e-10 on K6/FI tensors).

**Measure success via:**

- `<job>/runtime.json`: `section_optimisation_s`, `shell_runtime_prepass_s`, `section_shell_model_s`
- Micro-profile: call counts to `mitc4_stiffness`, `abd_stack`, `spsolve` / `splu`
- Regression: `section_optimisation/tests/test_shell_fi_*.py`, `section_shell_model/tests/test_periodic_cluster_basis.py`

---

### A1. Station-level FE caching (optimiser hot path)

**Status: done** — Path C load/geometry split via `periodic.py` caches; FI process pool uses `initializer=` for worker warm-up; real-process cache persistence test green.

#### Current state (post-implementation)

Two in-process module-level caches in `section_optimisation/engine/shell_fi_eval.py`:

```python
_PIPELINE_B_CACHE: dict[tuple, Any] = {}      # geometry/topology only
_TAPERED_PAIR_CACHE: dict[tuple, Any] = {}    # tapered pair topology
```

| Cache | Key | What is cached | Status |
|-------|-----|----------------|--------|
| `_PIPELINE_B_CACHE` | `_station_topology_key(sdef, bg, s_idx)` incl. **thickness** | `PipelineBSection` | OK |
| (inside `engine/periodic_cache.py`) | `_PERIODIC_ASSEMBLY_CACHE`, `_PERIODIC_K6_CACHE` | Assembly + K6 + **panel ABD** | Load-independent; used by Path C |

Path C: `_run_section_shell_solve` → `run_section_periodic_bc` → `_cached_periodic_k6` (geometry hit) → load-only 6×6 solve + recovery.

`use_periodic_bc` was removed from `shell_fi_batch` (2026-07-14); the periodic section solve is always used. `_STATION_FE_RESULT_CACHE` (former clamped load-keyed cache) **removed**.

`shell_fi_batch` uses a persistent `ProcessPoolExecutor` with `initializer=_shell_fi_worker_init` — workers pay import/compile cost up front (parallel across workers) and retain warm module-level caches across tasks within each worker process.

#### Remaining work (optional)

1. ~~**Panel-level partial invalidation**~~ — **Profiled (not warranted).** Station-level topology keys are sufficient for typical SLSQP loops; see `benchmarks/benchmark_section_solve_parallelism.py` → `topology_cache_churn`.

#### Key files

| File | Role |
|------|------|
| `section_optimisation/engine/shell_fi_eval.py` | `_shell_fi_worker_init`, `_worker_cache_sizes`, `_get_shell_fi_pool`, `shell_fi_batch` |
| `section_shell_model/processing/static/operations/periodic/periodic_cache.py` | `_cached_periodic_k6`, `get_periodic_geometry_state`, `_PERIODIC_K6_CACHE` |
| `section_shell_model/api/periodic.py` | Theory router + FSDT periodic solvers (re-exports cache helpers) |
| `section_optimisation/tests/test_shell_fi_eval_cache.py` | Geometry cache smoke |
| `section_optimisation/tests/test_shell_fi_load_only_cache.py` | Load-only re-eval skips assembly |
| `section_optimisation/tests/test_shell_fi_worker_cache_persistence.py` | **New** — real-process worker cache persistence |
| `section_optimisation/tests/test_shell_fi_dirty_cache.py` | Dirty-station parity |

#### Acceptance criteria

- [x] Second `shell_fi_batch` with same geometry, different loads skips assembly (Path C; `test_shell_fi_load_only_cache.py`).
- [x] Dirty-station and cold-full parity tests green.
- [x] Laminate / thickness changes invalidate topology key.
- [x] Parallel `n_workers > 1` reuses warm geometry caches within each worker process (`test_shell_fi_worker_cache_persistence.py`).

---

### A2. Panel ABD threading (eliminate repeated CLT work)

**Status: done** — assembly + periodic K6 cache thread ABD; CCX calls `collect_panel_abd_stack` once **pre-solve** (not post-solve).

#### Current state (post-implementation)

| Location | Mechanism | Status |
|----------|-----------|--------|
| `engine/panel_abd.py` | `panel_abd_tuple_from_lam()` via `stiffness_bundle()` or `laminate_definition_from_legacy()` | **Canonical helper** |
| `engine/global_shell_assembly.py` | One ABD build per panel in assembly loop; returned in 8-tuple | **Done** |
| `api/periodic.py` | `_PERIODIC_K6_CACHE` carries `(abd_p, t_p, g_p, lab_p)`; FSDT periodic + tapered MITC4 consume it | **Done** |
| `calculix_ccx/periodic.py` | Pre-solve `collect_panel_abd_stack` | **Done** |
| `element_library/mitc4/utilities/mitc4_solve_orchestrator.py` | `panel_abd_tuple_from_lam` | **Done** |

#### Remaining work (optional)

1. ~~Thread HSDT ABD through assembly cache payload~~ — **Obsolete** (HSDT experimental backend removed 2026-07-14; no restore scheduled on this track).
2. ~~Drop legacy `lib.laminate_clpt.abd_stack` fallback in `panel_abd.py`~~ — **Done** (uses `laminate_definition_from_legacy()`).

#### Key files

| File | Role |
|------|------|
| `engine/panel_abd.py` | Shared ABD extraction |
| `engine/global_shell_assembly.py` | Assembly loop, 8-tuple return |
| `api/periodic.py` | K6 cache payload, bundle builders |
| `materials/laminate.py` | `LaminateStiffnessBundle`, `stiffness_bundle()` |
| `tests/test_panel_abd_threading.py` | Call-count regression |

#### Acceptance criteria

- [x] Single `panel_abd_tuple_from_lam` per panel per assembly miss (`test_panel_abd_threading.py`).
- [x] K6 and FI tensors unchanged (existing parity tests green).
- [x] No post-solve ABD rebuild on production periodic paths (FSDT threads ABD via K6 cache).

---

### A3. Orchestration parallelisation (cross-package)

**Status: done** — legacy FSDT loop, v2 loop, and prepass parallelised when `section_solve_n_workers > 1`.

#### Current state (post-implementation)

| Stage | Parallelism | Where |
|-------|-------------|-------|
| `section_geometry` | `ProcessPoolExecutor` | `stages_pkg/geometry.py` |
| `section_loads` | `ProcessPoolExecutor` | `stages_pkg/loads.py` |
| `shell_runtime_prepass` | **Parallel** (when workers > 1) | `shell_runtime._run_structural_shell_prepass` + JSON sidecar reload |
| `section_shell_model` | **Parallel** (when workers > 1) | `run_shell_station_loop` via `_shell_station_task` |
| `shell_fi_batch` (optimiser) | **Parallel** across stations | `shell_fi_eval.py` process pool |

`section_solve_n_workers` is threaded through `SectionShellModelParams`, recovery prep, pipeline, and both shell stage paths.

#### Remaining work (optional)

1. ~~**Benchmark** multi-station jobs~~ — **Done** (micro-benchmark harness). Optional: full `main_precompute` `runtime.json` A/B at `N_STRUCTURAL=16`.
2. ~~**Review** nested parallelism warning~~ — **Done** — `warn_nested_worker_oversubscription()` in `containers.py`; called from `pipeline.py`.

#### Sample benchmark result (2026-07-07, 20-core machine, 4 stations, 4 workers)

| Mode | Mean wall (s) | Notes |
|------|---------------|-------|
| Serial (`n_workers=1`) | 1.55 | 4 Path C station solves × 2 repeats |
| Parallel (`n_workers=4`) | 1.46 | One worker per station |
| Speedup | **1.06×** | Modest at N=4; spawn overhead dominates; expect larger gains at N=16 |

Regenerate: `python blade_precompute/section_shell_model/benchmarks/benchmark_section_solve_parallelism.py`

#### Key files

| File | Role |
|------|------|
| `orchestration/precompute/shell_runtime.py` | `_shell_station_task`, `run_shell_station_loop`, `_run_structural_shell_prepass` |
| `section_shell_model/io/prepass_station_analysis_io.py` | Prepass JSON manifest + pickle sidecar reload |
| `orchestration/precompute/stages_pkg/shell.py` | `section_shell_model_impl` (delegates to `run_shell_station_loop`) |
| `orchestration/precompute/pipeline.py` | `section_solve_n_workers`, prepass skip logic, calls `warn_nested_worker_oversubscription` |
| `orchestration/precompute/containers.py` | `warn_nested_worker_oversubscription()` |
| `benchmarks/benchmark_section_solve_parallelism.py` | Micro-benchmark + topology/deferred-perf profiling |

#### Acceptance criteria

- [x] `runtime.json` timing improvement measured on multi-station production job (optional benchmark; see `benchmarks/benchmark_section_solve_parallelism.py`).
- [x] PNG/JSON content matches serial run (parity tests green).
- [x] Process-based fan-out with `plt.close("all")` in worker (legacy + v2 paths).
- [x] v2 loop parallelised.
- [x] Prepass reloads station analysis from JSON sidecar in parent process.

---

### A4. Smaller perf items (lower priority, same track)

**Status: done** (2026-07 second pass).

| Item | Where | Status |
|------|-------|--------|
| Reuse `splu` in traction superposition | `engine/periodic_traction_superposition.py` `_KRR_FACTOR_CACHE` + `connected_assembly_core.reduce_propagate_solve` | **Done** |
| Gate edge-traction diagnostics | `processing/static/diagnostics/equilibrium_checks.py` (`SECTION_SHELL_EDGE_TRACTION_DIAG`) | **Done** |
| Vectorise first-build MITC4 Gauss loop | `pre_processing/element_library/mitc4/element.py` (`einsum` + `_KE_TEMPLATE_CACHE`) | **Done** |
| Legacy panel-local COO path | `panel_mitc4_model.py` | **Removed** with Path B (2026-07-14) |

See [PERFORMANCE_OPTIMISATION_REPORT.md](PERFORMANCE_OPTIMISATION_REPORT.md) for ranked findings — statuses updated.

---

## Track B — Maintainability (doc refresh + legacy thin-wall retirement)

**Goal:** Remove confusion from stale docs and deprecated MVP paths so new work targets Path C only.

**Do this track first** if the team is onboarding, cutting releases, or before large perf refactors — it reduces wrong imports and accidental use of placeholder resultants.

---

### B1. Documentation refresh

**Status: done** (2026-07 pass).

All checklist items below were addressed. Stale references now point at `api/periodic.py`, `api/pipeline_b.py`, `api/fi_recovery.py`, `engine/global_shell_assembly.py`, and `pre_processing/element_library/mitc4/`.

**Shim files:** `DEPRECATION_REGISTER.md` marks `api/thin_wall_bridge.py`, `api/recovery_adapter.py`, `api/legacy_clamped.py`, etc. as removed. If any shim files reappear on disk, delete them — canonical imports are the modules listed in the architecture baseline above.

#### Reference map (for future edits)

| Stale | Replace with |
|-------|----------------|
| `api/thin_wall_bridge.py` / `api/legacy_clamped.py` | `api/periodic.py`, `api/pipeline_b.py` (by symbol) |
| `api/recovery_adapter.py` / `engine/shell_recovery_builder.py` | `api/fi_recovery.py`, `processing/static/diagnostics/equilibrium_checks.py` |
| `lib/recovery_adapter.py`, `lib/global_mitc4_assembly.py`, `lib/mitc4_element.py` | `engine/global_shell_assembly.py`, `pre_processing/element_library/mitc4/` |

#### Completed checklist (2026-07)

**Package docs:** `PERFORMANCE_OPTIMISATION_REPORT.md`, `SHELL_MODEL.md`, `DEFERRED_RESEARCH.md`, `PERIODIC_CLUSTER_BASIS_SINGULARITY.md`, `PERIODIC_CCX_HOMOGENIZATION.md`, `CALCULIX_CCX_ELEMENT_LIBRARY_RESTRUCTURE.md`, `README.md`.

**Repo docs:** `docs/planning.md`, `docs/BEAM_SECTION_MODEL_LIMITATIONS_TAPERED_PERIODIC.md`, `docs/architecture/METHODOLOGY_MANUAL.md`, `docs/architecture/hsdt_quad4_shell_design.md`, `docs/architecture/calculix_s4r_shell_design.md`, `docs/architecture/area_diagrams_shell_load_mapping.md`, `docs/architecture/phase_d_panel_buckling_cutover_plan.md`, `docs/architecture/REFACTOR_BACKLOG_blade_precompute.md` (thin_wall_bridge split marked done).

**Examples / plans:** `examples/area-diagrams-FSDT-shell/README.md`, `plans/area-diagrams-shell/03_strip_vs_shell_comparison.md`.

**Module docstrings:** `pipeline_b.py`, `mitc4_solve_orchestrator.py`, `hsdt_solve_orchestrator.py`, `connected_assembly_core.py`, `calculix_ccx` solvers.

---

### B2. Legacy thin-wall retirement

**Status: done** — optimiser on periodic section solve only; production CLPT on `materials`; clamped Path B FE deleted (2026-07-14).

#### Symbols (retired with Path B)

| Symbol | Former home | Status |
|--------|-------------|--------|
| `panel_station_shell_resultants` | `api/legacy_clamped.py` | Removed with Path B; deprecation-register row historical |
| `run_section_with_shell_mapping` / `run_section_both` / `run_section_with_mitc4_shell` | `api/legacy_clamped.py` | **Deleted** with `api/legacy_clamped.py` |

Do **not** reintroduce `legacy_clamped.py` or a clamped Path B oracle.

#### Callers — migration status

| Caller | Status |
|--------|--------|
| `section_optimisation/engine/shell_fi_eval.py` | **Done** — periodic section solve only; isotropic FI via `materials` CLPT |
| `section_shell_model/io/job_outputs.py` | **Done** — gated regression/demo-only; prepass uses periodic station analysis |
| `section_shell_model/tests/test_section_shell_model.py` | Vendor thin-wall `run_section` oracle removed; geometry conversion covered package-locally |
| `tests/test_b_vs_c_path_regression.py` | **Deleted** with Path B |

#### `lib.laminate_clpt` / `sys.path` shim

**Migrated (2026-07):** assembly, panel ABD, batch Hashin, homogenisation, recovery builders, optimiser isotropic FI (`shell_fi_eval` → `materials.clpt_recovery` / `ClptPlyRow`), CCX periodic unit fixtures (`ClptPlyRow`).

**Vendor path fallback (demo-only):**

- `examples/area-diagrams-FSDT-shell/*` — optional demo bootstrap only
- Topology defaults are package-local (`pre_processing/constitutive/default_laminates.py`)
- Production containment: `tests/test_legacy_clpt_import_boundary.py` (`lib.laminate_clpt` allow-list is **empty**; production modules must not call `ensure_stress_model_lib_path`)

#### Retirement sequence

1. ~~Update docs (B1).~~ **Done.**
2. ~~Migrate assembly + `collect_panel_abd_stack` off `lib.laminate_clpt` (A2).~~ **Done.**
3. ~~Add `pytest.mark.legacy` + CI exclusion.~~ **Done.**
4. ~~Migrate `job_outputs.py` and examples to Path C.~~ **Done** — gated as regression/demo-only; v2 prepass on Path C.
5. ~~Finish CLPT migration (`ply_mid_stresses`, adapter Ply shim, homogenisation/recovery paths).~~ **Done.**
6. ~~Remove Path B from optimiser (`use_periodic_bc=False`).~~ **Done.**
7. ~~Drop `panel_abd.py` / `local_clpt_shell` legacy fallback paths.~~ **Done** (2026-07 follow-up).

#### DEPRECATION_REGISTER

Row added for `panel_station_shell_resultants` (status: deprecated/pending). See `docs/architecture/DEPRECATION_REGISTER.md`.

---

## Suggested execution order

### Completed (2026-07 passes)

1. ~~**B2** Migrate `job_outputs.py` to Path C~~ **Done**
2. ~~**A3** v2 station loop parallelisation + prepass JSON reload~~ **Done**
3. ~~**B2** Finish CLPT migration~~ **Done**
4. ~~**A2** HSDT / legacy ABD threading~~ **Done**
5. ~~**A4** Wire splu reuse + profiling-driven items~~ **Done**
6. ~~**B2** Remove Path B from optimiser~~ **Done**
7. ~~**Docs** Update PERFORMANCE_OPTIMISATION_REPORT.md~~ **Done**
8. ~~**Follow-up** Multi-station parallelism benchmark + nested-worker warning~~ **Done**
9. ~~**Follow-up** Drop `panel_abd.py` / `local_clpt_shell` CLPT shims~~ **Done**
10. ~~**Follow-up** Profile topology-cache churn; assess deferred perf #8–9~~ **Done** (not warranted on hot path)
11. ~~**Follow-up** Delete clamped Path B FE + naming pass~~ **Done (2026-07-14)**
12. ~~**Follow-up** Extract `engine/periodic_cache.py`; narrow `shell_solve_registry`~~ **Done (2026-07-14)**
13. ~~**Follow-up** Invert pre_processing → processing diagnostics ownership~~ **Done (2026-07-14)**

### Optional next work

Historical measurement / research pointers:

1. ~~**Production `runtime.json` A/B**~~ — **Done (2026-07-07)** — see orchestration measurement doc.
2. **Deferred research** — see [DEFERRED_RESEARCH.md](DEFERRED_RESEARCH.md) and § Out of scope below (CCX, `transformed_basis` K6, Defect K; **not** 7th warping or geometric nonlinear).
3. ~~**Legacy MVP / demo CLPT shims**~~ — **Done**.
4. ~~**Narrow `shell_solve_registry`**~~ — **Done** — deleted unused `solve_global_coupled` dispatcher; kept `register_backend` / `BACKENDS` / `resolve_backend` for CCX opt-in.
5. ~~**Extract periodic caches**~~ — **Done** — now `processing/static/operations/periodic/periodic_cache.py`; `api/periodic.py` re-exports for probes.
6. ~~**pre_processing → processing import inversion**~~ — **Done** — `TopologyV2Diagnostics` owned by mesh/topology; formulation shape validation owned by `pre_processing/element_library/formulation_validation.py` (processing modules re-export).
7. ~~**Delete clamped FE Path B**~~ — **Done (2026-07-14)** — `api/legacy_clamped.py`, B-vs-C tests, `use_periodic_bc` flag, dead `panel_mitc4_model.py` removed; docs use “periodic section solve” (no Path C letter).
8. ~~**Channel-A via `simulation_runner`**~~ — **Done** — `SectionPeriodicRunner` + `simulation_runner/api/`; package-level `api/` removed; geometry/FI live on layer APIs.

### Architecture / SoC backlog (R1–R15 — complete)

Do not wire deferred research into optimiser constraints. Historical ledger:

| # | Item | Status | Notes / owners |
|---|------|--------|----------------|
| R1 | **Dual-assembly collapse** | **Done** | Periodic cache → `assemble_section_from_panels` (`build_connected_panel_maps` + formulation cache + `assemble_section_system`); K6 homogenisation retained. |
| R2 | **Secondary recovery wiring** | **Done** | Production periodic recovers via `periodic_secondary_recovery` → `ShellSecondaryResultsOrchestrator` with retained cache; traction/modes share the bridge. |
| R3 | **Strip clamped-only assembly branches** | **Done** | Assembly-data callers → `assemble_global_shell` / `assemble_for_periodic_k6`; cusp penalty + `solve_global_coupled_shell` + dead FSDT global-backend registration deleted (2026-07-14). Traction/CCX keep independent contour-support helpers. |
| R4 | **Unify theory registries** | **Done** | `engine/shell_theory_registry.py` owns periodic/station/global callables; old maps are compatibility views. |
| R5 | **Split `io/example_plots.py`** | **Done** | Canonical: `processing/static/diagnostics/example_plots.py` (+ `vis.py`); warn-stage `io/example_plots.py` shim deleted. |
| R6 | **Split `pre_processing/containers/types.py`** | **Done** | Leaf modules: `station_types`, `recovery_types`, `buckling_types`, `station_analysis_types`; `types.py` is a facade. |
| R7 | **Relocate `engine/panel_abd.py`** | **Done** | Canonical: `pre_processing/constitutive/panel_abd.py`; warn-stage engine shim deleted. |
| R8 | **Fold `assemble_section_system` into `processing/static/operations/`** | **Done** | Canonical: `processing/static/operations/assemble_global.py`; `engine/section_assembly.py` re-exports. |
| R9 | **Element-agnostic `compute_resultants_from_modes`** | **Done** | Legacy homogenisation path deleted; production + tapered stubs use `periodic_secondary_recovery` → secondary orchestrator. |
| R10 | **Gaussian / nodal resolution tiers** | **Done** | Opt-in secondary Gaussian (`mitc4_resultants_at` @ 2×2 GP) + nodal bilinear extrapolation; see [RESULTS_DESIGN.md](../processing/static/results/RESULTS_DESIGN.md). |
| R11 | **Rename / clarify `bc_mode="full_clamp"`** | **Done** | Default `contour_support`; `full_clamp` alias removed after one-cycle warn. |
| R12 | **`sys.path` / `multi_cell_blade_section` shims** | **Done** | Optimiser off `lib.laminate_clpt`; package-local `Panel` + `default_laminates.py`; topology has no vendor bootstrap. Production plot/NACA helpers no longer call `ensure_stress_model_lib_path`; demos only. |
| R13 | **Silent public shim warnings** | **Done** | Warn-stage shims (`engine/panel_abd`, `io/example_plots`, `_ensure_stress_imports`, `bc_mode=full_clamp`) retired. |
| R14 | **`use_hsdt_interlaminar` flag chain** | **Done** | Residual fixture kwarg removed from `tests/test_precompute_pipeline_order.py`; no orchestration field remains. |
| R15 | **Move compare helpers out of `api/fi_recovery.py`** | **Done** | Canonical: `processing/static/diagnostics/path_compare.py`; `api/fi_recovery` re-exports. |
| R16 | **Channel-A via `simulation_runner`** | **Done** | `SectionPeriodicRunner` + `simulation_runner/api/`; package-level `api/` dissolved; prep via `pre_processing/*/api/`. Docs: linear-static = default orchestration; periodic = specialized FI path. See [SHELL_MODEL.md](SHELL_MODEL.md). |

### Deferred research (do not schedule into optimiser)

| Item | Doc |
|------|-----|
| Periodic × `transformed_basis` K6 — **parked permanently** (Spike 1B); production stays on `shared` | [PERIODIC_CLUSTER_BASIS_SINGULARITY.md](PERIODIC_CLUSTER_BASIS_SINGULARITY.md) |
| CCX periodic acceptance (policy closed); reaction K6 diagnose closed (no-promote); MPC redesign still deferred | [DEFERRED_RESEARCH.md](DEFERRED_RESEARCH.md), [PERIODIC_CCX_HOMOGENIZATION.md](PERIODIC_CCX_HOMOGENIZATION.md) |
| Cusp traction penalty analytics (#8 only; #9 Closed-obsolete) | [PERFORMANCE_OPTIMISATION_REPORT.md](PERFORMANCE_OPTIMISATION_REPORT.md) — #8 research-only, off hot path |
| Defect K: accepted on periodic FI (`shared`); harness in `defect_k_mesh_sweep.py` | [SHELL_MODEL.md](SHELL_MODEL.md) |
| Load application: provenance + single periodic free set for face traction (**closed**) | [LOAD_APPLICATION_DOUBLE_COUNTING.md](LOAD_APPLICATION_DOUBLE_COUNTING.md) |
| Hub attachment / warping restraint / root stress raisers | Out of scope for `section_shell_model` (multi-scale BC) |
| Root-segment CCX warping-restraint diagnostic (**landed**; validation only — no promote) | [ROOT_SEGMENT_CCX_DIAGNOSTIC.md](ROOT_SEGMENT_CCX_DIAGNOSTIC.md) |

### Suggested next execution order

1. ~~**R1 + R2**~~ **Done**
2. ~~**R3 + R11**~~ **Done** — cusp diagnostic + full-solve spine deleted; contour supports remain on traction/CCX helpers
3. ~~**R4…R15**~~ **Done** — including R12 package-local defaults/Panel and warn-stage shim retirement
4. ~~**Housekeeping leftovers**~~ **Done** — vendor thin-wall oracle removed; CCX fixtures on `ClptPlyRow`; docs synced
5. ~~**Feature track** — promote FSDT interface `τ13/τ23` interlaminar FI~~ **Done** — optimiser uses four-tier bridge + interface shear (scalar fallback for single-ply / untrusted Q); theory shear operators behind `resolve_shear_recovery_operator`; FRD bending merge returns coverage stats with explicit RESERVED reasons
6. ~~**Root-segment CCX diagnostic**~~ **Done** — clamped multi-row S4R vs periodic RVE at station 0; quarantined validation only (no production promote)
7. **Parked:** deferred research only (do not schedule into optimiser) — see table above

### Optional polish (not blocking production)

| Item | When | Notes |
|------|------|-------|
| Demo vendor bootstrap | Optional | `examples/area-diagrams-FSDT-shell/*` may still bootstrap vendor tree |
| ~~Interlaminar KS impact check~~ | **Done (2026-07-14)** | `tests/test_interlaminar_ks_impact.py` — first-ply S13 raises KS vs max-ply; homogeneous midplane matches scalar peak; heterogeneous G diverges and KS tracks interface τ |
| FI worker pickle size | If profiling shows it | Laminate fingerprint interning / slimmer worker payloads |
| CCX FRD coverage → 100% | Validation-only | Coverage stats already returned; raise remap gate only if promoting CCX Hashin trust |
| ~~Align S13 allowable first-ply vs max-ply~~ | **Done (2026-07-14)** | Canonical `interlaminar_allowable_pa` in `compute_hashin_fi.py`; quaternary + optimiser + `api/fi_recovery` share it (first-ply laminate helper + per-ply helper) |

### Original ordering (for reference)

**Optimiser speed:** A2 → A1 → A3 → A4 — **all landed**.

**Maintainability-first:** B1 → B2 CLPT → B2 demo migration → Track A — **all landed**.

**2026-07-14 Path B retirement + SoC + FI follow-up:** clamped FE deleted; periodic caches extracted; registry narrowed; R1–R15 complete; warn-stage aliases removed; vendor housekeeping + interface-shear / four-tier optimiser FI landed.

---

## Verification commands

```powershell
# Package fast gate (FSDT production; excludes ccx/slow/legacy)
python -m pytest blade_precompute/section_shell_model -q `
  -m "not slow and not ccx and not legacy"

# Wiring + optimiser FI caches + ABD threading + parallelism micro-bench tests
python -m pytest `
  tests/test_section_shell_mitc4_wiring.py `
  tests/test_recovery_adapter_tapered_pipeline.py `
  blade_precompute/section_shell_model/tests/test_panel_abd_threading.py `
  blade_precompute/section_optimisation/tests/test_shell_fi_eval_cache.py `
  blade_precompute/section_optimisation/tests/test_shell_fi_load_only_cache.py `
  blade_precompute/section_optimisation/tests/test_shell_fi_worker_cache_persistence.py `
  blade_precompute/section_optimisation/tests/test_shell_fi_provenance_gate.py `
  blade_precompute/section_optimisation/tests/test_shell_fi_dirty_cache.py `
  blade_precompute/section_shell_model/tests/test_section_solve_parallelism_benchmark.py -q

# Interface-shear / four-tier FI promotion + FRD bending unit coverage
python -m pytest `
  blade_precompute/section_shell_model/tests/test_interface_shear_fi_promotion.py `
  blade_precompute/section_shell_model/tests/test_fsdt_interface_shear_recovery.py `
  blade_precompute/section_shell_model/tests/test_tapered_interlaminar_fi.py `
  blade_precompute/section_shell_model/calculix_ccx/tests/test_frd_bending_apply.py `
  blade_precompute/section_shell_model/tests/test_legacy_clpt_import_boundary.py -q

# Production contract locks (theory defaults, lazy CCX, no deleted paths)
python -m pytest `
  blade_precompute/section_shell_model/tests/test_production_contract_locks.py -q

# Orchestration shell stages
python -m pytest blade_precompute/orchestration -q -k "shell"

# Optional: CCX E2E (requires ccx on PATH)
python -m pytest blade_precompute/section_shell_model/calculix_ccx/tests -q -m ccx

# Regenerate parallelism benchmark JSON (optional)
python blade_precompute/section_shell_model/benchmarks/benchmark_section_solve_parallelism.py
```

**Parity locks:** K6 matrices and FI tensors vs pre-change baselines (~1e-10); `test_periodic_cluster_basis.py` (research xfails), `test_shell_fi_*`, `test_production_contract_locks.py`. (B-vs-C regression suite deleted with Path B.)

---

## Out of scope (see DEFERRED_RESEARCH.md)

**Will not feature in `section_shell_model` (policy, 2026-07):**

- 7th warping / bimoment mode on the periodic unit cell
- Geometric nonlinear Newton loop
- Numba/Cython element kernels
- Keeping a clamped Path B “oracle”

**Deferred / not on current track:**

- ~~`transformed_basis` periodic K6 formulation fix~~ **Parked permanently** (Spike 1B 2026-07-14; production stays on `shared`)
- ~~CCX periodic acceptance hardening~~ **Done (policy)** — load-aligned ~10% / torsion Nxy `rel=0.85`
- ~~CCX reaction K6 diagnose~~ **Done (no-promote)** — MPC Lagrange `RF` ≠ beam conjugates; hybrid MITC4 `K6` stays; MPC redesign still deferred
- Cusp traction analytics (perf #8; off hot path; #9 Closed-obsolete with Path B)
- Defect K non-collinear junction traction residuals — **accepted** on periodic FI; measure via mesh-sweep harness (Track 2A)

---

## Commit guidance

Use [Conventional Commits](https://www.conventionalcommits.org/): separate commits per track item, e.g.:

- `perf(section_shell_model): thread panel ABD from assembly cache`
- `perf(section_optimisation): split station FE cache by geometry and loads`
- `perf(orchestration): parallelise shell_runtime station loop`
- `docs(section_shell_model): refresh imports after facade removal`
- `refactor(section_shell_model): migrate collect_panel_abd_stack off lib.laminate_clpt`
- `refactor(section_shell_model): delete clamped Path B FE entry`
- `refactor(section_shell_model): extract periodic assembly/K6 caches`
