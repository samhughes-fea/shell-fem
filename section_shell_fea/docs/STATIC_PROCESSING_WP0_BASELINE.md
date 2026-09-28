# Static processing WP0 — baseline freeze

**Date:** 2026-09-27  
**Programme:** Static processing subpackage refactor + operation-family performance.

## D0.4 — Architecture guard baseline

Focused suite (pre-move):

```text
pytest .../test_element_formulation_import_boundary.py
       .../test_fe_row_ownership.py
       .../test_formulation_cache_assembly_parity.py
       .../test_solve_quality.py
       .../test_runner_four_tier_pipeline.py -q
```

**Result:** `28 passed`.

## D0.1 — Import surface (deep modules)

### Operations (canonical bodies to move)

| Module | Family |
|--------|--------|
| `solve_states.py` | contracts |
| `physics_contract.py` | contracts |
| `prepare_local.py` | preparation |
| `assemble_global.py` | assembly |
| `section_assembly.py` | assembly |
| `constrain_global.py` | reduction |
| `reduce_constrained.py` | reduction |
| `solve_reduced.py` | solution |
| `solve_constrained.py` | solution |
| `reconstruct_field.py` | recovery |
| `disassemble_field.py` | recovery |

### Operations (legacy root aliases; remain at package root)

| Module | Notes |
|--------|-------|
| `global_shell_assembly.py` | legacy 8-tuple / test helpers |
| `connected_assembly_core.py` | silent shim → numbering/constraints |
| `connected_panel_topology.py` | silent shim → pre_processing mesh |
| `cap_tie_constraints.py` | silent shim → constraints.cap_tie |

### Diagnostics (canonical bodies to move)

| Module | Family |
|--------|--------|
| `solve_quality.py`, `solve_phase_timing.py`, `dof_provenance.py` | solve |
| `equilibrium_checks.py` | equilibrium |
| `asme_gci.py`, `cascade_mesh_convergence.py`, `grid_independence_study.py`, `plot_grid_independence.py`, `defect_k_mesh_sweep.py`, `cap_peak_sampling.py`, `stability_scorecard.py`, `cascade_trust.py` | convergence |
| `k6_parity.py`, `path_compare.py`, `formulation_validation.py`, `topology_diagnostics.py` | parity |
| `mesh_plots.py`, `example_plots.py`, `thin_wall_plots.py`, `loading_vis.py` | plots |

## D0.2 / D0.3 — Numerical and phase evidence

Numerical parity continues to be owned by existing fixtures:

- `test_formulation_cache_assembly_parity.py`
- `test_runner_four_tier_pipeline.py`
- `test_solve_quality.py`

Phase timing instrumentation remains `SolvePhaseCollector` /
`diagnostics.solve_phase_timing` (post-move:
`diagnostics.solve.solve_phase_timing`). Family microbenchmarks are recorded
under WP3–WP6 against this pre-move green baseline.
