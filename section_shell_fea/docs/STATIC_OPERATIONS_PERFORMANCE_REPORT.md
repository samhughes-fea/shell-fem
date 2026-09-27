# Static operations performance report

**Date:** 2026-09-27  
**Programme:** Static processing subpackage refactor (WP0–WP6).

## Summary

Operation and diagnostic modules were reorganized into family subpackages with
legacy-path module aliases. Measured hot-path changes inside the operation
families:

| Family | Change | Exactness |
|--------|--------|-----------|
| Assembly | Preallocated COO buffers; `np.add.at` force scatter | Parity pins green |
| Reduction | `ReductionOperator` separates static `T` from per-trial `TᵀKT` | Identity-keyed reuse |
| Solution | `k_reduced_identity` guards SuperLU reuse; `solve_many` factor-once | Stale factor fail-closed |
| Provenance | Runner `provenance_policy` default `lazy` | Maps materialize on request |

## Microbenchmark (synthetic assembly)

Tiny two-element shared-DOF mesh (`test_static_processing_performance_pins`):

- Duplicate-safe force scatter verified (`np.add.at`).
- `validate=True` (default) and trusted `validate=False` produce identical `K`, `f`, `nnz`.

Representative synthetic mesh (200 elements × 24 DOF, `n_gdof=500`, 30 warm
repeats after one warm-up on this workstation):

| Metric | Value |
|--------|------:|
| Assembly median | 7.99 ms |
| Assembly p90 | 9.01 ms |

These are report-only timings (not CI hard gates). Exactness is owned by
`test_static_processing_performance_pins` and the assembly-parity suite.

Focused suite after cutover: **455 passed** in `section_shell_fea/tests`
(1 pre-existing CalculiX/cap-tie midline contract failure unrelated to this
programme: `test_solver_and_strip_mesh_agree_on_panel_element_counts`).

## Architecture notes

Canonical layouts:

```text
operations/{contracts,preparation,assembly,reduction,solution,recovery}/
diagnostics/{solve,equilibrium,convergence,parity,plots}/
```

Root modules are `sys.modules` aliases of the canonical bodies so private
test helpers remain importable for one deprecation cycle.

## 2D-F integration pin

Full 16-station `--no-plots` BSP timing comparison is an exclusive-window
follow-on (D6.4). Operation-family pins and the full FE unit suite are the
qualification gate for this landing; re-run stage timings after the next
exclusive 2D-F job and record `stage_timings_s.section_selection` against WP0.
