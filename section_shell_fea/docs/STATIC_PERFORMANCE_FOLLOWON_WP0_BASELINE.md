# Static performance follow-on WP0 — exactness and timing baseline

**Date:** 2026-09-27  
**Programme:** Static processing performance follow-ons (post `7f537a6a`).

## Exactness contract (non-negotiable)

- Current scalar / uncached / stepped paths are the oracle.
- Optimizations may change scheduling, allocation, batching, and exact-key reuse only.
- No constitutive, constraint, recovery, FI, or solve-quality policy changes.
- No tolerance widening, diagnostic suppression, mesh coarsening, or tier omission.
- Incomplete or uncertain cache identities are misses.
- Changed `K` always rebuilds `K_reduced` and SuperLU.

## D0.1 — Numerical oracle ownership

Owned by focused pins (not binary golden files):

| Case | Pin / suite |
|------|-------------|
| Assembly K/f / nnz / symmetry | `test_static_processing_performance_pins` |
| ReductionOperator T reuse + K_rr refresh | same |
| Factor-once multi-RHS + stale LU refuse | same + WP1 pins |
| Lazy vs eager provenance | same |
| Four-tier cascade + FI | `test_runner_four_tier_pipeline`, `test_solve_quality` |
| Constraint composer / cap-tie | `test_constraint_composer`, cap-tie suites |

Oracle quantities compared under existing repository tolerances:

- sparse `K`, `f`, `T`, `K_reduced`, independent DOFs
- displacement, reactions, residuals
- secondary resultants; tertiary ply strain/stress / `tau_iface`
- every FI mode, trust flags, governing panel/element/ply/face/mode

## D0.2 — Production timing baseline

Phase names in `SolvePhaseCollector` (WP1 restores production `execute()` coverage):

- formulation: `element_stiffness_matrices`, `element_force_vectors`
- compose: `compose_constraints` (added WP1)
- kinematics: constrain / reduce / solve / reconstruct
- cascade: primary / secondary / tertiary / quaternary

Warm-median and allocation evidence is recorded under WP5 against this freeze.
Assembly microbenchmark (post-7f537a6a, 200×24 DOF mesh): median ≈ 8 ms (report-only).

## D0.3 — Exactness policy harness

- Bitwise equality where arithmetic order is unchanged (e.g. single primary matvec).
- Existing engineering tolerances where batching reorders floating-point ops; investigate every delta.
- Fail on changed result shape, trust state, governing provenance, or diagnostic availability.

## Exit gate

Focused baseline suites green before WP1+ land. Full `section_shell_fea/tests` qualification is WP5.
