# FE preprocessing cleanup WP0 — exactness and performance baseline

**Date:** 2026-09-28  
**Programme:** FE preprocessing cleanup and element performance  
**Plan:** generalised / unified / standardised / maintainable / exact element library

## Exactness contract (non-negotiable)

- Current production MITC4 formulation-cache path is the numerical oracle.
- Later WPs may change scheduling, allocation, batching, registration shape, and
  exact-key reuse only.
- No constitutive, recovery, FI, mesh, or solve-quality policy changes.
- No tolerance widening, diagnostic suppression, mesh coarsening, or tier omission.
- Incomplete or uncertain cache identities are misses (refuse reuse).
- Fresh vs cached results must remain bitwise-identical where arithmetic order is
  unchanged; existing engineering tolerances apply elsewhere.

## Frozen production behaviour (WP0)

| Topic | Current behaviour (freeze) |
|-------|----------------------------|
| Default formulation | `ShellTheory.FSDT` → `MITC4-FSDT` |
| Production Ke path | `Mitc4Quad4Element` → `assemble_mitc4_stiffness` (no `_KE_TEMPLATE_CACHE`) |
| Typed constitutive | `PanelConstitutiveSpec.fsdt` preferred; tuple ABD is compatibility |
| Outboard length | Factory `_element_L_s` averages inboard/outboard when `zones_outboard` set |
| Donnell length | `L_s_donnell` remains plain inboard contour length |
| Recovery / buckling length | Recompute inboard contour length (disagrees with blended assembly `L_s`) |
| Secondary theory default | `resolve_kernel(theory or "mitc4")` — unsafe when `theory is None` |

## D0.1 — Element-path benchmark harness

| Case | Harness |
|------|---------|
| Uniform laminate panel | `tests/test_fe_preprocessing_wp0_baseline.py` |
| Mixed laminate panels | same |
| Timing phases | `element_stiffness_matrices`, `element_force_vectors` via `SolvePhaseCollector` |
| Allocation | `tracemalloc` peak during `build_formulation_cache` |

Commands (from repo root):

```bash
pytest -c pytest.ini src/blade_structure_utilities/fem_utilities/section_shell_fea/tests/test_fe_preprocessing_wp0_baseline.py -q
```

Performance claims use warm medians over repeated runs; WP0 records the harness
and baseline quantities. WP5 compares against this freeze.

## D0.2 — Numerical exactness pins

| Case | Pin |
|------|-----|
| Formulation-cache vs legacy inline assembly | `test_formulation_cache_assembly_parity` |
| MITC4 operator stiffness einsum vs Gauss loops | `test_mitc4_operator_stiffness` |
| Typed Spec vs tuple ABD Ke | `test_fe_preprocessing_wp0_baseline` |
| Outboard blended `L_s` vs recovery/buckling | `test_fe_preprocessing_wp0_baseline` |
| Donnell plain length preserved | same |

Oracle quantities under existing repository tolerances:

- element `K_e`, `F_e`
- sparse assembly `K`, `f` (existing parity suite)
- effective `L_s` / `L_s_donnell`
- formulation ID and DOF layout identity

## Exit gate

Focused WP0 suite green before WP1+ land. Full `section_shell_fea/tests`
qualification is WP5.
