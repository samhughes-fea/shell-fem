# FE preprocessing cleanup WP3 — attributable timing note

**Programme:** FE preprocessing cleanup and element performance
**Scope:** formulation-owned caching / MITC4 optimisation (WP3)

## What changed

- A generic per-build `ShellFormulationTemplateCache`
  (`pre_processing/element_library/template_cache.py`) is created once per
  `build_formulation_cache` call and handed to the MITC4 factory via the optional
  `template_cache=` keyword.
- `Mitc4Quad4Element.element_stiffness_matrix` computes an **exact** template key
  (float64 bytes of `L_s`, `L_x`, `ndof`, and the constitutive blocks A/B/D/A_s or
  the legacy ABD tuple + thickness + G_eff — never `round()`), and reuses the built
  `K_e` for identical keys within one build. The key is computed without
  constructing the shape/material operators, so a hit also skips
  `ShapeFunctionOperator` / `MaterialStiffnessOperator` work.
- The module-level `_KE_TEMPLATE_CACHE` (reached only via the legacy
  `mitc4_stiffness(...)` entrypoint) was moved from rounded keys to exact float
  bits and given `clear_ke_template_cache()` / `ke_template_cache_stats()` for
  tests. Production `Mitc4Quad4Element` no longer touches it — it uses the per-build
  cache instead (compatibility-only module cache).

## Attributable measurement

Harness: `tests/test_fe_preprocessing_wp0_baseline.py` D0.1 style build of a
NACA2412 single-web section, `n_elements_per_panel=8` (7 panels).

| Quantity | Value |
|----------|-------|
| Elements per build | 85 |
| Unique stiffness templates | 41 |
| Within-build cache hits | 44 |
| Refusals (incomplete identity) | 0 |
| Reuse fraction (hits / elements) | ≈ 52% |
| Warm median `build_formulation_cache` | ~145 ms (this machine; indicative) |

Interpretation: roughly half of per-element stiffness assemblies on a uniform mesh
collapse to an exact template hit (uniform `L_s` within a panel + shared panel
material). The reuse is **bitwise-identical** to a fresh assembly (verified in
`mitc4/tests/test_mitc4_template_cache.py`), preserving the WP0 oracle.

## Exactness

- Keys are exact (float bits); a one-ULP `L_s` difference is a distinct template
  (pinned by `test_per_build_cache_exact_no_rounding_collision` and the module-cache
  equivalent). No rounded cache keys are used for production reuse.
- Incomplete identities (`key is None`) are **refused** (miss, never stored).

## Vectorisation decision

The einsum operator path (`assemble_mitc4_stiffness`) is already vectorised over
Gauss points. No additional speculative vectorisation was introduced: the measured
win came from exact template reuse, not from re-shaping the inner kernel. Per the
WP3 brief, speculative vectorisation with no measured bottleneck was skipped.
