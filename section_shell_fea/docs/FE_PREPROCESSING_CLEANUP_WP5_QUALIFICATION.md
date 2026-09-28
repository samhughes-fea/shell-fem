# FE preprocessing cleanup WP5 — qualification

**Programme:** FE preprocessing cleanup and element performance
**Baseline:** [`FE_PREPROCESSING_CLEANUP_WP0_BASELINE.md`](FE_PREPROCESSING_CLEANUP_WP0_BASELINE.md)
**Design properties:** generalised · unified · standardised · maintainable · exact

This report qualifies WP1–WP4 against the frozen WP0 oracle and confirms no
tolerance was widened.

## Before / after

| Topic | WP0 (before) | After WP1–WP4 |
|-------|--------------|---------------|
| Formulation selection | theory→factory `_ELEMENT_FACTORY_REGISTRY` façade | `formulation_id` primary via `registry._FORMULATIONS`; theory maps only through `default_formulation_id`; CalculiX on `shell_solve_registry` |
| Registration hooks | descriptor / factory / kernel / projector / constitutive_adapter | + optional `template_key_strategy`, `validate_element`, `validate_cache` (MITC4 registers concrete validators) |
| Element geometry record | `L_s` recomputed independently by assembly (blended), recovery (plain), buckling (plain) | `L_s` / `L_s_donnell` / `L_x` / `formulation_id` **persisted** on `ShellStiffnessObject` / `ShellForceObject`; recovery and buckling prefer `stiff.L_s` |
| Secondary kernel default | `resolve_kernel(theory or "mitc4")` (unsafe when `theory is None`) | `_resolve_secondary_kernel` prefers `cache.formulation_id` via `resolve_kernel_by_id`; `None` → `ShellTheory.FSDT`; strings mapped safely |
| MITC4 template reuse | rounded module `_KE_TEMPLATE_CACHE` keys | exact per-build `ShellFormulationTemplateCache` (float bits); module cache is compat-only, now exact + stats/clear |
| Phase timing seam | `formulation_factory` imported the **processing** collector (dependency inversion) | neutral `pre_processing/element_library/phase_timing.py`; processing `solve_phase_timing` re-exports it |
| `mitc4/__init__.py` | `from .element import *` | explicit `__all__` (element API + factory + projector) |
| `mitc4/utilities/__init__.py` | re-exported `collect_panel_abd_stack` | removed; consumers import from the constitutive home |
| Donnell ratio | `L_s_donnell` plain vs blended `L_s` | unchanged (intentional; still pinned by WP0) |

## Extension recipe

See [`../pre_processing/element_library/README.md`](../pre_processing/element_library/README.md)
§ "Adding a formulation (extension recipe)" for the descriptor → factory → kernel →
projector → register → build flow and the dependency diagram. The AST gate in
`tests/test_fe_preprocessing_wp1_extension.py` enforces that the generic modules
(`formulation_factory`, `registry`, `contracts`, `shell_element_base`,
`formulation_validation`) contain no top-level MITC4 imports and no hard-coded Gauss
rules. A non-MITC4 fake formulation is registered and driven end-to-end through
`formulation_id` alone, with a meta-path guard proving no fresh `mitc4` import occurs.

## Exactness — tolerances NOT widened

The WP0 frozen tolerances (`test_wp0_exactness_tolerances_are_frozen`) are unchanged:

- assembly `K`/`f` max abs `1e-10` (`test_formulation_cache_assembly_parity`)
- element `Ke` typed-vs-tuple atol `1e-12`
- `L_s` identity atol `0.0`
- bitwise-identical when arithmetic order is unchanged

Template reuse returns bitwise-identical `K_e` to a fresh assembly
(`mitc4/tests/test_mitc4_template_cache.py`), keys are exact float bits (no
rounding), and incomplete identities are refused. No constitutive, recovery, FI,
mesh, or solve-quality policy changed.

## WP2 L_s agreement

`tests/test_fe_preprocessing_wp2_geometry.py` pins that, with scaled `zones_outboard`
(blended `L_s` ≠ plain contour length), assembly / recovery / buckling now read the
**same** persisted blended `L_s` from the cache:

- persisted `stiff.L_s` == factory blended `_element_L_s`;
- secondary recovery output equals `recover_generalized(u, stiff.L_s, ...)` and
  differs from the plain-length recovery;
- geometric-stiffness scatter equals `assemble_panel_geometric_stiffness(L_s=stiff.L_s)`
  and differs from the plain-length `Kg`.

The original WP0 pin
(`test_zones_outboard_exposes_assembly_recovery_buckling_L_s_disagreement`) is left
intact: it exercises the low-level `_element_L_s` helpers directly (which still
recompute plain length), documenting the historical disagreement the cache now
closes at the consumer level.

## WP3 performance

See [`FE_PREPROCESSING_CLEANUP_WP3_TIMING.md`](FE_PREPROCESSING_CLEANUP_WP3_TIMING.md):
≈52% of per-element stiffness assemblies reuse an exact template on a representative
uniform mesh (85 elements → 41 unique templates, 44 hits, 0 refusals). No speculative
vectorisation was added (einsum path already vectorised over Gauss points).

## Remaining deprecations / gaps

- **Retired façade — theory-keyed element factory.** `register_element_factory` /
  `_ELEMENT_FACTORY_REGISTRY` removed (WP8). Native resolution is
  `formulation_id` / `default_formulation_id` only; CalculiX stays on
  `shell_solve_registry` (`compat-fe-element-factory-registry-facade` status
  `removed`). See
  `element_library/docs/FORMULATION_AND_SOLVE_DISPATCH.md`.
- **Removed re-export.** `mitc4.utilities.collect_panel_abd_stack` re-export removed
  (`removed-fe-mitc4-utilities-collect-panel-abd-reexport`); consumers use
  `pre_processing.constitutive.panel_abd`.
- **`pre_processing → processing` inversions (beyond phase timing).** Remaining
  ones are intentional `loading/` compatibility shims (`__init__.py`, `vis.py`,
  `section_force_vector_to_nodal_load.py`) that re-export the SFV **apply** step
  owned by `processing.static.operations.assembly.load_application`. The
  element-library timing inversion and the former
  `reference_thin_wall → cap_tie.geometry` (`kind_of`) inversion are eliminated
  (local `_panel_kind` helper).
- **Thin-wall reference/Vlasov adapters moved (WP4 D4.3.4 — completed).** The FE
  reference/Vlasov adapters (`section_vlasov.py`, `sectorial_warping.py`) now live in
  `pre_processing/reference_thin_wall/`; the canonical thin-wall *theory* remains in
  `geometry_utilities.section_thin_wall_model` (TW-002). `pre_processing/loading/thin_wall/`
  is now a time-bounded compatibility shim that re-exports from the new home and emits
  a `DeprecationWarning` (registered as `compat-fe-loading-thin-wall-shim`). Production
  callers (`calculix_ccx/thin_wall_loads.py`, `calculix_ccx/thin_wall_adapter.py`,
  `pre_processing/loading/shear_centre_loading_axis.py`,
  `pre_processing/containers/station_analysis_types.py`,
  `processing/static/diagnostics/plots/example_plots.py`,
  `processing/static/diagnostics/plots/thin_wall_plots.py`) and the FE tests now import
  the new path.
- **`outline_frames` callers migrated (WP4 D4.3.3 — completed).** The FE callers of
  `pre_processing/loading/builders/outline_frames` (`processing/static/diagnostics/plots/loading_vis.py`,
  `tests/test_section_frame_canon.py`, `pre_processing/loading/tests/test_vis.py`) now
  import the canonical
  `load_utilities.section_loads.surface_traction.outline_frames`;
  `builders/outline_frames.py` remains the `DeprecationWarning` re-export.
- **`pre_processing/loading/io` unused (WP4 D4.3.3b).** `load_field_json_payload`
  delegates to `SectionStationLoadField.to_json_dict()` and has no inbound consumers
  (the serialization round-trip test uses `to_json_dict` / `from_json_dict` directly).
  Registered as `compat-fe-loading-io-load-field-json` for removal once confirmed dead;
  not deleted in this cleanup.
- **MITC4 validation hooks** (`validate_element` / `validate_cache`) are invoked
  automatically by `build_formulation_cache` (see
  `element_library/docs/FORMULATION_VALIDATION_LIFECYCLE.md`). MITC4 registers
  concrete validators in `mitc4/validation.py`; the always-on
  `validate_formulation_cache_shapes` remains the first cache guard.

## Test results

Focused WP5 suite + new WP1/WP2/WP3 suites: **all green**.

```
pytest -c pytest.ini \
  tests/test_fe_preprocessing_wp0_baseline.py \
  tests/test_formulation_registry.py \
  tests/test_formulation_cache_assembly_parity.py \
  tests/test_element_agnostic_kernel.py \
  tests/test_element_formulation_import_boundary.py \
  tests/test_fe_preprocessing_wp1_extension.py \
  tests/test_fe_preprocessing_wp2_geometry.py \
  pre_processing/element_library/mitc4/tests/
```

Full `section_shell_fea` package: 573 passed, 1 pre-existing failure
(`test_shell_midline_strip_contract.py::test_solver_and_strip_mesh_agree_on_panel_element_counts`
— a CalculiX CCX cap-tie data-setup issue present in the prior `lastfailed` cache,
unrelated to WP1–WP5).

## Deferred FE follow-ons (out of dirty-tree triage)

These items stay **explicitly deferred** — they do not enter the BSU /
stiffness-selection / CDFAM commit stack. Compatibility shims remain registered
until **2026-11-30**.

| Follow-on | Status | Boundary |
|---|---|---|
| Relocate `traction_face_load` helpers with warning shims | Deferred | Keep current homes; move only with a registered expiry shim |
| Relocate `shear_centre_loading_axis` with warning shims | Deferred | Same shim policy as mesh/loading compatibility layer |
| MITC4 `validate_element` / `validate_cache` hooks | **Done (WP7)** | Lifecycle specified; generic wiring; MITC4 concrete validators registered |
| Retire `_ELEMENT_FACTORY_REGISTRY` façade | **Done (WP8)** | Native graph only; CalculiX stays on solve dispatch |
| Delete `pre_processing/mesh` and `pre_processing/loading` shims | **Not in scope** | Shims stay; registered expiry **2026-11-30** |

Preserved shim presence (must remain until expiry):

- `pre_processing/mesh/` (e.g. `shell_mesh_map.py` re-export)
- `pre_processing/loading/` (package + thin-wall / outline_frames / io re-exports)
