# Deprecated aliases — `section_shell_model`

Inventory of live compatibility aliases and the phased plan to retire them.

**Policy:** follow [`docs/architecture/DEPRECATION_REGISTER.md`](../../../docs/architecture/DEPRECATION_REGISTER.md) — one-cycle compatibility where inbound usage is non-zero; emit `DeprecationWarning` on public entry points only (avoid hot loops); remove when inbound usage is zero.

**Status legend (this doc):**

| Status | Meaning |
|--------|---------|
| `silent` | Alias present; no runtime warning |
| `warns` | Emits `DeprecationWarning` |
| `legacy` | Documented as regression-only; may not yet warn |
| `removed` | Deleted; retained here only as historical note |

---

## A. MITC4 → theory-agnostic renames

Largest compatibility surface. Prefer the canonical names in all new code.

| Alias | Canonical | Defined in | Status | Inbound usage |
|-------|-----------|------------|--------|---------------|
| `Mitc4PanelMesh` | `StripPanelMesh` | formerly `section_strip_mesh.py` | **removed** (Wave 4) | — |
| `Mitc4Cluster` | `StripJunctionCluster` | same | **removed** (Wave 4) | — |
| `Mitc4SectionMesh` | `SectionStripMesh` | same | **removed** (Wave 4) | — |
| `build_mitc4_mesh` | `build_section_strip_mesh` | formerly `section_strip_mesh.py` | **removed** (Wave 4) | — |
| `discretise_panel` | `discretise_strip` | formerly `strip_discretise.py` / `panel_discretise.py` | **removed** (Wave 2) | — |
| `all_panel_mitc4_results` | `all_panel_results` | formerly `types.py` `@property` | **removed** (Wave 2) | — |
| `all_panel_mitc4_diagnostics` | `all_panel_diagnostics` | same | **removed** (Wave 2) | — |
| `mitc4_panel_abd` | `panel_abd` | same | **removed** (Wave 2) | — |
| `mitc4_panel_thickness_m` | `panel_thickness_m` | same | **removed** (Wave 2) | — |
| `mitc4_panel_G_eff` | `panel_G_eff` | same | **removed** (Wave 2) | — |
| `mitc4_panel_labels` | `panel_labels` | same | **removed** (Wave 2) | — |
| `.mitc4` | `.shell_solve` | `SectionShellRecoveryBundle` | **removed** (Wave 2) | — |
| `ShellMitc4Result` | `ShellPanelSolveResult` | formerly `types.py` | **removed** (Wave 1) | — |
| `collect_mitc4_panel_clt_stack` | `collect_panel_abd_stack` | formerly `mitc4_solve_orchestrator.py` | **removed** (Wave 1) | — |
| `solve_global_coupled_mitc4` | `solve_global_coupled_shell` | formerly `engine/global_shell_assembly.py` | **removed** (Wave 4) | — |
| `build_panel_topology` | `build_section_v2` | formerly `topology.py` | **removed** (Wave 1) | — |
| `SubcomponentMidline` | `ShellMidlineStrip` | formerly `shell_inputs_from_section.py` | **removed** (Wave 1) | — |

### Plot naming note (`vis.py`)

`save_shell_dashboard_figure` / `save_shell_section_mesh_figure` are **preferred** theory-agnostic wrappers over `save_mitc4_v2_*`. Do **not** deprecate the `save_shell_*` names; prefer migrating callers *toward* them and treating `save_mitc4_v2_*` as the implementation detail.

---

## B. Already warning (`DeprecationWarning`)

| Alias / trigger | Canonical | Defined in | Status |
|-----------------|-----------|------------|--------|
| kwargs `ccx_work_dir`, `ccx_exe` on `run_section_periodic_bc` / ~~`run_section_with_mitc4_shell`~~ / `run_section_periodic_bc_from_load_field` | `external_backend_options={'work_dir', 'exe'}` | public API only | **removed** (Wave 1) |
| `bc_mode="full_clamp"` | `bc_mode="contour_support"` | formerly `engine/connected_assembly_core.normalize_bc_mode` | **removed** (Wave 5) |
| `engine.panel_abd` | `pre_processing.constitutive.panel_abd` | formerly `engine/panel_abd.py` | **removed** (Wave 5) |
| `io.example_plots` | `processing.diagnostics.example_plots` / `vis` | formerly `io/example_plots.py` | **removed** (Wave 5) |
| `_ensure_stress_imports` | `_vendor_paths.ensure_stress_model_lib_path` | formerly `_utils/_stress_model_path.py` | **removed** (Wave 5) |

(CCX backend native kwargs on `calculix_ccx.periodic.run_section_periodic_bc_ccx` etc. are **not** the same surface and remain.)


---

## C. Documented-deprecated / legacy adapters

| Item | Canonical | Defined in | Status |
|------|-----------|------------|--------|
| `run_section_with_mitc4_shell` / `api/legacy_clamped.py` | `run_section_periodic_bc(..., shell_theory=ShellTheory.FSDT)` | formerly `api/legacy_clamped.py` (+ package `__init__`) | **removed** — clamped section FE path deleted; periodic is the only section shell FE entry |
| `reduce_propagate_solve` | `SectionLinearStaticRunner.from_global(...).solve()` | formerly `engine/connected_assembly_core.py` | **removed** (Wave 3) |
| `section_solve_legacy_dict` / `.legacy_dict()` | typed `SectionLinearStaticResult` | `simulation_runner/section_linear_static.py` | retained for diagnostics (not scheduled) |
| `loading_axis_yz=` on `run_section_periodic_bc` / tapered / CCX | `resultants=BeamResultantVector(..., about_yz=...)` | `api/periodic.py`, `pre_processing/loading/beam_resultant_handoff.py` | **warns** (`DeprecationWarning`) |
| Scalar `N,Vy,Vz,My,Mz,T` kwargs alone | `resultants=BeamResultantVector` | same | **silent** shim (centroid-referenced); migrate callers |
| `SectionShellRecoveryBundle.y_sc` / `.z_sc` | `.y_e` / `.z_e` (elastic centroid, **not** shear centre) | `station_analysis_types.py` | **silent** dual-write; prefer `y_e`/`z_e` |

---

## D. Out of scope (not scheduled for removal)

| Symbol | Why not deprecated |
|--------|-------------------|
| `run_section_periodic_bc_mitc4` | FSDT implementation behind theory dispatch — not an alias |
| `MITC4_KERNEL` | Production default kernel singleton |
| `build_shell_constitutive_stack_from_legacy_section_properties` | Names input shape (legacy section-property objects), not a rename alias |
| `interface_constraint_mode="shared"` | Constraint mode enum value, not a symbol alias |
| `save_shell_dashboard_figure`, `save_shell_section_mesh_figure` | Preferred names; wrap `save_mitc4_v2_*` |

Already-removed Phase 4–8 shims (`compute_secondary_from_cache`, `compute_tertiary_hashin`, tertiary Hashin property shims, `panel_station_shell_resultants`, `run_section_with_shell`, HSDT package, top-level `core/` / `element_library/` shims, etc.) are tracked only in the architecture register as `removed` / `completed`.

---

## Deprecation plan

```mermaid
flowchart LR
  silent[Silent MITC4 aliases]
  warn[DeprecationWarning]
  migrate[Migrate callers]
  remove[Delete aliases]
  silent --> warn --> migrate --> remove
```

### Wave 0 — Document

- This file + register rows in `DEPRECATION_REGISTER.md`.
- Correct stale in-package docs that still claim removed shims warn or exist.

### Wave 1 — Low-risk / low-inbound — **done**

Migrate in-repo callers, then remove (or warn-once if any external use remains):

1. `collect_mitc4_panel_clt_stack`
2. `ShellMitc4Result`
3. `build_panel_topology` / `SubcomponentMidline`
4. Finish CCX kwargs `ccx_work_dir` / `ccx_exe` (already warn → drop after migration)

### Wave 2 — Heavy MITC4 naming layer — **done** (in-repo)

Migrate in-repo callers; warn-once on residual public entry points; delete hot-path properties:

1. Mesh: `build_mitc4_mesh` / `Mitc4*` — dropped from public `__all__`; warn-once wrappers retained (deleted in Wave 4)
2. Discretisation: callers → `discretise_strip`; deleted `panel_discretise.py` shim and `discretise_panel` alias
3. Bundle fields: `*_mitc4_*` properties / `.mitc4` **removed** after migrating all in-repo readers
4. Solver: callers → `solve_global_coupled_shell`; warn-once `solve_global_coupled_mitc4` wrapper retained (deleted in Wave 4)

### Wave 3 — Legacy clamped FE surface — **done** (deleted)

- `run_section_with_mitc4_shell` / `api/legacy_clamped.py` **removed**; periodic `run_section_periodic_bc` is the only section shell FE path
- `reduce_propagate_solve` removed; `periodic_traction_superposition` uses `SectionLinearStaticRunner` directly
- `section_solve_legacy_dict` / `.legacy_dict()` retained as diagnostic helpers (typed result is preferred)

### Wave 4 — Cleanup — **done**

- Deleted residual warn-once wrappers: `build_mitc4_mesh`, `Mitc4*` `__getattr__`, `solve_global_coupled_mitc4`
- Register rows marked `removed` / `completed`; status columns in this file updated

### Wave 5 — Warn-stage SoC shims — **done**

Zero inbound in-repo usage after one-cycle `DeprecationWarning`:

1. `engine/panel_abd.py` — deleted; use `pre_processing.constitutive.panel_abd`
2. `io/example_plots.py` — deleted; use `processing.diagnostics.example_plots` / `vis`
3. `bc_mode="full_clamp"` alias — removed; use `contour_support`
4. `_utils/_stress_model_path._ensure_stress_imports` — deleted; production `section_shell_model` callers no longer bootstrap the vendor path (demo scripts only)

---

## Implementation notes

- Prefer migrating **in-repo** callers (tests, examples, `section_optimisation`, CCX) in the same change as warnings for Wave 1.
- Do **not** warn inside per-element / per-panel hot loops.
- Conventional commits: `docs(section_shell_model): …` for inventory; `refactor(section_shell_model): …` for removals.
