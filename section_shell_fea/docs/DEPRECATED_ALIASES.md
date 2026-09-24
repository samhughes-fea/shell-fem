# Deprecated aliases — `section_shell_fea`

> **Working inventory:** live warn / silent / legacy / retained rows only.
> Historical Wave 1–6 removals are summarized in the [appendix](#appendix-removed-waves)
> (not a working checklist). Prefer the two FE runners /
> `run_section_linear_static_from_resultants`.

**Policy:** follow [`docs/architecture/DEPRECATION_REGISTER.md`](../../../docs/architecture/DEPRECATION_REGISTER.md)
— one-cycle compatibility where inbound usage is non-zero; emit
`DeprecationWarning` on public entry points only (avoid hot loops); remove when
inbound usage is zero.

**Status legend**

| Status | Meaning |
|--------|---------|
| `silent` | Alias present; no runtime warning |
| `warns` | Emits `DeprecationWarning` |
| `legacy` | Documented as regression-only; may not yet warn |
| `retained` | Kept intentionally (diagnostics / named convenience) |

---

## Live surface (migrate callers)

| Item | Canonical | Defined in | Status |
|------|-----------|------------|--------|
| `SectionLinearStaticRunner(about_yz=...)` | `about_xy=...` | `simulation_runner/section_linear_static_runner.py` | **warns** |
| `BeamResultantVector.about_yz` / `.with_about_yz` | `.about_xy` / `.with_about_xy` | `section_contract/beam_resultant_vector.py` | **warns** |
| `SectionDeformationAverage1a.about_yz` | `.about_xy` | `processing/static/results/compute_primary/section_deformation_average.py` | **warns** |
| `build_shell_panel_topology` | build `ShellMeshInputs` via mesh handoff, then `shell_panel_topology_from_shell_inputs` | `pre_processing/mesh/api/shell_panel_topology.py` | **warns** (tests still use factory wrapper) |
| `ShellPanelTopology.panels` | `.zones` | same | **warns** |
| `ThinWallResult.panels` / related `.panels` | `.zones` | FE result types | **warns** / migrate |
| Scalar `N,Vy,Vz,M22,Mz,T` kwargs alone | `resultants=BeamResultantVector` | `SectionLinearStaticRunner` | **silent** (centroid-referenced) |
| `SectionShellRecoveryBundle.y_sc` / `.z_sc` | `.y_e` / `.z_e` (elastic centroid, **not** shear centre) | `station_analysis_types.py` | **silent** dual-write |
| `assemble_global_shell` | `assemble_section_from_zones` | `processing/static/operations/global_shell_assembly.py` | **legacy** (8-tuple shim through next major) |
| `connected_assembly_core` numbering/MPC symbols | `pre_processing.mesh.global_numbering` / `processing.static.constraints` | `processing/static/operations/connected_assembly_core.py` | **silent** shim |
| `operations/cap_tie_constraints` | `processing.static.constraints.cap_tie` | `processing/static/operations/cap_tie_constraints.py` | **silent** shim |
| `SectionLinearStaticRunner.from_cache` / `from_global` | `solve_constrained_system` | `simulation_runner/section_linear_static_runner.py` | **legacy** (explicit constraints only) |
| `beam_resultants_to_nodal_load` on production execute | frozen `SectionForceVector` + `section_force_vector_to_nodal_load` | `pre_processing/loading/beam_resultants_to_nodal_load.py` | **legacy** (regression/reference) |
| `USE_TAPERED_INTERLAMINAR` / `_legacy_channel_a_*tapered*` | prismatic quaternary IL on linear track | `shell_fi_eval.py` | quarantined (default `False`) |
| `hashin_il` / `ShellHashinFiResult.interlaminar` | FSDT interface-shear diagnostic (`interface_shear`) | `compute_quaternary` | **silent** compatibility; excluded from governing |
| `max_fi_hashin` | plane-stress Hashin native FI; prefer `max_utilization_governing` | station evaluate / sizing trial | **silent** alias |
| `fsdt_periodic_hashin_audit` / `fsdt_periodic_hashin` | `failure_assessment` | `io/job_outputs.py`, `io/station_json.py` | **silent** dual-write through next major |

### Plot naming note (`vis.py`)

`save_shell_dashboard_figure` / `save_shell_section_mesh_figure` are **preferred**
theory-agnostic wrappers over `save_mitc4_v2_*`. Do **not** deprecate the
`save_shell_*` names.

---

## Out of scope (not scheduled for removal)

| Symbol | Why not deprecated |
|--------|-------------------|
| `MITC4_KERNEL` | Production default kernel singleton |
| `section_solve_legacy_dict` / `.legacy_dict()` | Retained for diagnostics |
| `build_shell_constitutive_stack_from_legacy_section_properties` | Names input shape, not a rename alias |
| `interface_constraint_mode="shared"` | Constraint mode enum value |
| `save_shell_dashboard_figure`, `save_shell_section_mesh_figure` | Preferred names |
| FE `shape_section_force_vector*` convenience / demo-bind | Named leaks — see FE-row matrix / BSU handoff |
| Constitutive re-export of demo `WallLaminate` | Named leak |

---

## Appendix: removed waves (historical)

MITC4 → theory-agnostic renames, Channel-A periodic / K6 / clamped FE entry,
`engine/` package retirement, unused `types/` barrel (stdlib shadowing), Pipeline-B →
`ShellPanelTopology` rename, and warn-stage SoC shims (`engine.panel_abd`,
`io.example_plots`, `bc_mode="full_clamp"`, …) are **done**. The worker
`section_shell_fea/api/` barrel is **retained** as the explicit
`API_STANDARDS` seam; do not reintroduce a shadowing `types/` package.
Details live in git history and
[`DEPRECATION_REGISTER.md`](../../../docs/architecture/DEPRECATION_REGISTER.md).

```mermaid
flowchart LR
  silent[Silent aliases]
  warn[DeprecationWarning]
  migrate[Migrate callers]
  remove[Delete aliases]
  silent --> warn --> migrate --> remove
```

### Implementation notes

- Prefer migrating **in-repo** callers in the same change as warnings.
- Do **not** warn inside per-element / per-panel hot loops.
- Conventional commits: `docs(section_shell_fea): …` for inventory;
  `refactor(section_shell_fea): …` for removals.
