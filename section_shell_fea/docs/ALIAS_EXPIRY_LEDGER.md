# Alias expiry ledger (WP9) — post **2026-11-30** retirement plan

Compatibility aliases remain registered until **2026-11-30**. This ledger
classifies every FE preprocessing path involved in the adapter-home programme.
Hard deletion of expiring aliases must not run before that date.

## Classification

| Path / symbol | Class | Post-expiry action |
|---------------|-------|--------------------|
| `pre_processing/mesh/` (impl + README) | **internal adapter home** | **Retain** |
| `pre_processing/loading/` FE adapters (`traction_face_load`, `shear_centre_loading_axis`, quarantine `beam_resultants_*`) | **internal adapter home** | **Retain** |
| `pre_processing/reference_thin_wall/` | **internal adapter home** | **Retain** |
| `pre_processing/mesh/naca_profile.py` | **retained FE-demo helper** | **Retain** |
| Public `section_shell_fea` barrel exports | **public worker export** | **Retain** (only public surface) |
| `pre_processing/topology/*` warning shims | **expiring alias** | **Remove** after zero inbound imports |
| `pre_processing/loading/thin_wall/*` warning shims | **expiring alias** | **Remove** after all callers use `reference_thin_wall` |
| `pre_processing/loading/builders|api|containers|io` Channel-B shims | **expiring alias** | **Remove**; import BSU `section_loads` |
| `pre_processing/loading/section_force_vector_to_nodal_load.py` | **expiring alias** | Prefer `processing...load_application` |
| `beam_resultants_to_nodal_load` | **retained regression helper** | **Retain** (quarantined) |
| BSU `mesh_utilities.section_mesh` / `load_utilities.section_loads` / `geometry_utilities.section_thin_wall_model` | **public sibling workers / kernel** | **Retain** — sole owners of generation / shaping / theory |

## Pre-expiry gates (must stay green)

1. Three adapter homes remain importable to worker-internal callers.
2. Expiring alias packages still emit `DeprecationWarning` and re-export only.
3. No FE facade calls `SectionMeshRunner.execute` / `SectionLoadsRunner.execute`.
4. BSU kernels remain the sole mesh-generation, load-shaping, and thin-wall-theory owners.

## Post-expiry checklist (after 2026-11-30)

1. Confirm zero inbound imports of each expiring alias (repo-wide grep).
2. Delete `pre_processing/topology/` package.
3. Delete `pre_processing/loading/thin_wall/` aliases.
4. Delete Channel-B shim trees under `loading/{builders,api,containers,io}` if unused.
5. Update `DEPRECATED_ALIASES.md` and deprecation register statuses/evidence.
6. Commit: `refactor(section-shell-fea)!: retire expired preprocessing deep aliases`.
7. Run deprecation checker + focused FE adapter / public-barrel / import-boundary tests.

## See also

- [`DEPRECATED_ALIASES.md`](DEPRECATED_ALIASES.md)
- [`.github/governance/deprecation_register.json`](../../../../../.github/governance/deprecation_register.json)
