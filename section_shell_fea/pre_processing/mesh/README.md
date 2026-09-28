# mesh — FE mesh adaptation over BSU section_mesh

Internal FE home for adapting BSU strip meshes into FE assembly maps.

↑ Parent: [`../README.md`](../README.md) · Package: [`../../README.md`](../../README.md)

## Ownership

| Package | Owns |
|---------|------|
| [`mesh_utilities.section_mesh`](../../../../mesh_utilities/section_mesh/README.md) | Strip-mesh generation and typed handoff (`SectionStripMesh`, `ShellStripMeshingInputs`, `SectionMeshRunner`) |
| **`pre_processing.mesh` (this home)** | FE maps, global numbering, junction clustering, `BoundSectionShellInputs` |
| Public `section_shell_fea` barrel | Canonical worker export surface |

Import `SectionStripMesh` and related products from the public
`mesh_utilities.section_mesh` barrel. This package does **not** re-export the
upstream mesh worker and must not call `SectionMeshRunner.execute()` during FE
execution — orchestration supplies frozen strip products.

Thin-wall NACA demo helpers live under
[`../../visualisation/demos/thin_wall_naca_demo_profile.py`](../../visualisation/demos/thin_wall_naca_demo_profile.py);
`naca_profile.py` is a warning alias (expires **2027-03-31**).

`pre_processing.topology` is a logic-free warning-alias package for one
migration cycle (expires **2026-11-30**).

## Exact I/O

| Path | Role |
|------|------|
| [`panel_node_assembly.py`](panel_node_assembly.py) | Node maps, panel endpoints, junction clustering |
| [`global_numbering.py`](global_numbering.py) | Global node assignment / cluster frames |
| [`section_shell_assembly_map.py`](section_shell_assembly_map.py) | Connected multi-panel assembly maps |
| [`api/bound_section_shell_inputs.py`](api/bound_section_shell_inputs.py) | `BoundSectionShellInputs` handoff DTO + `bind_section_shell_inputs` |
| [`shell_mesh_map.py`](shell_mesh_map.py) | Warning alias → `panel_node_assembly` (2027-03-31) |
| [`connected_panel_topology.py`](connected_panel_topology.py) | Warning alias → `section_shell_assembly_map` (2027-03-31) |
| [`api/shell_panel_topology.py`](api/shell_panel_topology.py) | Warning alias → `bound_section_shell_inputs` (2027-03-31) |
| [`naca_profile.py`](naca_profile.py) | Warning alias → demo NACA profile (2027-03-31) |

## See also

- [`../../../../mesh_utilities/section_mesh/README.md`](../../../../mesh_utilities/section_mesh/README.md)
- [`../../docs/TAXONOMY.md`](../../docs/TAXONOMY.md)
