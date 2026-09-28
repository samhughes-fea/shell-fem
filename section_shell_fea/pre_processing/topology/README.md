# topology — FE mesh maps / global numbering

↑ Parent: [`../README.md`](../README.md) · Package: [`../../README.md`](../../README.md)

Canonical home for FE-owned **topology** seams (formerly under
`pre_processing.mesh`, which now re-exports from here as a compatibility shim).

## Purpose / ownership

| Owns | Does **not** own | Why |
|------|------------------|-----|
| FE **maps** over the strip mesh (`PanelGlobalMap`, `panel_maps_from_section_strip_mesh`) | Strip-mesh **generation** ([`mesh_utilities.section_mesh`](../../../../mesh_utilities/section_mesh/README.md)) | Mesh knots/geometry are upstream |
| Global node/DOF numbering (`global_numbering.py`) | Assembly / constraints / solve ([`../../processing/`](../../processing/README.md)) | Topology stops at index maps |
| `ShellPanelTopology` handoff DTO (`api/`, `connected_panel_topology.py`) | Constitutive / element `Ke` (siblings) | Separation of concerns |

## Exact I/O

| Symbol / path | Role |
|---------------|------|
| [`shell_mesh_map.py`](shell_mesh_map.py) | `PanelGlobalMap`, `panel_maps_from_section_strip_mesh`, `normal_2d` |
| [`global_numbering.py`](global_numbering.py) | Connected global node/DOF numbering |
| [`connected_panel_topology.py`](connected_panel_topology.py) | Connected multi-panel topology assembly |
| [`api/shell_panel_topology.py`](api/shell_panel_topology.py) | `ShellPanelTopology` handoff DTO |

| In | `SectionStripMesh` (upstream), material `zones` |
| Out | Per-panel maps + global numbering consumed by `simulation_runner` / assembly |

## Compatibility

`pre_processing/mesh/*.py` are thin shims importing from this package (tracked in
the deprecation register). New code imports from `pre_processing.topology`.

## See also

- [`../README.md`](../README.md) — pre_processing homes
- [`../../mesh_utilities/section_mesh/README.md`](../../../../mesh_utilities/section_mesh/README.md) — strip-mesh generation (upstream)
- [`../../processing/README.md`](../../processing/README.md) — assembly/solve consumers
