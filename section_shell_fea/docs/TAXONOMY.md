# Shell model taxonomy

Canonical glossary for the FE geometry and constitutive hierarchy used by
`section_shell_fea`. Layout family codes (`SystemType{X}{Y}-{Z}`) live in
[`../../blade_structure_utilities/geometry_utilities/section_geometry/docs/reference/system_type_xyz_taxonomy.md`](../../blade_structure_utilities/geometry_utilities/section_geometry/docs/reference/system_type_xyz_taxonomy.md)
and answer a different question; this document covers station → element.

Results tiers (primary → quaternary) are documented in
[`processing/static/results/RESULTS_DESIGN.md`](../processing/static/results/RESULTS_DESIGN.md).

## Hierarchy

```text
Station (z_station)                    — analysis slice with spanwise depth
  └── Section                          — closed 2D cross-section assembly
        └── Subcomponent               — structural role: skin | web | spar cap
              └── Strip                — FE-agnostic mid-surface polyline handoff
                    └── MaterialZone   — constitutive region (constant t, material)
                          └── Element  — MITC4 (or other) shell quad; inherits zone material
```

| Term | Meaning | Primary types |
|------|---------|---------------|
| **Station** | Analysis slice at spanwise coordinate `z_station`. Owns a section plus depth along the beam axis (unit-strip / strip width). | `ShellSectionStationGeometry`, `ShellSectionStationAnalysis`, `SectionDefinition.station_z` |
| **Section** | Closed 2D assembly of subcomponents in the cross-section plane (no spanwise extent by itself). | `MultiCellSection`, `SectionStripMesh`, `SectionDefinition` |
| **Subcomponent** | Structural role: outer skin, shear web, or spar cap. May own **one or more** material zones when thickness/layup varies along the contour. | `OuterSkin` / `ShearWeb` / `SparCap`; `ShellMidlineStrip.kind` ∈ `{skin,web,cap}` |
| **Strip** | FE-agnostic mid-surface polyline + thickness handoff (often 1:1 with a zone; skin may split into multiple strips). | `ShellMidlineStrip` |
| **MaterialZone** | Definitive **constitutive region**: one laminate or isotropic material, one thickness, one FI criterion — shared by all its elements. Not an FE element. (`Panel` is a compat alias.) | `MaterialZone`, `StripPanelMesh`, `panel_abd_stacks[pi]`, `panel_constitutive_kind` |
| **Element** | One MITC4 (or other) shell quad along the zone contour; inherits material from its zone; spanwise edge length = station strip depth. | `ShellStiffnessObject`, `ShellPanelResultants` (per-element despite the name) |

### Ownership (separation of concerns)

| Role | Object | Owns |
|------|--------|------|
| Structural part | **Subcomponent** | Kind (skin/web/cap), topology role; may contain **N material zones** |
| Constant constitutive region | **MaterialZone** | Thickness, material, isotropic/orthotropic kind, ABD, FI criterion |
| FE discretisation | **Element** | Geometry/DOF only; material via parent zone |

Constitutive ownership is :class:`~blade_structure_utilities.material_utilities.section_bind.MaterialZone`
(`Panel` remains a compatibility alias). Prefer `MaterialZone` over `LayupZone`
(the latter excludes isotropic metals). Mesh/FE topology still uses “panel” in
names such as `ShellPanelTopology.zones` (constitutive list on the topology DTO),
`PanelGlobalMap`, and `StripPanelMesh`
(mesh units, not the constitutive type). Aerospace English often reserves “panel”
for a buckling bay (`PanelBuckling*`); that sense is separate. Kind-level
`skin_lam` / `cap_lam` / `web_lam` remains the default bind policy; per-strip
overrides use `strip_lam_by_label=` on `bind_shell_material_zones`.

## Section vs station

- **Section** = the 2D cut: closed assembly of subcomponents in the cross-section plane.
- **Station** = that section placed at spanwise `z`, with a finite depth along the beam axis (unit strip / strip width used by the shell elements’ spanwise edges).

A station therefore has a depth dimension; a section alone does not.

## Force-vector glossary (do not conflate)

Map load objects onto the blade → station → section → element ladder by
**where the data lives**. Never say “force vector” unqualified.

| Level | Type / symbol | Role speech |
|-------|---------------|-------------|
| Station cut | `BeamResultantVector` | Six-component cut resultants at station `z` |
| **Section** (contour) | **`SectionForceVector`** | **External force vector** — FE-agnostic force densities on section midlines. Shaped **outside** FE (`shape_section_force_vector` / `distribute_beam_resultants`). |
| **Element** | `F_e` / `Fe` on `ShellForceObject` | **Element force vector** — local consistent load on one 4-node shell element |
| Station FE system | `f` / `f_global` | **Global force vector** — assembled nodal RHS for the **station shell mesh** (`K u = f`). Not blade-global; not the section-contour DTO |

Empty mesh units are 4-node shell quads in `SectionStripMesh` / `StripPanelMesh`
(geometry + connectivity only). Constitutive attach is `bind_shell_material_zones`
(`material_utilities.section_bind`); FE applies external → `Fe`/`f` and forms `Ke`.

## Junction-bounded panels

Outer skin is two SDF solids — `outer_skin_SS` (suction / camber-upper) and
`outer_skin_PS` (pressure / camber-lower). Midline export emits one open
half-ring per side. [`build_shell_strip_topology` /
`split_skin_at_junctions`](../../../geometry_utilities/section_geometry/topology/shell_strip_topology.py)
then splits each side at **web feet** into nested arcs:

```text
outer_skin:{SS|PS}:{LE|bay_{i}|TE}
```

Cap endpoints snap onto those arcs for mesh junctions but do **not** invent
extra named skin roles. There is no separate “LE joint” / “TE joint” kind:
LE/TE are skin-arc names at the tip ends of each half-ring.

| Part | Panelisation |
|------|----------------|
| **Skin** | Nested SS/PS × LE/bay_i/TE open strips (`kind="skin"`) |
| **Each shear web** | Own open strip → own panel between its two skin feet |
| **Upper / lower spar cap** | Own subcomponent, own panel each; cap–web T-junctions insert shared vertices but do **not** split the cap into multiple panels |

Illustrative section (2 webs + upper/lower caps) after topology:

```text
Skin panels:  SS:{LE, bay_0, TE}  and  PS:{LE, bay_0, TE}
Web panels:   web0, web1
Cap panels:   upper_cap, lower_cap   (T-junction nodes only)
```

Zone emission order in `bind_shell_material_zones` / `shell_material_zones_from_mesh_inputs`
is **skin → caps → webs** — do not assume `panels[i]` matches raw `midlines[i]`.

## Thickness and layup defaults (symmetric section)

**Capability:** every panel may have an independent thickness and laminate
(panel-owned constitutive state). That enables detailed zoning and intentional
bend–twist coupling.

**Default section policy:**

| Rule | Default | Intentional override |
|------|---------|----------------------|
| Within one subcomponent | All of its panels share **equal thickness** and the **same layup** | Per-panel thickness / stack for local zoning |
| Upper vs lower skin | Symmetric (mirror) thickness + layup | Asymmetric stacks/angles for bend–twist |
| Upper vs lower spar caps | Symmetric (mirror) thickness + layup | Same |
| Shear webs | Each web is its own subcomponent → one panel; thickness equal within that web; webs may differ only if the design sets them so | Per-web thickness/layup |

Today skin segments already inherit one `skin.thickness_m` after split. Kind-level
`skin_lam` / `cap_lam` in `bind_shell_material_zones` likewise forces shared layups. The
**free panel** model is the long-term contract; equal-within-subcomponent /
upper–lower symmetry is the **default**, not a hard architectural limit.

## Frames and constitutive kind

Constitutive behaviour is **constant per panel** (same ABD, same thickness,
same material kind for every element on that panel). Elements never own an
independent material assignment.

Frames follow [`coordinate_frames/README.md`](../../../geometry_utilities/coordinate_frames/README.md):
**C** (curvilinear MITC4 DOFs) coincides with **E** (element/laminate) on each
facet; **M** is the ply material frame after `T_ply`.

`ShellFormulationCache` records per-panel:

- `panel_constitutive_kind[pi]` ∈ `{"isotropic", "orthotropic"}`
- `panel_sigma_allow_pa[pi]` — finite for isotropic panels; unused (NaN) for orthotropic

| Kind | Material | Tier 3 | Tier 4 (quaternary FI) |
|------|----------|--------|------------------------|
| **Orthotropic** | `LaminateDefinition` | **E** mid-surface + **M** face stresses (`T_ply`) | **Hashin** on **M** mid-face (+ interlaminar from **E** shear) |
| **Isotropic** | `IsotropicMaterial` | **M** ≡ **E** (`n_ply=1`) | **Von Mises** on mid-face + `sigma_allow`; Hashin skipped |

Upstream section-definition flags (`SubcomponentGeometry.is_isotropic` /
`is_composite`) are the geometry-side source of truth; the formulation cache
carries the panel-level copy so FI routing is panel-owned, not name-heuristic.

See also [RESULTS_DESIGN.md](../processing/static/results/RESULTS_DESIGN.md) (quaternary
consumes `cache.panel_constitutive_kind`).
