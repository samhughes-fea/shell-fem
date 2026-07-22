# Shell model taxonomy

Canonical glossary for the FE geometry and constitutive hierarchy used by
`section_shell_model`. Layout family codes (`SystemType{X}{Y}-{Z}`) live in
[`section_geometry/docs/system_type_xyz_taxonomy.md`](../../section_geometry/docs/system_type_xyz_taxonomy.md)
and answer a different question; this document covers station → element.

Results tiers (primary → quaternary) are documented in
[`processing/results/RESULTS_DESIGN.md`](../processing/results/RESULTS_DESIGN.md).

## Hierarchy

```text
Station (z_station)                    — analysis slice with spanwise depth
  └── Section                          — closed 2D cross-section assembly
        └── Subcomponent               — structural role: skin | web | spar cap
              └── Strip                — FE-agnostic mid-surface polyline handoff
                    └── Panel          — constitutive region (constant t, material)
                          └── Element  — MITC4 (or other) shell quad; inherits panel material
```

| Term | Meaning | Primary types |
|------|---------|---------------|
| **Station** | Analysis slice at spanwise coordinate `z_station`. Owns a section plus depth along the beam axis (unit-strip / strip width). | `ShellSectionStationGeometry`, `ShellSectionStationAnalysis`, `SectionDefinition.station_z` |
| **Section** | Closed 2D assembly of subcomponents in the cross-section plane (no spanwise extent by itself). | `MultiCellSection`, `SectionStripMesh`, `SectionDefinition` |
| **Subcomponent** | Structural role: outer skin, shear web, or spar cap. May own **one or more** panels when thickness/layup varies along the contour. | `OuterSkin` / `ShearWeb` / `SparCap`; `ShellMidlineStrip.kind` ∈ `{skin,web,cap}` |
| **Strip** | FE-agnostic mid-surface polyline + thickness handoff (often 1:1 with a panel; skin may split into multiple strips). | `ShellMidlineStrip` |
| **Panel** | Definitive **constitutive region**: one laminate or isotropic material, one thickness, one FI criterion — shared by all its elements. Not an FE element. | `Panel`, `StripPanelMesh`, `panel_abd_stacks[pi]`, `panel_constitutive_kind` |
| **Element** | One MITC4 (or other) shell quad along the panel contour; inherits material from its panel; spanwise edge length = station strip depth. | `ShellStiffnessObject`, `ShellPanelResultants` (per-element despite the name) |

### Ownership (separation of concerns)

| Role | Object | Owns |
|------|--------|------|
| Structural part | **Subcomponent** | Kind (skin/web/cap), topology role; may contain **N panels** |
| Constant constitutive region | **Panel** | Thickness, material, isotropic/orthotropic kind, ABD, FI criterion |
| FE discretisation | **Element** | Geometry/DOF only; material via parent panel |

Do **not** introduce a parallel “layup zone” type while `Panel` already carries constitutive ownership. If a rename is ever needed, prefer `MaterialZone` / `ConstitutiveRegion` over `LayupZone` (the latter excludes isotropic metals). Aerospace English often reserves “panel” for a buckling bay (`PanelBuckling*`); that sense is separate from the constitutive region.

## Section vs station

- **Section** = the 2D cut: closed assembly of subcomponents in the cross-section plane.
- **Station** = that section placed at spanwise `z`, with a finite depth along the beam axis (unit strip / strip width used by the shell elements’ spanwise edges).

A station therefore has a depth dimension; a section alone does not.

## Junction-bounded panels

[`build_shell_strip_topology` / `_split_skin_at_junctions`](../../section_geometry/interface/shell_strip_topology.py)
splits the closed outer-skin ring at arc-length projections of **web feet and
cap endpoints**. Each resulting open strip (`skin:seg0`, …) becomes one panel —
skin panels run **between** those junctions (LE/TE arcs and bay skins).

There is no separate named “LE joint” / “TE joint” type: LE/TE appear as
skin-segment boundaries where the split set (and ring wrap) puts them.

| Part | Panelisation today |
|------|--------------------|
| **Skin** | Many panels, junction-bounded (split at web/cap attachments) |
| **Each shear web** | Own open strip → own panel between its two skin feet |
| **Upper / lower spar cap** | Own subcomponent, own panel each; cap–web T-junctions insert shared vertices ([`_insert_cap_web_interior_junctions`](../../section_geometry/interface/shell_strip_topology.py)) but do **not** split the cap into multiple panels |

Illustrative section (2 webs + upper/lower caps) after topology:

```text
Skin panels:  ... | LE nose | upper bay | TE | lower bay | ...   (split at web/cap feet)
Web panels:   web0, web1                                     (one each)
Cap panels:   upper_cap, lower_cap                           (one each; T-junction nodes only)
```

Panel emission order in `build_section_v2` is **skin → caps → webs** — do not
assume `panels[i]` matches raw `midlines[i]`.

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
`skin_lam` / `cap_lam` in `build_section_v2` likewise forces shared layups. The
**free panel** model is the long-term contract; equal-within-subcomponent /
upper–lower symmetry is the **default**, not a hard architectural limit.

## Material frame and constitutive kind

Constitutive behaviour is **constant per panel** (same ABD, same thickness,
same material kind for every element on that panel). Elements never own an
independent material assignment.

`ShellFormulationCache` records per-panel:

- `panel_constitutive_kind[pi]` ∈ `{"isotropic", "orthotropic"}`
- `panel_sigma_allow_pa[pi]` — finite for isotropic panels; unused (NaN) for orthotropic

| Kind | Material | Tier 3 (material frame) | Tier 4 (quaternary FI) |
|------|----------|-------------------------|------------------------|
| **Orthotropic** | `LaminateDefinition` | Ply stresses rotated into ply material axes (`T_ply`) | **Hashin** (in-plane + interlaminar) |
| **Isotropic** | `IsotropicMaterial` | No fibre axes — laminate frame ≡ material frame | **Von Mises** from laminate-frame membrane stress + `sigma_allow`; Hashin skipped |

Upstream section-definition flags (`SubcomponentGeometry.is_isotropic` /
`is_composite`) are the geometry-side source of truth; the formulation cache
carries the panel-level copy so FI routing is panel-owned, not name-heuristic.

See also [RESULTS_DESIGN.md](../processing/results/RESULTS_DESIGN.md) (quaternary
consumes `cache.panel_constitutive_kind`).
