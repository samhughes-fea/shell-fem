# section_shell_model

**Pure FE** cross-section analysis. Two classic simulation runners:

| Runner | Role |
|--------|------|
| `SectionLinearStaticRunner` | Linear static: stages + always P→S→T→Q on `solve()` |
| `SectionLinearBucklingRunner` | Linear buckling from static prestress |

Meshes each structural subcomponent's mid-surface into strip shell elements and recovers
shell resultants / Hashin FI via a contour-supported linear-static cascade.

**Architecture:** meshing, junction topology, and resultant recovery are **element-agnostic**.
MITC4 is the production FSDT kernel; CalculiX thin-wall is optional validation only.

**Taxonomy:** station → section → subcomponent → strip → panel → element — see [`docs/TAXONOMY.md`](docs/TAXONOMY.md).

**Main track:** `run_section_linear_static_from_resultants` /
`run_station_hashin` — beam resultants → nodal \(f\) → full cascade.
See [`docs/SECTION_SOLVE_STAGES.md`](docs/SECTION_SOLVE_STAGES.md).

This package does **not** own optimisation or fatigue products.

---

## Pipeline position

```
section_geometry  (ShellMidlineStrip polylines via contract/)
section_loads     (SectionStationLoadField tractions)
        │
        ▼
section_shell_model   ←── this package (pure FE)
        │  linear-static / buckling runners
        │  ShellStationBucklingBundle, recovery bundles
        ▼
orchestration / analysis consumers
```

Used as a library — no `__main__.py`. `main_precompute.py` drives the shell stage via
`SectionShellModelStage` with MITC4 v2 strip mesh default (`SECTION_SHELL_USE_MITC4_V2`).

See [docs/SHELL_MODEL.md](docs/SHELL_MODEL.md).

---

## Methodology

### Physical model: FSDT shell with CLT laminate

#### Kinematic theory — FSDT (Reissner–Mindlin)

Each panel is modelled as a **First-order Shear Deformation Theory (FSDT)** shell. Unlike the Kirchhoff–Love (classical thin-plate) assumption, FSDT admits transverse shear strains through the thickness — the shell director is not constrained to remain normal to the mid-surface. This is essential for correctly modelling thick composite laminates and for capturing shear at strip junctions.

Nodal kinematics: 3 translations + 2 independent director rotations = **5 DOF/node**.

#### Element technology — MITC4 (anti-locking)

Standard bilinear FSDT elements suffer from **transverse shear locking** — spurious shear stiffness that makes thin elements artificially stiff. The **Mixed Interpolation of Tensorial Components** (MITC4) element eliminates this by interpolating the covariant transverse shear strains (`e_rt`, `e_st`) from tying points at the element edge midpoints, decoupling shear from membrane/bending without reduced integration.

- 4-node quadrilateral, 4 Gauss points (2×2) for membrane and bending
- Assumed transverse shear field from 4 tying points (one per edge midpoint)
- Element thickness direction set from the subcomponent mid-surface normal at the strip polyline

#### Constitutive model — CLT laminate (A/B/D)

Each element's material is assembled from the ply stack using **Classical Laminate Theory (CLT)**:

```
[N]   [A  B] [ε⁰]       A = Σₖ Qₖ tₖ            (membrane)
[M] = [B  D] [κ ]       B = Σₖ Qₖ tₖ z̄ₖ          (coupling)
                         D = Σₖ Qₖ (tₖ z̄ₖ² + tₖ³/12)  (bending)
```

where `Qₖ` is the rotated reduced stiffness matrix for ply `k`, and `z̄ₖ` is the ply mid-plane offset from the laminate mid-surface. The coupling matrix `B` is non-zero for asymmetric laminates (relevant for spar caps with unequal upper/lower cap ply counts).

Transverse shear is accounted for at the FSDT kinematic level; no additional shear correction factors are needed beyond the MITC4 anti-locking treatment.

#### Connection to SDF implicit geometry

The MITC4 strip meshes are built directly from `ShellMidlineStrip` polylines — constructive mid-surface loci from `midline_polyline()` in `section_geometry`, rotated to the B-frame, with geometric junction topology applied by `build_shell_strip_topology` before handoff. This package receives polylines only — no SDF evaluation occurs here. FE junction clustering (`build_section_v2`) remains in this package.

### Strip mesh construction

Each subcomponent (outer skin, spar cap, shear web) is mapped to a strip of MITC4 elements:

1. The midline polyline `(N, 2)` from `ShellMidlineStrip` is extruded to a mesh of quadrilateral panels
2. Element width is set to the subcomponent arc-length divided by `n_elements_per_panel`
3. Element normals point in the through-thickness direction (outward for skin, chord-normal or flapwise for webs)
4. Material laminate from `LaminateDefinition` is mapped to each element's constitutive matrices

### Closed-cell topology (`build_section_v2`)

After individual strips are meshed, junctions between subcomponents must be connected. Three constraint modes are supported:

| Mode | When to use | Method |
|------|-------------|--------|
| `shared` | Production **linear-static** default | Direct DOF identification at junction clusters |
| `transformed_basis` | Non-collinear junctions (assembly diagnostics / fatigue singular-retry) | DOFs at junction projected into a shared frame aligned with the bisector of the two strips |
| `shared_rotated` | Near-collinear junctions with local orientation difference | Shared DOF with a rotation mapping between element frames |

Production FI uses `shared` under contour-supported linear-static BCs. Historical notes on periodic × `transformed_basis` singularity live in [docs/PERIODIC_CLUSTER_BASIS_SINGULARITY.md](docs/PERIODIC_CLUSTER_BASIS_SINGULARITY.md) (non-product).

### Shell resultant recovery

The **linear-static cascade** (`SectionLinearStaticRunner`, always P→S→T→Q) recovers full N/M per panel from the MITC4 solve under contour support.
Legacy thin-wall mapping placeholders (Ny, Mx, My, Mxy = 0) applied only to a removed MVP path.

Per-element resultants at Gauss points feed quaternary Hashin (`fi_recovery`) and optional CLPT helpers (`solve_station_clpt_shell`).

### Constitutive export

`BeamSectionConstitutiveInput` aggregates section stiffness / centres for recovery handoff where available from the station analysis builder. Channel-A Saint-Venant periodic `K6` homogenisation is **removed** (Phase 0); do not reintroduce it as a product surface.

### Shell buckling

`ShellStationBucklingBundle` carries linearised (eigenvalue) panel / section buckling data per station via `SectionLinearBucklingRunner`:

- Geometric stiffness matrix assembled from pre-buckling membrane state `(Nx, Nxy)`
- Critical load factor `λ_cr` from smallest eigenvalue of `K_e + λ·K_g = 0`
- Mode shapes for visualisation

### Provenance tracking

Every result field in `ShellPanelResultants` is tagged with a `FieldProvenance` record:

```python
FieldProvenance(kind=ProvenanceKind.MITC4, note="K6_midsurface_v1")
```

This enables downstream consumers to distinguish fully computed quantities from placeholders, and to audit which theory produced each field.

---

## Key data structures

### `ShellMeshInputs`
Input to the MITC4 mesher; produced by `build_shell_mesh_inputs()`.

| Field | Type | Description |
|-------|------|-------------|
| `strips` | list[StripInput] | One entry per subcomponent: polyline, thickness, laminate |
| `junction_pairs` | list[(str, str)] | Subcomponent label pairs that share a junction node |
| `constraint_mode` | str | `"transformed_basis"` / `"shared_rotated"` / `"shared"` |

### `StripPanelMesh`
Strip-mesh element data for one subcomponent.

| Field | Type | Description |
|-------|------|-------------|
| `nodes` | `(n_nodes, 3)` float | Node coordinates in section frame |
| `connectivity` | `(n_elems, 4)` int | Element node indices |
| `gauss_data` | list | Per-element Gauss point weights and shape function derivatives |
| `constitutive` | list | Per-element `A, B, D, H` matrices |
| `label` | str | Subcomponent identifier |

### `SectionStripMesh`
Full section: all strips + junction topology.

| Field | Description |
|-------|-------------|
| `panels` | list[`StripPanelMesh`] — one per subcomponent |
| `junction_constraints` | DOF constraint equations across strip boundaries |
| `global_node_map` | Local → global node index mapping |

### `ShellSectionStationAnalysis`
Complete per-station analysis bundle.

| Field | Description |
|-------|-------------|
| `geometry` | `ShellSectionStationGeometry` — midline polylines and dimensions |
| `constitutive` | `BeamSectionConstitutiveInput` — K6, mass for beam handoff |
| `shell_resultants` | `tuple[ShellPanelResultants, ...]` — per-panel N/M recovery |
| `recovery_point_set` | `ShellRecoverySectionPointSet` — Gauss-point ply recovery data |
| `buckling` | `ShellStationBucklingBundle \| None` — panel buckling assessment |
| `provenance` | `ShellStationProvenance` — theory/version audit trail |

### `ShellPanelResultants`
Shell membrane and bending resultants at **one element** (carries `panel_index` / element identity). Despite the type name, this is elemental resolution — not one object per constitutive panel. See [`docs/TAXONOMY.md`](docs/TAXONOMY.md).

| Field | Type | Description |
|-------|------|-------------|
| `Nx` | float (+ `provenance` dict) | Axial membrane force [N/m] |
| `Ny` | float (+ `provenance` dict) | Transverse membrane force [N/m] |
| `Nxy` | float (+ `provenance` dict) | Shear flow [N/m] |
| `Mx, My, Mxy` | float (+ `provenance` dict) | Bending moments [N·m/m] |

Full N/M are recovered on the linear-static cascade. Historical thin-wall MVP mapping
(`panel_station_shell_resultants`, now **removed**) used zero placeholders for
`Ny`/`Mx`/`My`/`Mxy`. Each field's `provenance` entry (`FieldProvenance`) records whether it
is trusted or a placeholder.

### `BeamSectionConstitutiveInput`
Shell-derived constitutive data for beam model handoff (Saint-Venant K6).

| Field | Type | Description |
|-------|------|-------------|
| `z_station` | float | Spanwise position [m] |
| `K6` | `(6, 6)` float | Shell-homogenised section stiffness |
| `mass_per_length` | float \| None | μ [kg/m] |
| `inertia_terms` | dict[str, float] | Section inertia terms |
| `centres` | dict[str, float] | Elastic / shear centre coordinates [m] |
| `warping_metadata` | dict[str, Any] | Vlasov / warping diagnostics |
| `provenance` | `ShellStationProvenance` | Theory/version audit trail |

### `ShellStationBucklingBundle`
Linearised buckling results per station.

| Field | Description |
|-------|-------------|
| `lambda_cr` | list[float] — critical load factors (eigenvalues) |
| `mode_shapes` | list[array] — eigenvectors (nodal displacements) |
| `panel_geometries` | Panel dimensions and aspect ratios |
| `prebuckling_resultants` | `(Nx, Nxy)` state used to build geometric stiffness |

### `ShellRecoveryMaterialPoint` / `ShellRecoveryIsotropicPoint`
Ply-level stress and failure data at one Gauss point.

| Field | Description |
|-------|-------------|
| `sigma_1, sigma_2, tau_12` | In-plane stresses in material frame [Pa] |
| `hashin_fi` | Hashin failure index (composite plies) |
| `von_mises_fi` | Von Mises failure index (isotropic) |
| `ply_index` | Through-thickness ply index |
| `z_local` | Distance from mid-plane [m] |

### `FieldProvenance` / `ProvenanceKind`

```python
class ProvenanceKind(str, Enum):
    DERIVED      = "derived"       # from section recovery (thin-wall mapping)
    PLACEHOLDER  = "placeholder"   # not yet recovered; explicit default
    RESERVED     = "reserved"      # reserved for future higher-order recovery
    MITC4        = "mitc4"         # recovered from MITC4 panel shell solve
    HSDT         = "hsdt"          # recovered from native HSDT (8-DOF) solve
    CALCULIX_CCX = "calculix_ccx"  # recovered from external CalculiX (ccx) S4R solve
```

`DERIVED`, `MITC4`, `HSDT`, and `CALCULIX_CCX` are *trusted* (see `provenance_is_trusted`); `PLACEHOLDER` fields must not govern Hashin FI.

---

## Package layout

```
section_shell_model/
├── pre_processing/        # mesh/api, loading/api, dispatch, element_library, containers/
├── processing/            # static/ (ops, P→S→T→Q, diagnostics), buckling/ (P→S)
├── simulation_runner/     # section_linear_static_runner, section_linear_buckling_runner,
│                          # api/ (linear_static, linear_buckling, station_recovery)
├── interface/             # cross-package adapters (station analysis, traction, recovery)
├── calculix_ccx/          # external CalculiX (ccx) S4R validation backend
├── io/                    # job_outputs, v2_station_json
├── validation/            # historical / CCX oracles (non-product)
├── _utils/                # cross-cutting helpers
├── docs/
│   ├── SHELL_MODEL.md
│   ├── SECTION_SOLVE_STAGES.md
│   ├── LINEAR_STATIC_MAIN_TRACK_PHASE0.md
│   ├── ELEMENT_AGNOSTIC_ARCHITECTURE.md
│   ├── TAXONOMY.md
│   ├── DEPRECATED_ALIASES.md
│   └── DEFERRED_RESEARCH.md
├── tests/
└── vis.py
```

---

## Quick start

```python
from blade_structure_utilities.section_shell_model import (
    ShellTheory,
    build_shell_mesh_inputs,
    build_section_v2,
    build_section_strip_mesh,
    build_structural_grid_shell_constitutive,
    run_section_linear_static_from_resultants,
)

# strips: list[ShellMidlineStrip] from section_geometry
mesh_inputs = build_shell_mesh_inputs(strips)
section_topology = build_section_v2(mesh_inputs)
strip_mesh = build_section_strip_mesh(section_topology, n_elements_per_panel=12)

# Wrap pre-tabulated K6 (does not homogenise from mesh):
# station_z is a sequence of length n, K6 has shape (n, 6, 6).
constitutive = build_structural_grid_shell_constitutive(
    station_z=[5.0],
    K6=k6_array,  # shape (1, 6, 6)
)

# Production FI path (only section shell FE entry)
cascade = run_section_linear_static_from_resultants(pb, N=1.0, shell_theory=ShellTheory.FSDT)
```

---

## References

- Bathe, K.J. & Dvorkin, E.N. (1986). A formulation of general shell elements — the use of mixed interpolation of tensorial components. *IJNME* 22, 697–722.
- Batoz, J.L. & Dhatt, G. (1990). *Modélisation des structures par éléments finis.* Hermès.
- Vlasov, V.Z. (1961). *Thin-Walled Elastic Beams.* Israel Program for Scientific Translations.
- [docs/SHELL_MODEL.md](docs/SHELL_MODEL.md) — handoff contract, multi-scale BCs, post-MVP refactor path
- [docs/PERIODIC_CCX_HOMOGENIZATION.md](docs/PERIODIC_CCX_HOMOGENIZATION.md) — periodic-BC CalculiX S4R homogenisation design/status (partial)
- [docs/PERIODIC_CLUSTER_BASIS_SINGULARITY.md](docs/PERIODIC_CLUSTER_BASIS_SINGULARITY.md) — periodic × cluster-basis MPC singularity (partly addressed; deeper formulation bug deferred)
- [contract/README.md](../contract/README.md) — `ShellMidlineStrip` geometry handoff contract
- [section_geometry/README.md](../section_geometry/README.md) — upstream producer of the strip polylines
- [materials/README.md](../materials/README.md) — CLT laminate (A/B/D) and failure-criteria primitives
- This package supersedes the retired `section_properties` midsurface-strip path (1D membrane theory); that package has been removed.
