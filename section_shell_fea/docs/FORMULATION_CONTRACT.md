# Shell formulation, projection, and failure contracts

Normative WP0 contracts for `section_shell_fea`. Discrete formulation
`MITC4-FSDT` is the production Reissner–Mindlin (FSDT) element. Frame
mathematics lives in `geometry_utilities.coordinate_frames`; this package
consumes it and must not reimplement rotation matrices.

## Theory vs formulation vs backend

| Term | Meaning | Example |
|------|---------|---------|
| Continuum theory | Weak-form kinematics | `FSDT` (Reissner–Mindlin) |
| Discrete formulation | Registered element | `MITC4-FSDT` |
| Backend | External solver, not a native element | CalculiX `S4` |
| Analysis DOF layout | One explicit nodal DOF vector per analysis | `[u_x, u_s, w, beta_s, beta_x, beta_z]` |
| Native operator layout | Element `B_e`, `D_e`, generalized strain | MITC4 eight-component `N/M/Q` |
| Canonical result layout | Dense shared result vectors | resultants `[N11,N22,N12,M11,M22,M12,Q1,Q2]`; stress `[σ11,σ22,σ33,τ23,τ13,τ12]` |

Registries are keyed by discrete formulation. Theory-to-default mapping:
`FSDT → MITC4-FSDT`. Automatic DOF union is forbidden. Extra DOFs require an
explicit extended analysis layout.

## Frame chain

`S → B` before FE. B midlines define strip frame **C**. On each flat MITC4
facet **C ≡ E**. Ply recovery applies `T_σ` / `T_ε` **E → M**. Parent
`(ξ,η)` and assembled “global” indices are not structural frames.

## MITC4 topology and DOFs

- Q4, nodes `(i0, i1, i2, i3)` contour then extruded.
- Nodal DOFs in **C**: `[u_x, u_s, w, β_s, β_x, β_z]`.
- Generalized resultants in **E**: `[N11, N22, N12, M11, M22, M12, Q1, Q2]`.
- Work-conjugate strains: `[ε11, ε22, γ12, κ11, κ22, κ12, γ13, γ23]`.
- Native contributions embed into the analysis layout with exact zero blocks
  for unused DOFs.

## Capability vocabulary

Numeric arrays stay dense. Zeros are interpreted only with capability tags:

| Tag | Meaning |
|-----|---------|
| `modelled` | Present in the weak form |
| `constitutive_recovered` | Recovered from constitutive law |
| `postprocessed_reconstruction` | Diagnostic reconstruction (not governing) |
| `theory_constrained_zero` | Theory requires the component to be zero |
| `not_modelled_zero` | Absent from the weak form; stored as 0 |

MITC4 ply-face stress: `σ11, σ22, τ12` are `constitutive_recovered`;
`σ33 = 0` is `not_modelled_zero`. Reconstructed `τ13, τ23` at interfaces are
`postprocessed_reconstruction`.

## Failure models

Authoritative composite criterion for MITC4: `hashin_1980_plane_stress`.
`hashin_1980_3d` is implemented for future formulations and is **unavailable**
for MITC4 because `σ33` is a not-modelled weak-form zero.

No delamination / cohesive criterion is implemented. FSDT interface shear is
diagnostic-only and excluded from governing envelopes and sizing gates.

Cross-criterion comparison uses **utilization** `1 / RF` from the positive
reserve factor to onset. Native FI remains for equation traceability.

Default required criteria: plane-stress Hashin (composite MITC4), Von Mises
(isotropic). Extreme-fibre is opt-in. Requesting 3D Hashin on MITC4 fails closed.

Statuses: `available`, `screening_only`, `unavailable`, `invalid_input`.

## Migration names

| Legacy | Authoritative |
|--------|----------------|
| `hashin` / `max_fi_hashin` | plane-stress Hashin native FI (deprecated alias) |
| `hashin_il` / interlaminar Hashin | FSDT interface-shear diagnostic |
| governing `fi` | `max_utilization_governing` |
| delamination claim | not implemented |

Deprecated aliases remain readable through the next major release and are
excluded from new sizing logic immediately.

## Verification / handover

Focused suites (external mesh/GCI campaigns excluded):

```text
python -m pytest src/blade_structure_utilities/fem_utilities/section_shell_fea/tests -q
python -m pytest src/blade_structure_utilities/material_utilities/failure/tests -q
python -m pytest src/blade_structure_precompute/module/section_selection/tests/unit/test_fi_hard_constraint.py -q
```

Known exclusions: grid/cascade/defect-k/cap-peak studies that need campaign
meshes. Those are reported separately and are not waived as green.
