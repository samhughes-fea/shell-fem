# constitutive — panel ABD prep

Zone constitutive extraction shared by assembly and recovery.

↑ Parent: [`../README.md`](../README.md) · Package: [`../../README.md`](../../README.md)

## Purpose / ownership

| Owns | Does **not** own |
|------|------------------|
| `panel_abd_tuple_from_lam` / ABD stacks for panels | Laminate *policy* / ply stacking ([`section_bind`](../../../../material_utilities/section_bind/README.md)) |
| Compact ABD layout for recovery | Material libraries / Hashin strengths |
| Thickness / \(G_{\mathrm{eff}}\) scalars for FSDT shear | Remeshing or SFV |

---

## Mathematics

Classical laminate theory ABD (6×6) from the bound zone laminate:

\[
\begin{Bmatrix}\mathbf{N}\\\mathbf{M}\end{Bmatrix}
=
\begin{bmatrix}\mathbf{A}&\mathbf{B}\\\mathbf{B}&\mathbf{D}\end{bmatrix}
\begin{Bmatrix}\boldsymbol{\varepsilon}_{0}\\\boldsymbol{\kappa}\end{Bmatrix}.
\]

Effective transverse shear modulus used by the MITC4 shear block (isotropic
fallback when only \(E,\nu\) are exposed):

\[
G_{\mathrm{eff}}
=
\frac{E}{2(1+\nu)}.
\]

ABD enters \(\mathbf{D}\) in

\[
\mathbf{K}_e=\int_{\Omega_e}\mathbf{B}^{\mathsf{T}}\mathbf{D}\mathbf{B}\,\mathrm{d}A
\]

([`../element_library/`](../element_library/README.md)). Frame: laminate **E**
(coincides with curvilinear **C** on the facet).

---

## Exact I/O

| Path | Role |
|------|------|
| [`panel_abd.py`](panel_abd.py) | `panel_abd_tuple_from_lam`, `collect_panel_abd_stack`, `compact_panel_abd_from_stacks` |

### `panel_abd_tuple_from_lam(lam) -> (ABD 6×6, t, ν, E, G_eff)`

| In | Bound laminate (`stiffness_bundle()` / `build_ABD()` / legacy convertible) |
| Out | ABD, thickness [m], Poisson, \(E\), \(G_{\mathrm{eff}}\) |

### `collect_panel_abd_stack(panels) -> (abd_stack, thickness_m, G_eff, labels)`

| In | Sequence of panel / zone objects with laminates |
| Out | `(n_valid,6,6)` ABD stack + thickness / \(G_{\mathrm{eff}}\) vectors + labels |

---

## PNG pointers

| Figure | Meaning |
|--------|---------|
| Tertiary CLT face recovery (consumes ABD) | [`../../processing/static/results/compute_tertiary/examples/outputs/x3_stations_composite.png`](../../processing/static/results/compute_tertiary/examples/outputs/x3_stations_composite.png) |
| Panel face semantics | [`../../processing/static/results/compute_tertiary/examples/outputs/panel_face_semantics.png`](../../processing/static/results/compute_tertiary/examples/outputs/panel_face_semantics.png) |

---

## Provenance / what happens today

| Claim | Today |
|-------|-------|
| ABD source of truth | Bound `MaterialZone` laminate via `stiffness_bundle()` |
| FE does not invent layups | Thickness / ply policy owned by bind + orchestration |

## See also

- [`../element_library/README.md`](../element_library/README.md)
- [`../../docs/SHELL_MODEL.md`](../../docs/SHELL_MODEL.md)
- [`../../../../material_utilities/section_bind/README.md`](../../../../material_utilities/section_bind/README.md)
