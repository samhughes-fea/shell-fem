# Static results cascade (index)

Index for **`processing/static/results/`**: how primary → secondary → tertiary → quaternary hand off. Methodology bodies live in each tier’s L4 `docs/README.md` — this page is glue only.

## Table of contents

1. [Tier cascade](#tier-cascade)
2. [Frame ladder](#frame-ladder)
3. [SI units](#si-units)
4. [What each tier does not recompute](#what-each-tier-does-not-recompute)
5. [Mid-face criteria vs extreme-fibre](#mid-face-criteria-vs-extreme-fibre)
6. [Demo policy](#demo-policy)
7. [Tier deep docs](#tier-deep-docs)

---

## Tier cascade

```mermaid
flowchart LR
  primary[compute_primary]
  secondary[compute_secondary]
  tertiary[compute_tertiary]
  quaternary[compute_quaternary]
  primary -->|"ShellPrimaryResultSet U_e"| secondary
  secondary -->|"ShellSecondaryResultSet resultant_rows"| tertiary
  tertiary -->|"ShellTertiaryResultSet sig_ply_mat"| quaternary
  quaternary --> out[ShellQuaternaryResultSet_FI]
```

| Tier | Directory | Product | Owns |
|------|-----------|---------|------|
| 1 | `compute_primary/` | `ShellPrimaryResultSet` | Solve disassembly (`U_e`); [`compute_primary/docs/README.md`](../compute_primary/docs/README.md) |
| 2 | [`compute_secondary/`](../compute_secondary/docs/README.md) | `ShellSecondaryResultSet` | Shell resultants `[N; M]` (+ optional Q1/Q2) |
| 3 | [`compute_tertiary/`](../compute_tertiary/docs/README.md) | `ShellTertiaryResultSet` | CLT → per-plate face \(\boldsymbol{\sigma}_{123}\) (+ interface \(\tau\)) |
| 4 | [`compute_quaternary/`](../compute_quaternary/docs/README.md) | `ShellQuaternaryResultSet` | Hashin / Von Mises / extreme-fibre FI + governing |

---

## Frame ladder

Canonical names: [`coordinate_frames/README.md`](../../../../../../geometry_utilities/coordinate_frames/README.md).
Do not mix blade **B** / **S** / **R** XYZ into E/M stress objects. Avoid bare “local/global”.

| Stage | Frame | Stress / force objects |
|-------|-------|------------------------|
| Primary (mesh / DOFs) | Station mesh in **B**; nodal DOFs in curvilinear **C** (≡ **E** on each facet); assembled FE system | `d_global`, `U_e` (FE-system “global” ≠ blade XYZ / **R**) |
| Secondary | **E** (element = laminate) \((x_1,x_2,x_3)\) | `N11…M12` / `Q1,Q2` = \(N_{11}\ldots\) / \(Q_1,Q_2\) |
| Tertiary (mid-surface / interface) | **E** | \(\boldsymbol{\varepsilon}_0\), \(\boldsymbol{\kappa}\); `tau_iface` \([\tau_{13},\tau_{23}]\) |
| Tertiary (stored plies) | **M** (ply material) after `T_ply` | `sig_ply_mat`, `eps_ply_mat` `(n_elem, n_ply, 3_faces, 3)` |
| Quaternary | **M** (same as tertiary face product) | Hashin / Von Mises / extreme-fibre on chosen face(s) |

**CLPT mental model:** a laminate is a **stack of thin plates**. Isotropic recovery is that algorithm on **one** plate; composite recovery is the **same** algorithm on each plate in the stack (see [tertiary — stack of plates](../compute_tertiary/docs/README.md#clpt--stack-of-plates)).

**Panel face reading** (laminate `FACE_OUTER` / `FACE_INNER` → blade sides): skin fluid/void; spar-cap skin/void; web **fore**/**aft** (LE/TE sides — not fluid/void). **Runtime-enforced** at mesh-input construction (`assert_panel_x3_conventions`). See [tertiary panel face semantics](../compute_tertiary/docs/README.md#panel-face-semantics-blade-section) and [`panel_face_semantics.png`](../compute_tertiary/examples/outputs/panel_face_semantics.png).

Legacy field names `z_ply` / `ply_interfaces_z` mean laminate **\(x_3\)**, never global \(Z\).

---

## SI units

| Quantity | Unit |
|----------|------|
| Stress, yield / ultimate strengths | Pa |
| Membrane / shear resultants \(N\), \(Q\) | N/m |
| Moment resultants \(M\) | N·m/m |
| Thickness, \(x_3\), \(L_s\), \(L_x\) | m |
| Failure index | dimensionless (1 = onset) |

---

## What each tier does not recompute

| Tier | Does **not** |
|------|----------------|
| Secondary | Solve the FE system; recover ply stresses; evaluate FI |
| Tertiary | Recover shell resultants from \(U_e\); evaluate Hashin / VM / extreme-fibre |
| Quaternary | Run CLT; invent face stresses not present in tertiary |

---

## Mid-face criteria vs extreme-fibre

Tertiary is the **recovery product**: every trusted element has per-plate `(outer, mid, inner)` stresses. That closes the former “mid-only recovery” gap.

Quaternary then **selects** stations by criterion (by design, not a tertiary defect):

| Criterion | Stress station | Strength vocabulary |
|-----------|----------------|---------------------|
| Hashin (composite) | Mid face `sig[..., 1, :]` | Ply `Xt`/`Xc`/`Yt`/`Yc`/`S12` |
| Von Mises (isotropic) | Mid face plate 0 | Yield \(\sigma_y\) ← `yield_strength` / `panel_sigma_y_pa` |
| Extreme-fibre (every panel) | Laminate free surfaces \(\pm h/2\) | Ultimates \(\sigma_{\mathrm{uts}},\sigma_{\mathrm{ucs}},\tau_{\mathrm{us}}\) |

Do **not** feed Von Mises with ultimates or extreme-fibre with \(\sigma_y\). Forbidden on these Q strength APIs: `*_allow*` names.

---

## Demo policy

Tier `examples/` plot scripts are **synthetic / helper-level demos**. They illustrate nesting, CLPT per-plate faces, mid-face Hashin/VM, and extreme-fibre ultimates. They are **not** artefacts of `Shell*ResultsOrchestrator.compute()`.

- PNGs under each `examples/outputs/` are **committed** documentation figures (linked from each tier’s deep README).
- Tertiary adds `free_surfaces_vs_mids_composite.png` and `panel_face_semantics.png` (skin fluid/void, spar skin/void, web fore/aft); quaternary adds `extreme_fibre_fi_envelope.png` and `vm_vs_extreme_fibre_bending.png`.
- Regenerate all three demos:

```bash
python blade_structure_utilities/fem_utilities/section_shell_fea/processing/static/results/examples/run_all_result_demos.py
```

---

## Tier deep docs

| Tier | Deep README | Key equations → functions |
|------|-------------|---------------------------|
| Secondary | [`../compute_secondary/docs/README.md`](../compute_secondary/docs/README.md) | [N/M / Q recovery](../compute_secondary/docs/README.md#key-equations--functions) |
| Tertiary | [`../compute_tertiary/docs/README.md`](../compute_tertiary/docs/README.md) | [ABD / faces / \(+\hat n\)](../compute_tertiary/docs/README.md#key-equations--functions) |
| Quaternary | [`../compute_quaternary/docs/README.md`](../compute_quaternary/docs/README.md) | [VM / Hashin / extreme-fibre / IL](../compute_quaternary/docs/README.md#criteria-formulas) |
| Tertiary stations (closed) | [`../compute_tertiary/docs/roadmap_x3_stations.md`](../compute_tertiary/docs/roadmap_x3_stations.md) | — |
