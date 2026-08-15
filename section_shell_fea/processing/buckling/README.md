# buckling — geometric stiffness / eigen

Linear buckling track: prestress from static secondary → \(K_g\) → eigen.

↑ Parent: [`../README.md`](../README.md) · Package: [`../../README.md`](../../README.md)

## Purpose / ownership

| Owns | Does **not** own |
|------|------------------|
| Assemble geometric stiffness \(\mathbf{K}_g\) | Static quaternary FI |
| Section / panel eigenproducts | SFV shaping / remesh |
| Primary → secondary buckling assessments + station fold | Campaign sizing loops |

**Prestress bridge:** static **secondary** resultants only — never quaternary FI.

---

## Mathematics

Default **section** track (connected assembly, same reduction \(\mathbf{T}\) as static):

\[
\mathbf{K}_r\,\boldsymbol{\phi}
=
\lambda\,(-\mathbf{K}_{g,r})\,\boldsymbol{\phi},
\qquad
BI
=
\frac{1}{\lambda_{\min}}.
\]

Optional **panel** track: classical per-strip eigenproduct (`eigen_track="panel"`).

Geometric stiffness is assembled from mid-surface prestress \((N,M,Q)\) recovered
on the static secondary tier.

Deep taxonomy: [`results/RESULTS_DESIGN.md`](results/RESULTS_DESIGN.md).

---

## Exact I/O

| Path | Role |
|------|------|
| [`operations/assemble_geometric.py`](operations/assemble_geometric.py) | Section \(K_g\) assembly |
| [`operations/solve_eigen.py`](operations/solve_eigen.py) | Eigenproduct |
| [`operations/eigen_states.py`](operations/eigen_states.py) | Eigen state bags |
| [`results/compute_primary/`](results/compute_primary/) | Modal kinematics |
| [`results/compute_secondary/`](results/compute_secondary/) | Panel assessments / \(BI\) |
| [`results/fold_station_bundle.py`](results/fold_station_bundle.py) | `ShellStationBucklingBundle` |

Public entry: [`../../simulation_runner/`](../../simulation_runner/README.md)
`SectionLinearBucklingRunner` / `api/linear_buckling.py`.

| In | Static solve state + secondary prestress |
| Out | `ShellStationBucklingBundle` (P→S + fold) |

---

## PNG pointers

| Figure | Meaning |
|--------|---------|
| Static S prestress context | [`../static/results/compute_secondary/examples/outputs/resultants_Nx_by_panel.png`](../static/results/compute_secondary/examples/outputs/resultants_Nx_by_panel.png) |
| Mesh the eigenproduct acts on | [`../../visualisation/output/section_centerline_mesh_v2/section_mesh_2C-F_i000.png`](../../visualisation/output/section_centerline_mesh_v2/section_mesh_2C-F_i000.png) |

(Committed buckling-mode fringe PNGs are not yet under `visualisation/output/`;
use runner / consumer viz when needed.)

---

## Provenance / what happens today

| Claim | Today |
|-------|-------|
| Default eigen track | Full-section `section` |
| Not part of static Q | Buckling does not write Hashin FI |
| Runner | Sibling of linear-static — not nested inside Q |

## See also

- [`results/RESULTS_DESIGN.md`](results/RESULTS_DESIGN.md)
- [`../static/README.md`](../static/README.md)
- [`../../docs/SECTION_SOLVE_STAGES.md`](../../docs/SECTION_SOLVE_STAGES.md)
- [`../../docs/SHELL_MODEL.md`](../../docs/SHELL_MODEL.md)
- [`../../simulation_runner/README.md`](../../simulation_runner/README.md)
