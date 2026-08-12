# element_library — MITC4 / shell formulations

Element stiffness kernels for the FE worker.

↑ Parent: [`../../README.md`](../../README.md)

## Purpose / ownership

| Owns | Does **not** own |
|------|------------------|
| MITC4 Q4 FSDT shell element (`mitc4/`) | Strip mesh generation ([`section_mesh`](../../../../mesh_utilities/section_mesh/README.md)) |
| Formulation factory / validation | Global assemble / BC / solve ([`../../processing/static/`](../../processing/static/README.md)) |
| Element-local \(B\), \(D\), shape functions | Layup policy / ABD assembly (material libraries + bind) |

---

## Mathematics

Production kernel is **MITC4** (mixed interpolation of tensorial components) under
first-order shear deformation theory (FSDT). Per element:

\[
\mathbf{K}_e = \int_{\Omega_e} \mathbf{B}^\mathsf{T}\,\mathbf{D}\,\mathbf{B}\,\mathrm{d}A
\]

with laminate ABD (and shear) condensed into \(\mathbf{D}\) from the bound zone’s
laminate. Meshing and recovery remain **element-agnostic** at the cascade layer;
this home owns the operator that fills `Ke`.

Physical model summary: [`../../docs/SHELL_MODEL.md`](../../docs/SHELL_MODEL.md).

---

## Exact I/O

| Symbol / path | Role |
|---------------|------|
| [`mitc4/`](mitc4/) | Production MITC4 implementation |
| [`formulation_factory.py`](formulation_factory.py) | Theory → element constructor |
| [`shell_element_base.py`](shell_element_base.py) | Shared shell element Protocol / base |
| [`kernel.py`](kernel.py) | Element-agnostic helpers |

Consumers use the package barrel / runner — not deep `mitc4.*` imports from
outside this worker.

---

## Provenance / what happens today

| Claim | Today |
|-------|-------|
| Production theory | MITC4 FSDT |
| CCX | Quarantined validation only ([`../../calculix_ccx/`](../../calculix_ccx/README.md)) |
| Element-agnostic cascade | Result tiers do not hard-code MITC4 types |

## See also

- [`../../README.md`](../../README.md)
- [`../../docs/SHELL_MODEL.md`](../../docs/SHELL_MODEL.md)
- [`../../docs/ELEMENT_AGNOSTIC_ARCHITECTURE.md`](../../docs/ELEMENT_AGNOSTIC_ARCHITECTURE.md)
