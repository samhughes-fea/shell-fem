# Scope

`shell-fem` is layered shell and solid finite elements for composite
structures, blades above all: the high-fidelity reference model of the suite.

## In scope

- Shell (Tri3, Quad4, Quad8, curved Quad8 / Tri6) and layered solid (Hex8 /
  Hex20) elements, links, springs and supports.
- Building a full 3D blade model from the geometry kernel's lofted geometry and
  shell mesh, with ply books.
- Loads, linear and geometrically nonlinear statics, buckling, mass.
- Stress recovery through the thickness, failure and progressive damage.
- Beam section properties extracted from the shell model, as a reference for the
  section model.
- Benchmarks and verification cases.

## Out of scope

| Topic | Where it belongs |
| --- | --- |
| Blade and section geometry, the shell mesh | [implicit-geometry-kernel](https://github.com/samhughes-fea/implicit-geometry-kernel) |
| Ply and laminate constitutive behaviour | [laminate-constitutive-model](https://github.com/samhughes-fea/laminate-constitutive-model) |
| Cross-section stiffness models | [anisotropic-section-model](https://github.com/samhughes-fea/anisotropic-section-model) |
| Beam solvers | [1d-fem](https://github.com/samhughes-fea/1d-fem) |
