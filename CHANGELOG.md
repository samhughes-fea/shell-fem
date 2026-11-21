# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- **Warped flat shells move rigidly.** Flat Quad4 and Quad8 elements work on their
  nodes projected onto a mean plane. On a warped element, a rigid rotation strained
  that projection (a residual of about 1e-2 of K on one warped Quad4), which made
  twisted and doubly curved meshes of flat elements far too stiff. Each projected
  point is now tied to its node by a rigid offset along the normal (MacNeal's
  warping correction; `warping_correction=True`, the default). Flat elements on
  flat meshes are unchanged bit for bit; the curved elements are not affected.

### Changed

- **One linear buckling path.** `linear_buckling(mesh, bcs, load_case, n_modes=1)` now
  runs `LinearBucklingRunner`, so both calls use the same stages
  (`geometric.AssembleGeometricStiffness`, `eigen.SolveBucklingEigenproblem`,
  `ComputeBucklingModes`). Its signature and `BucklingResult` are unchanged: positive
  factors, modes scaled to a largest absolute component of 1 (now always +1), and
  per-element mode energy. The runner reports `geometric_stiffness` and `eigen_solve` on
  the `blade.stage` logger. The eigen-solve switches from dense to Lanczos at
  `processing.eigenpairs.DENSE_LIMIT` (800 free DOFs, the vibration solve's limit), not
  1500. The old stage classes `processing.buckling.operations.AssembleGeometricStiffness`
  and `SolveBucklingEigenproblem` (also importable from `processing.buckling`) still work
  with a DeprecationWarning and go in 0.3.0.
- **Kernel v0.9.3.** CI, the release check and the README install implicit-geometry-kernel
  `v0.9.3` instead of the non-release tag `K9`, and `pyproject.toml` requires `>=0.9.3`.
  `tests/test_kernel_pcomp.py` imports `ShellMesh` from `implicit_geometry_kernel.mesh`,
  since 0.9.3 drops the flat `mesh.shell_mesh` module. Kernel 0.9.3 meshes rib and
  tip-closure cells as triangles, so `BladeBuilder.from_shell_mesh` takes a 3-node cell
  as one `tri3` (it used to fan it about a new centre node into three); polygons of five
  or more nodes from earlier kernels are still fanned.

- **Package layout.** The flat `laminate_*` modules moved into `pre_processing/`
  (containers, element_library, property_library, load_library, mesh_library,
  model_check), `processing/` (static operations and results, progressive damage),
  `simulation_runner/` (static, progressive damage, and the solver backends that were
  `kernels/`), `post_processing/` (VTK, beam section properties) and `verification/`.
  Imports point one way only, checked by `tests/test_layout.py`. The public API
  (`from shell_fem import ...`) is unchanged. The old paths (`shell_fem.laminate_*`,
  `shell_fem.kernels`, `shell_fem.run_solver_benchmarks`) still import, with a
  DeprecationWarning, until 0.3.0. The two compatibility files `laminate_fea_engine.py`
  and `laminate_kernels.py` are now old paths too. `laminate_blade_kernel_mesh` follows
  them: the builder is `pre_processing/mesh_library/blade_kernel_mesh.py` and
  `solve_kernel_pcomp` is `simulation_runner/static/kernel_pcomp.py`.
- The `shell-fem-benchmarks` command runs `shell_fem.verification.solver_benchmarks`.

### Added

- Geometric nonlinearity is scoped, not implemented: `geometric_nonlinearity_scope()`
  lists what exists (linear static `K u = F`, linear buckling `(K + λ Kg) φ = 0`,
  shell membrane `Kg`, solid Cauchy-stress `Kg`, dead pressure on the undeformed
  normal) and what is missing (Newton, follower pressure, `Kg` in the static
  tangent, updated geometry, arc-length post-buckling). `GeometricNonlinearRunner`,
  `NewtonEquilibrium` and `AssembleFollowerPressure` raise
  `GeometricNonlinearityNotImplemented`. Linear statics and linear buckling are
  unchanged.
- Labelled `StripHeaveAddedMass` on shells (Newman 1977 / Theodorsen 1935):
  `m_a' = rho_fluid * pi * (chord/2)**2`, spread over the chord as areal density
  `rho_fluid * pi * chord / 4` on the local-normal translation. Default vibration
  is unchanged; pass `added_mass=` to `vibration_modes` to attach it. Solids skip
  the strip model.
- `ply_stack_stress_strain` returns ply stress and strain at the bottom, mid and
  top of every ply (the through-thickness counterpart of `outer_ply_stress_strain`).
  Pass `EquilibriumStressRecoverer` to carry interlaminar shear and σz; the
  constitutive default is `ExactStressRecoverer`.
- `PlyPointField.normal_stress` is σz at each ply sample (zero unless the
  recoverer is equilibrium). VTK ply cells write that as s33.
- Condensed RZT (`RefinedZigZagModel(condensed=True)`) is tested beside FSDT on
  the same 6-DOF sandwich mesh, with both recoverers through the full stack.
- `ResultsExtractor.formulation()` returns the element strain operators (`B_membrane`,
  `B_bending`, `B_shear`), section matrices (`C_ee`, `C_ss`), shape functions,
  Jacobian and quadrature weights at the same recovery points as `gauss_points()`.
  `to_csv` writes a `formulation.csv` beside the result-level files.
- `PlyPointField` now includes in-plane total, mechanical and material strains at
  each ply sample (bottom, mid and top of every ply).
- Layered Hex8 and Hex20 solids assemble a consistent mass matrix and a Cauchy-stress
  geometric stiffness, so a local solid patch runs in `vibration_modes` and
  `linear_buckling` / `LinearBucklingRunner`, not only in a static solve. Rotational
  DOFs of the solids stay inert; the solver still holds solid-only rotations at zero.
- `GlobalTotals.mass` is the model mass: laminate areal density times element area,
  summed over shells and solids.
- `VibrationModes` reports generalised mass (`φ^T M φ`), mass participation
  (`φ^T M r` for the six rigid-body directions about the origin) and per-element
  modal strain energy (`½ φ_e^T K_e φ_e`).
- `BucklingResult` reports per-element strain energy of each buckling mode.
- Result CSVs from `ResultsExtractor.to_csv` start with `# schema=shell-fem/results@1`.
- `shell_fem.verification.shell_benchmarks`: the shell obstacle course
  (Scordelis-Lo roof, pinched cylinder, pinched hemisphere, twisted beam in both
  directions) for any element, mesh and element options, with
  `run_obstacle_course`. Example 27 runs it on all five elements and
  `tests/test_shell_benchmarks.py` checks it.
- `deflection_profile` reads a static solution as section-mean translation, rotation
  and twist along one axis. The span is z.
- `outer_ply_stress_strain` returns CLPT in-plane stress and strain at the outer
  face of the bottom ply and the top ply (`eps = eps0 + z kappa`). `z` is measured
  from the element reference surface, so a laminate offset stays the eccentricity.
- `linear_buckling(mesh, bcs, load_case, n_modes=1)` returns the positive load factors
  of one reference load and the corresponding modes.
- `LinearBucklingRunner(mesh, bcs, load_case, n_modes).execute().get_results()` runs
  the same reference load, then `AssembleGeometricStiffness`, `SolveBucklingEigenproblem`
  (dense or Lanczos; positive load factors, or both signs)
  and `ComputeBucklingModes`. `LinearBucklingResults` holds the modes (`BucklingMode`:
  load factor and mode shape scaled to a unit largest translation, `as_solution()` for
  VTK), the reference static solution and the stage timings.
- Shell elements gain `membrane_prestress` and `geometric_stiffness_matrix(..., resultants=)`
  for a uniform membrane prestress. `geometric_stiffness_matrix(u_elem, state)`, the call
  `linear_buckling` makes, is unchanged. Hygrothermal prestress counts. Layered solids
  contribute Cauchy-stress geometric stiffness on the same path.
- `shell_fem.verification.buckling_benchmarks`: simply supported plates in compression
  (isotropic or specially orthotropic, any aspect ratio), shear and restrained heating,
  a cantilever column and a cylinder in axial compression, each against its classical
  closed form. Example 28 runs them on all five elements and `tests/test_buckling.py`
  checks them. `isotropic_laminate` takes an optional thermal expansion coefficient.
- `vibration_modes` returns the shell model's natural frequencies and
  mass-normalised mode shapes. Those are the blade modes. A hub, shaft or rotor
  is not assembled.
- The static solve as one class per stage: `AssembleGlobalSystem`,
  `ConstrainGlobalSystem`, `ReduceConstrainedSystem`, `SolveReducedSystem`,
  `AssembleLoadVector`, `ReconstructGlobalField`, `ComputePrimaryResults`, passing
  the system states `GlobalSystem`, `ConstrainedSystem`, `ReducedSystem`,
  `FactorisedSystem`.
- `LinearStaticRunner(mesh, bcs, load_cases, combinations).execute().get_results()`
  returns `LinearStaticResults` (solutions by name plus `StaticDiagnostics`, the wall
  time of each stage). `StructuralGlobalSolver` keeps its API and runs the same stages,
  with the same results to the last bit.
- `KernelMeshBladeBuilder.build_pcomp` and `solve_kernel_pcomp` take the kernel shell
  mesh and one PCOMP per kernel element. Plies are bottom to top. `offset` is the
  node surface above the laminate mid-plane as a fraction of the thickness
  (`eccentricity / thickness` for a cap whose nodes stay on the skin mid-surface).
  The constitutive mid-plane is `-offset * thickness`, and the Nastran deck writes
  that as PCOMP `Z0 = -(0.5 + offset) * thickness`. `solve_kernel_pcomp` runs the
  native static solver on that mesh.
- `BladeBuilder.from_shell_mesh(shell_mesh, geometry=None)` builds the shell model on the
  geometry kernel's own shell mesh (`implicit_geometry_kernel.mesh.shell_mesh`, kernel
  0.9.2), with no remeshing: kernel quads become `quad4` elements, rib and tip-closure
  polygons `tri3` fans about a centroid node. Each element's laminate is looked up per
  layer by wall label, region or kind (`"rib"` for ribs; the tip closure is skin), and a
  capped skin is the cap + skin stack with the kernel's eccentricity as its offset.
  Normals are the kernel's (skin out of the section, webs to the trailing edge, ribs
  and tip along +z). `BladeMesh.shell_mesh` keeps the kernel mesh. EA, EI and GJ match
  `from_geometry` within 0.8 % on a prismatic 2D-F blade.
- Requires implicit-geometry-kernel 0.9.2. CI installs git tag K9
  (`d3f6c4fb74bff47131fd3dcbabb1228c3eea6b0b`), which still reports package version 0.9.2.

## [0.2.0] - 2026-11-11

- Requires implicit-geometry-kernel 0.9.1 or later (CI installs the v0.9.1 tag and the
  system GEOS library, `libgeos-dev`, which the kernel now loads). Layout codes carry
  their anchoring and rib tokens (`"2D-F-LC-0R"`) and `BladeGeometry` takes
  `pivot_fraction`; see the kernel's `docs/MIGRATING.md`.
- `BladeBuilder.from_geometry`: a spar cap bonded to a skin stays on the skin mid-surface,
  where the kernel now keeps it, and the stacked cap + skin laminate has an offset of half
  the cap thickness inward. Skin and capped-skin elements share one reference surface;
  the stack's eccentricity is in the laminate. Section stiffness is as before (EI within
  0.3 % on the example 20 section).
- CI installs laminate-constitutive-model at its v0.2.0 tag.
- Examples 24 to 26 read the IEA 15 MW section data from the private reference-blades
  catalogue (`examples/_shared/reference_blades.py`) and skip without it. The
  committed `iea15mw_sections.json` and its producer script are removed; figures and
  results stay.

## [0.1.0] - 2026-11-05

### Changed

- **Split.** This repository is shell-fem, the finite-element half of
  composite-laminate-model, with its history. The package is `shell_fem`. The ply
  and laminate behaviour (materials, `Ply`, `Laminate`, ABD, the CLPT / FSDT / RZT /
  layerwise section stiffness, hygrothermal states, temperature-dependent materials,
  failure criteria, `PaganoPlate`) moved to
  [laminate-constitutive-model](https://github.com/samhughes-fea/laminate-constitutive-model),
  which this package depends on and re-exports, so `from shell_fem import Laminate`
  works. Their tests and examples 05 and 13 moved with them.
- The benchmark command is `shell-fem-benchmarks` (was `laminate-benchmarks`).

### Changed

- The package is now `composite_laminate_model`, split out of composite-laminate-fea
  with its history. The implicit geometry is its own package,
  [implicit-geometry-kernel](https://github.com/samhughes-fea/implicit-geometry-kernel)
  (formerly `blade_geometry_model`), and thin-walled beam-section theory is
  [anisotropic-section-model](https://github.com/samhughes-fea/anisotropic-section-model),
  together with former examples 27 to 32. `Airfoil`, `Station`, `SkinRegion` and
  `ShearWeb` are still importable from `composite_laminate_model`.

### Removed

- The `laminate_modelling_package` compatibility shim.

### Added

- `implicit_geometry_kernel` now holds a fork, with history, of blade-structure's
  implicit section geometry: `section_geometry` (signed-distance solids with CSG,
  40 structural layouts, wall midlines and cell-wall loops) with `section_contract`,
  `coordinate_frames`, `area_2d` and `shoelace_integration`, and their 785 tests.
  `src/implicit_geometry_kernel/FORK.md` records the fork point and the files that stayed
  in blade-structure.
- `BladeGeometry` lofts one of those layouts between `Station`s (chord, twist, pitch
  axis, offset and airfoil interpolated linearly) and `section_at(z)` returns a
  `SectionCut`: wall midlines with thickness and region labels, cell loops and any
  dropped webs, in the `BladeBuilder` blade frame. Wall thicknesses may vary along the
  span, webs sit at chord fractions or at fixed distances from the leading edge, and
  walls thinner than the geometry's manufacturing floors are built exactly.
- `ThinWalledSection.from_geometry(cut, laminates)` builds a section from a cut, with
  laminates bound by label, region or kind and spar caps merged into the skin they are
  bonded to. On the example 20 section it matches `from_blade_mesh` to within 1% in EA,
  EI and GJ.
- `BladeBuilder.from_geometry(geometry, n_span, element, element_size, wall_elements)`
  meshes a `BladeGeometry` as shells on its walls' mid-surfaces, with spar caps merged
  into their skins and element sets named after the geometry's labels. It meshes any
  of the 40 layouts, and on the example 20 section it matches the region-based
  `BladeBuilder` model to within 1% in EA, EI and GJ.
- `SectionCut.mid_surface()`: the wall graph that `from_geometry` and the mesher share.
- Example 32: a blade from its implicit geometry.

### Fixed

- `BladeGeometry.section_at` now builds the section at the station's twist, so
  flapwise (`-F`) webs stay normal to the rotor plane on a twisted blade instead of
  turning with the chord like chord-normal (`-CN`) webs.
- Caps reaching less than their own thickness past a web onto the next skin, and bare
  skin slivers that short at a cap's end, are no longer separate walls. Whether they
  showed up depended on the sampling, so the section's walls changed along the span.

### Changed

- Python 3.11 or newer is required (the forked geometry needs it; 3.9 and 3.10 are
  past end of life). Shapely and Numba are now dependencies.
- pytest imports test modules by path (`--import-mode=importlib`) and also collects
  the geometry's tests under `src/implicit_geometry_kernel`.
- The code is split into three packages, each depending only on the ones before it:
  `implicit_geometry_kernel` (`Airfoil`, `Station`, `SkinRegion`, `ShearWeb`),
  `composite_laminate_model` (laminate mechanics, shell and solid FE, solver,
  kernels, `BladeBuilder`, `beam_section_properties`) and
  `anisotropic_section_model` (`ThinWalledSection`, `RestrainedTorsionBeam`,
  `GeneralizedBeam`, `TimoshenkoBeam`). The section modules drop their `laminate_`
  prefix (`laminate_gbt` is now `anisotropic_section_model.gbt`, and so on).
- `laminate_modelling_package` still imports and re-exports the same public names,
  and its old submodules are aliases of the new ones. It will be removed after the
  first release.
- `tests/test_package_split.py` checks every import against those boundaries.

## composite-laminate-fea 0.1.0 (never released)

First release of `composite-laminate-fea` (import name `laminate_modelling_package`).

### Added

- 3D orthotropic materials, plies and laminates, temperature-dependent properties
  and a material library.
- CLPT, FSDT, refined zig-zag and layerwise kinematics behind one generalised
  section formulation.
- Tri3, Quad4, Quad8, curved Quad8 (default) and Tri6 shells; layered Hex8 and
  Hex20 solids with shell-to-solid coupling; RBE2 and RBE3 links.
- Blade section builder (airfoils, skin regions, spar caps, shear webs, twist,
  taper), mould-line or mid-surface skins, ply books with ply drops, and beam
  section properties extracted from the shell model.
- Pressure fields, line, body and hygrothermal loads; load cases solved with one
  factorisation and load combinations.
- Through-thickness stress recovery, including interlaminar stresses from 3D
  equilibrium; Pagano's exact 3D solution as a benchmark.
- Maximum stress, Tsai-Hill, Tsai-Wu and Hashin failure criteria, a delamination
  onset check, and ply-by-ply stiffness knockdown for progressive damage.
- Native NumPy/SciPy, CalculiX and Nastran solver kernels, a benchmark runner
  (`laminate-benchmarks`), Gmsh meshing, model checks and ParaView export.
- Twenty-six worked examples, each checked against a reference.

### Packaging

- MIT licence, `pyproject.toml` packaging (setuptools), version taken from
  `laminate_modelling_package.__version__`.
- The benchmark runner moved into the package
  (`laminate_modelling_package.run_solver_benchmarks`).
- A release workflow that builds and checks the sdist and wheel; publishing to
  PyPI is manual and off by default (see `RELEASING.md`).
