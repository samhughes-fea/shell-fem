# Deferred research backlog (section_shell_fea)

Items tracked after the FI production loop. **Out-of-scope** items below will **not** be
implemented in `section_shell_fea`; remaining backlog items may still be promoted later.
Do not wire backlog items into product constraints until explicitly promoted.

**Production FE entry (Phase 0):** contour-supported linear-static cascade
(`run_section_linear_static_from_resultants` / `SectionLinearStaticRunner`).
Channel-A periodic product surfaces (`api.periodic` / `api.k6` /
`operations/periodic/`) and the `section_optimisation` package are **removed**.
See [LINEAR_STATIC_MAIN_TRACK_PHASE0.md](LINEAR_STATIC_MAIN_TRACK_PHASE0.md).

## Out of scope — will not feature

The following were considered during historical periodic unit-cell design but are
**excluded from the `section_shell_fea` scope**. Do not schedule or design
against them in this package.

### 7th warping / bimoment mode (historical periodic-cell extension)

**Background:** The former periodic 6-mode basis dropped restrained warping
(`B`, `dB_dx`) that a clamped global-coupled section solve could retain. A 7th
unit-cell mode (warping gradient `dpsi/dx`, bimoment recovery) was explored as a
possible extension.

**Policy:** **Will not feature** in `section_shell_fea`. The production section
shell solve is a 6-DOF beam-resultant linear-static cascade; bimoment-driven
checks belong outside this package (e.g. beam/global models) if needed.

> **Status:** Clamped section FE (`api/legacy_clamped.py`) and Channel-A periodic
> (`run_section_periodic_bc`) product paths are **deleted**. Production entry is
> `run_section_linear_static_from_resultants`. (`pipeline_b` / Pipeline B remains
> as geometry only.)

### Geometric nonlinear Newton loop

**Background:** A former `engine/shell_geometric_nonlinear.py` stub exposed
`nonlinear_shell_solve_not_implemented`. That module was **removed** with the
`engine/` package retirement. Production assembly and
`SectionLinearStaticRunner` solves are **linear static** (`K u = f`).
`ShellAnalysisRegime.NONLINEAR` has no registered theories.

**Policy:** **Out of scope — will not feature** in `section_shell_fea`. This is
**not** a scheduled milestone. Linear load application uses contour-supported
assembly + exclusive beam vs face-traction ownership
([LOAD_APPLICATION_DOUBLE_COUNTING.md](LOAD_APPLICATION_DOUBLE_COUNTING.md));
do not read
[`physics_contract.py`](../processing/static/operations/physics_contract.py) as a
Newton roadmap (`ResidualTangentPhysics` is a reserved contract only).

### HSDT native package revival

**Policy:** **Out of scope — will not feature.** The former
`pre_processing/element_library/hsdt/` package, `ShellTheory.HSDT`, and
`register_hsdt_backend()` were deleted. `ProvenanceKind.HSDT` remains only as
vocabulary for old artefacts. Production theories are FSDT-only; CCX is the sole
opt-in validation backend.

## Load application — double-counting / BC split (**closed**)

Problem 1 (load provenance) and Problem 2 (single free set for face traction on
the linear-static track) are implemented. Production FI remains beam-resultant by
default. See
[LOAD_APPLICATION_DOUBLE_COUNTING.md](LOAD_APPLICATION_DOUBLE_COUNTING.md).

## CCX periodic acceptance (policy closed 2026-07-14) — validation only

> **Non-product.** Channel-A periodic product APIs are removed. The notes below
> document the CCX validation oracle (`calculix_ccx/periodic.py`) only.

CCX-gated tests in ``test_ccx_periodic_acceptance.py`` run when ``ccx`` is on PATH.

**Spike 4A attribution:** The former aspirational 0.5× all-channel band failed because
(1) multi-key assertions on near-zero parasitic resultants (Fx under M22, Mz under N)
inflate relative error, and (2) torsion ``max|N12|`` ~0.20× MITC4 is S4R local
redistribution — already accepted by the engineering ``rel=0.85`` gate.
``emit_frd=False`` vs ``True`` does not change membrane integrals (``.dat`` path).

**Spike 4B policy:** Acceptance tracks validated engineering bands (~10% load-aligned
integrals; torsion N12 ``rel=0.85``). Xfails removed. Native reaction K6 redesign
(~300× stiff bending) remains a separate longer-horizon item — hybrid MITC4 K6 stays
for the CCX oracle. See [PERIODIC_CCX_HOMOGENIZATION.md](PERIODIC_CCX_HOMOGENIZATION.md).

**Reaction-K6 diagnose spike (2026-07-14):** Confirmed ref-node ``RF`` are MPC Lagrange
forces, not beam conjugates (axial ~1×; κ_y ~313× / κ_z ~2071× on NACA ``NE=4``; L-strip
also over-stiff). **No-promote** of ``assemble_k6_from_steps``; hybrid MITC4 ``K6``
stays. Repro: ``python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.run_ccx_reaction_k6_diag``.
A future promote path still needs MPC redesign — not a parse/sign fix.

**Status:** Closed (policy renegotiation); does not block production linear-static FI.
Reaction-K6 root-cause diagnose closed (no-promote); redesign still deferred.

## Legacy `multi_cell_blade_section` / `sys.path` shims

**Problem:** Examples and MVP tests historically imported `lib.laminate_clpt` via
`examples/section_stress_model` on `sys.path`.

**Resolved (2026-07 follow-up):** Production, topology defaults, and
CCX unit fixtures use `blade_structure_utilities.material_utilities.laminate`
(`ClptPlyRow` / `LaminateDefinition`). Package-local defaults live in
`material_utilities/section_bind/default_laminates.py`. Production plot/NACA/warping
helpers no longer call `ensure_stress_model_lib_path`; the vendor thin-wall
`run_section` oracle was removed from `tests/test_section_shell_fea.py`.
The HSDT element library was **deleted** (2026-07-14); do not revive vendor CLPT for it.

**Status:** Production / CI vendor dependency **closed**. Optional remaining
demo-only bootstrap: `examples/area-diagrams-FSDT-shell/*` (not a CI production
path). The clamped FE module (`api/legacy_clamped.py`) and optimiser package are gone.

## MVP thin-wall mapping retirement

**Symbol:** `panel_station_shell_resultants` (formerly in `api/legacy_clamped.py`)

**Status:** **removed** (see `docs/architecture/DEPRECATION_REGISTER.md`). The MVP mapping with
M=0 placeholders is gone, and the entire clamped section FE module has been deleted.
Production Hashin FI uses the linear-static cascade
(`run_section_linear_static_from_resultants` / `run_station_hashin`). Channel-A
periodic / Saint-Venant `K6` product surfaces are **removed** (Phase 0) — see
[`DEPRECATED_ALIASES.md`](DEPRECATED_ALIASES.md) and
[LINEAR_STATIC_MAIN_TRACK_PHASE0.md](LINEAR_STATIC_MAIN_TRACK_PHASE0.md).

## Cusp traction-balance penalty (perf finding #8 / Defect K diagnostic)

**Background:** At non-collinear 2-way junction clusters, soft traction-balance
penalties were optionally applied to the reduced stiffness ``K_rr`` via
``enforce_traction_balance_at_cusp=True`` on the former full-solve spine
``solve_global_coupled_shell``. Each cusp required ~20 unit-load
``mitc4_edge_shear_traction_integrated`` calls (4 DOFs × 4 Gauss × 2 panels);
penalty scale was ``traction_penalty_alpha * max(K_rr diagonal)`` (default α=`1e-2`).
The functional test ``test_traction_penalty_reduces_cusp_residual`` showed the
penalty could reduce cluster ``Tx`` residual on a synthetic 45° junction.

**Status (2026-07-14):** **Retired with the full-solve spine.** Production
linear-static FI never enabled the flag. Face traction uses
``build_clamp_supports`` / `traction_face_load` on the cascade; CCX uses
``section_export`` BCs — neither called ``solve_global_coupled_shell``. Non-collinear
junction traction residuals remain an **accepted limitation** (Defect K in
[`SHELL_MODEL.md`](SHELL_MODEL.md)) after Spike 1B parked historical
``transformed_basis`` periodic K6 — Measure via
``processing/static/diagnostics/defect_k_mesh_sweep.py``.
Analytical ∂Tx/∂d from edge B-matrices stays research-only; do not reintroduce soft
penalties on the production path.

## Periodic `transformed_basis` K6 (parked permanently) — historical

**Background:** After Approach A fixed the singularity, periodic `transformed_basis` K6
remained wrong on the N-gon discriminator (axial 0.8775× EA; torsion diverges).

**Spike 1B (2026-07-14):** Clamp / Approach-B condense candidates restore axial when
cluster `W` is pinned, but torsion stays ~N² divergent and fuller clamps break
bending/shear. No promote path.

**Policy:** Will **not** feature production periodic `transformed_basis` K6.
Channel-A periodic product surfaces are removed (Phase 0). Production linear-static
FI stays on `shared` junctions under contour support. Direct `transformed_basis`
assembly remains valid for non-periodic sharp-junction diagnostics / fatigue
singular-retry. See
[PERIODIC_CLUSTER_BASIS_SINGULARITY.md](PERIODIC_CLUSTER_BASIS_SINGULARITY.md).

**Track 2B consequence:** No geometry-gated production flip to `transformed_basis`.
Defect K remains an accepted limitation on `shared` junctions; primary reaction
metrics stay authoritative; cusp #8 analytical ∂Tx/∂d stays research-only
(not scheduled).

## Perf finding #9 — panel-local BC LIL (closed-obsolete)

**Status:** **Closed-obsolete.** The legacy `panel_mitc4_model._apply_bcs` LIL row-zeroing
path was deleted with clamped Path B FE. Production uses `reduce_propagate_solve` /
`SectionLinearStaticRunner`. No further work; see
[PERFORMANCE_OPTIMISATION_REPORT.md](PERFORMANCE_OPTIMISATION_REPORT.md).

## Beam→section resultant/stiffness matching (open)

**Problem:** Beam station resultants \(R(z)=(N,V_y,V_z,M_y,M_z,T)\) historically
reached section shell solves through ad-hoc, lossy handoffs. Five component
orderings coexisted; the moment reference axis was dropped in transit; bending
moments were never corrected for an offset axial force; frame metadata was absent;
strip midsurface `K6` and shell homogenised `K6` were independent with no parity
check; recovery-bundle `y_sc`/`z_sc` historically mislabelled the elastic centroid.

**Approach:** `blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector.BeamResultantVector`
— a single immutable vector carrying canonical `BEAM_IO_ORDER` storage, a frame
tag, and `about_yz`. Handoffs use `transported_to` (full \(M'=M+r\times F\)),
`rotated`, and `to_mode_conjugate` (named projection). Production FI prefers
`BeamResultantVector` into `run_section_linear_static_from_resultants`.
See [SHELL_MODEL.md](SHELL_MODEL.md) § Multi-scale boundary conditions.

**Still deferred (research-grade, not scheduled):**

- ~~Saint-Venant twist-about-shear-centre torsion mode~~ **Promoted as opt-in
  (2026-07):** ``run_section_linear_static_from_resultants(...,
  torsion_about_shear_centre=True)`` sets ``about_yz`` to the sectorial shear
  centre via ``shear_centre_about_yz_from_pipeline_b``. Default product path
  unchanged. **Not** a Channel-A periodic ``K6`` revive.
- Bending–shear gradient coupling \(\partial M/\partial x=V\) inside a historical
  prismatic unit cell (bimoment already out of scope above).
- Face traction as default production FI — orchestration stays beam-resultant by
  default; standalone `run_section_linear_static_from_load_field` remains available.
- Axial \(N\) / gravity / centrifugal envelopes on the beam-resultant path by default.
- F2.1 seeding of strip stiffness from shell homogenised `K6` — **formally parked**.
  The report-only diagnostic
  [`processing/static/diagnostics/k6_parity.py`](../processing/static/diagnostics/k6_parity.py)
  compares named conjugates (`N, M22, Mz, T`) only. Do not gate product constraints
  on that report until explicitly promoted.
  (`OptimBladeGeometry.beam_section_stiffness_source` already names
  `section_shell_fea` for constitutive exports — that is a separate export
  path, not F2.1 strip→shell K6 replacement.)

**Status:** `BeamResultantVector` contract + linear-static adoption landed;
strip↔shell `K6` parity is report-only; F2.1 shell-seed of strip `K6` stays parked.
The former optimiser product path that consumed these notes is removed (Phase 0).
