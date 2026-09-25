# Bug log — section_shell_fea

Canonical package bug / frame-canon log. Docs index: [`README.md`](README.md).
Root-log defects: [BL-014](../../../../../../docs/bug_log.md#bl-014) /
[BL-015](../../../../../../docs/bug_log.md#bl-015). Constraints/assembly
architecture record: [2026-09-20](#constraints--assembly-architecture-2026-09-20).

## Coordinate-frames canon compliance (2026-08-11/12)

Audit of this package against
[`coordinate_frames/README.md`](../../../geometry_utilities/coordinate_frames/README.md).
Pinned by [`tests/test_section_frame_canon.py`](../tests/test_section_frame_canon.py).

### Fixed

| # | Site | Defect | Fix |
|---|------|--------|-----|
| 1 | `outline_frames` (now `load_utilities/section_loads/surface_traction/`) | `bframe_to_sframe_rotation` was `rotate_S_to_B` (forward map) applied where B→S was meant, **and** the chord was read off column 0. On canonical B column 0 is flapwise, so `chord_x` was the thickness extent and XFOIL got geometry normalised by ~`1/(t/c)`; the TE-first roll and `xfoil_te_arc_offset` then shifted Cp/Cf onto the wrong arc station. | Deleted the local 2×2. `blade_to_naca_rotation` builds the matrix from `coordinate_frames.rotate_B_to_S` applied to the basis rows, then the NACA component swap; chord now read off column 1. `bframe_outline_to_sframe_chord` → `blade_outline_to_unit_chord`. |
| 2 | `results/api/fi_recovery.py` (×3), `compute_tertiary/local_clpt_shell.py` | Laminate-frame (**E**) ply stress fed straight to `hashin_fi`. Canon: FI is only valid in **M**; E-frame under-predicts (≈7× low on a 30° stack in the regression fixture). Reached production reporting via tier-4 `compute_governing_element` → `io/job_outputs._periodic_hashin_audit_json`. | Rotate E→M with `lam.stiffness_bundle().T_ply` (memoised delegate of `coordinate_frames.ply_stress_transform_E_to_M`). `StationCLPTShellResult` now carries `sig_ply_mat_mid` / `eps_ply_mat_mid` (**M**) alongside the **E** record. |
| 3 | `thin_wall_plots._lam_to_mat_strain` | Rotated engineering strain with `orthotropic.rotation_matrix_theta`, which despite its docstring is \(T_\sigma\), not \(T_\varepsilon\) — the same defect `compute_ply_stresses` was already fixed for. | Helper deleted; reads the M-frame fields off the CLPT result. |
| 4 | 5 sites | Canon transforms reimplemented locally: `shell_mesh_map.normal_2d`, `equilibrium_checks._normal_2d`, `recovery_point_set_builder._stress_laminate_to_material_matrix`, `thin_wall_plots._lam_to_mat_stress`, plus #1. | All delegate to the `coordinate_frames` barrel. `thin_wall_plots.outward_normals` keeps its vectorised form but is pinned equivalent to `section_normal_2d`. |
| 5 | `mesh_fringe.py`, `example_plots.py`, `thin_wall_plots.py` | No section figure used the plot contract — zero uses of `section_plot_axes` / `setup_section_axes` / `invert_xaxis`. Every figure drew `ax.plot(pts[:,0], pts[:,1])`, which the canon warns renders the section **rotated 90°**, with `"y [m]"` / `"z/c"` labels and no LE-on-RHS invert. | New `visualisation/section_axes.py` wraps the canon helpers; all section figures route coordinates through it and finish with `finish_section_axes`. |
| 6 | `thin_wall_plots.plot_section`, `example_plots` cell labels | Mixed frames on one axes: the airfoil outline and `spar_positions` were chord-frame (`x ∈ [0,1]`, LE origin) while panel nodes were section-frame. | Cell labels now map chord fractions onto the outline's **own** Y span, so annotation and geometry share a frame with no extra parameters. |
| 7 | `station_load_builder._synthetic_cp_cf` | Chord bias read off column 0 and normalised by `max(...)`, assuming a `[0, max]` chordwise column. | Reads the chordwise column and normalises across the LE→TE span. |

### Open

| # | Site | Status |
|---|------|--------|
| 8 | `shell_midline_export.build_shell_midline_strips` + `ShellMeshInputs` | **Emitter fixed (2026-08-13).** Residual was item 11; that fallback now uses \(+X_B\). |
| 9 | `compute_tertiary/examples/plot_x3_recovery_demo.py` | **Fixed (2026-09-21).** Face-semantics figure uses the plot contract; remaining heuristics read canonical B (`Y_B` chord, `X_B` upper/lower). |
| 10 | Section-plane axis naming | **Accepted naming debt.** Identifiers stay `y`/`z` (`about_yz`, …); arithmetic is self-consistent. The wrong canon statement in `equilibrium_checks.py` now reads \((X_B, Y_B)\). A full identifier rename is a separate campaign — not scheduled. |
| 11 | `mesh_utilities/section_mesh/handoff/panel_x3_orientation.py` | **Fixed (2026-09-21).** No-skin cap fallback faces \(\pm X_B\), not \(+Y_B\) toward the TE. |

**Next steps.** Items 8–11 and the DV-before-attribution search gap are closed
or accepted. Remaining in this package: optional `y`/`z` → \(X_B\)/\(Y_B\)
identifier rename (item 10) — not scheduled. Zone-by-zone design in
`section_layup` remains the longer-term replacement for the heuristic FI search,
not a defect in the empty-map guard.

Gauge R4 campaign blockers are not FE-frame defects. Root
[BL-022](../../../../../../docs/bug_log.md#bl-022) is fixed locally on unique
preflight `20260921_152622`. Official G7 is blocked on
[BL-023](../../../../../../docs/bug_log.md#bl-023) /
[BL-024](../../../../../../docs/bug_log.md#bl-024). Do not treat a
`non_mono_at_finalize` smoke as an FE Hashin regression.

### Count-primary handoff — open decisions owned by the root log

The next cross-package representation change is tracked under
[BL-019 open implementation decisions](../../../../../../docs/bug_log.md#open-implementation-decisions-for-the-count-primary-representation).
That register records all unresolved choices: component demand versus zone-level
built gauge, geometry expressiveness, catalogue-to-designed-grid iteration, grid
identity, span-region continuity, search dimensionality, cache keys, DTO
ownership, floors/ceilings, run-out and failure policy, compatibility schema,
cutover, tolerances, acceptance baselines, integer search and retirement of the
old quantisers. They are not duplicated here because `section_shell_fea` does not
own the sizing representation.

Those choices are resolved collectively by the
[structural-gauge graph](../../../../../../docs/bug_log.md#collective-target-decision-2026-09-21)
and implemented in `layup.gauge_graph`. Production campaign authority is
gauge after the 16-station WP6 cutover. This package's four FE obligations are wired:

1. Gauge-authority binds pass `require_thickness_match=True` so strip/laminate
   equality applies to designed stacks.
2. `shell_prep_cache_key` includes stack-family digest and repeat count.
3. Equal thickness with a different family is a constitutive cache miss.
4. Typed upstream infeasibility (`RequiredInfeasibleError`) must not be repaired
   inside FE; the FE worker still does not clip or substitute laminates.

These are acceptance obligations for BL-019, not a reopening of frame items
8–11 or a new shell-formulation defect.

### Item 8 in detail — two frames inside one `ShellMeshInputs`

The canon (`coordinate_frames/README.md`) fixes **B** as \(X_B\) flapwise, \(Y_B\)
edgewise, \(Z_B\) spanwise, with \(+Y_B\) toward the trailing edge and the origin on
the pivot chordline — so stored data satisfies \(Y_{LE} < 0 < Y_{TE}\). The LE only
*appears* on the right because the plot inverts the horizontal axis; the data is
never negated.

`ShellMeshInputs` currently satisfies that for its endpoints and violates it for
its strips. Measured on NACA 2412, chord 1.0, `2D-CN`, zero twist:

| field | value | chord lies on |
|---|---|---|
| `leading_edge_twisted_pitch_axis_blade_frame` | `(0.0000, -0.3330)` | col 1 — canonical B, \(-f_{pivot}c\) |
| `trailing_edge_twisted_pitch_axis_blade_frame` | `(-0.0000, +0.6667)` | col 1 — canonical B, \((1-f_{pivot})c\) |
| `midline_twisted_pitch_axis_blade_frame` | col0 extent ~thickness, col1 extent ~chord | **col 1** — canonical B via `chord_to_blade` |

The endpoints and strips both go through `chord_to_blade` (swap + pivot shift +
rotation). Historical note: an earlier emit used bare `rotate_chord_to_blade`
(LE origin); that path is retired.

**Consequence.** `panel_x3_orientation` orients webs by \(\hat n \cdot (\mathrm{LE} - m)\),
so it takes a dot product between a strip-frame normal and a canonical-B LE. It
is not flipping anything today — signs agree in every case checked — but it
erodes the margin on a hard assert that fixes `FACE_OUTER` / `FACE_INNER` for CLT
recovery:

| case | alignment as shipped | alignment in one frame |
|---|---|---|
| `2D-CN` t=0.00, `web_0` | **+0.3999** | +0.9786 |
| `2D-CN` t=0.20, `web_0` | +0.6599 | +0.9786 |
| `1B-CN` t=0.12, `web_0` | +0.8596 | +0.9973 |

Up to ~2.5× less margin against `AMBIGUOUS_ALIGNMENT_TOL`. A section whose mixed
score lands near zero would orient its webs arbitrarily, and no within-frame test
in these packages can see it.

**Fix.** Emit strips with `chord_to_blade` so the whole DTO is canonical B. The
LE/TE emit is already correct and needs no change.

#### Follow-up measurement (2026-08-13) — the emitter is now correct

The `chord_to_blade` strip emit above has landed
(`midlines/shell_midline_export.py`), and measures correct in isolation. On
NACA 2412, chord 1.0, `C` family, zero twist, via `AirfoilSDF.from_naca`:

| stage | col0 | col1 | chord on |
|---|---|---|---|
| `AirfoilSDF.vertices` (raw NACA) | 1.0000 | 0.1216 | col0 |
| `section.components_unrotated` | 0.9880 | 0.0770 | col0 |
| `midline_twisted_pitch_axis_blade_frame` (emitted) | 0.1186 | 0.9880 | **col1 — canonical B** |

`chord_to_blade` is `naca_to_S âˆ˜ rotate_S_to_B`, so its input must be **raw
NACA**; fed that, it is correct. `tests/test_section_frame_canon.py` and
`section_mesh/tests/test_section_plane_canon.py` both pass (40 tests).

**The residual `SectionFrameViolation` seen from `section_selection` was not
this defect.** Two separate causes, both outside this package:

1. **BSP fixture frame.** `evaluate_station(coords=…)` documents Scale **S**
   ordering (col0 = flapwise, col1 = edgewise), and
   `scale_adapter.scale_s_to_naca_xy` swaps to NACA on that basis. The shared
   `sample_airfoil_coords` fixture supplied *NACA*-ordered coordinates, so the
   swap put the chord on the thickness axis and the guard correctly rejected the
   payload. Every FI evaluation reported `station has NO strength check`.
2. **`_ch.M22` did not exist.** With the guard cleared, `evaluate.py` failed on
   `AttributeError: 'ShellSolverChannels' object has no attribute 'M22'`.
   `ShellSolverChannels` is `N, Vy, Vz, My, Mz, T`; the hand-rolled block also
   passed `M22=` to `from_named`, which only accepts `My`. Replaced with
   `BeamResultantVector.from_flap_edge(...)`, which owns that composition.

Item 8's strip emit is correct; item 11 (the remaining cross-frame
`panel_x3_orientation` fallback) is fixed — upper caps face \(+X_B\).

**Consequence worth flagging.** FI had never actually executed through the BSP
path. With both causes fixed it runs and scales correctly (`max_fi ∝ M`), which
exposed a third defect — a criterion gap in per-component attribution:

3. **`fi_by_component` was zero on every isotropic zone.**
   `evaluate.py` built `fi_by_panel` from `quaternary.hashin_fi.in_plane`
   **only**. On an isotropic zone Hashin does not apply and that array is
   identically zero, while the real value sits in `quaternary.von_mises_fi`.
   Measured on 2B-CN, NACA 2412, isotropic skin/spar/web:

   ```
   governing        = {'fi': 0.687, 'panel_index': 9, 'criterion': 'von_mises'}
   hashin_fi.in_plane : shape (261, 1)  nanmax 0.0000
   von_mises_fi       : shape (261,)    nanmax 0.6870
   ```

   Because every component then scored 0.0, the component-aware FI search in
   `optimize.py` picked its governing DV by *tuple order* (`t_spar` first)
   rather than by physics — on `0B` it grew `t_spar` 0.003 → 0.0488 while FI
   got **worse** (2.654 → 3.273), when `t_skin` was governing at 3.14 against
   the spar's 0.36. Fixed by attributing across both criteria (row-wise max,
   the same rule `governing` uses). `fi_by_component` now reports
   `{skin 0.4712, spar 0.073, web 0.687}` on that section, and the FI-informed
   loop converges.

   The loop itself was never missing — `optimize.py` has had a component-aware
   FI phase (16 outer iterations, ×1.5 expansion, 8-step bisection) all along.
   It was starved of the input it steers by.

4. **Fixed — the search no longer commits to a DV before it has attribution.**
   `_governing_dv_for_growth` returns `None` (stop) when the Hashin gate is
   unmet and `fi_by_component` is empty, all-zero, or non-finite. Declaration
   order is not a substitute. Pinned by
   `section_selection/tests/unit/test_governing_dv_for_growth.py`.
   Zone-by-zone design in `section_layup` remains the longer-term replacement
   for the heuristic search; it is no longer a blocker for the empty-map case.

## Constraints / assembly architecture (2026-09-20)

Record of the former split ownership, not a numerical defect. Root-log
[BL-014](../../../../../../docs/bug_log.md#bl-014) / [BL-015](../../../../../../docs/bug_log.md#bl-015)
own the 10× tied-web / shear-channel narrative; this entry does not duplicate it.

| Topic | Before | After |
|-------|--------|--------|
| Junction MPCs / supports | Built inside `assemble_section_from_zones` and the runner | `processing/static/constraints/` composer only |
| Cap ties | `operations/cap_tie_constraints.py` (now a shim) | `constraints/cap_tie/` rigid-offset MPCs |
| Global assembly | Runner scatter-added a second time after `assemble_section_from_zones` | One scatter-add; `f_total = f_element + f_station` |
| Solve lifecycle | `execute` / `from_cache` / `from_global` mixed policy | Production: `SectionLinearStaticRunner.execute()`; kernel: `solve_constrained_system` |
| Loads | Equipollent could apply `beam_resultants_to_nodal_load` inside FE | Every production channel hands a frozen `SectionForceVector` |
| Shims | — | `connected_assembly_core`, `cap_tie_constraints`, `assemble_global_shell`, `from_cache`/`from_global` through next major release |

Verification: colocated constraint unit tests, `tests/test_constraint_*.py`, existing cap-tie / cascade / FI suites, and the non-slow shell-FEA campaign. CCX remains validation-only (global backend registration; no raising station solver).

## Channel C load application (2026-09-25)

Fixed: [FEA-L-001](../pre_processing/loading/docs/bug_log.md#fea-l-001) — Channel C
samples were lumped onto the nearest node pair with re-derived weights; now
consistent nodal loads on each strip's own elements. Found with
section_loads [SL-007 … SL-012](../../../load_utilities/section_loads/docs/bug_log.md#sl-007).

## FEA-L-002 — mesh-convergence "interior" peaks were not mesh-independent (2026-09-25)

**Status:** Fixed 2026-09-25
**Area:** `processing/static/diagnostics/cascade_mesh_convergence.py`;
`tests/test_cascade_mesh_convergence.py`

### Symptom

With exact Channel C loads (section_loads SL-007 … SL-012) the tied-convergence
guard failed: tied skin peak |N12| 65.6 k / 64.3 k / **170.2 k** N/m on the 3-rung
ladder. The same spike existed under the old loads (126.8 k at the finest rung).
Tied web "interior" |N12| jumped 47.4 k → 270.3 k on the finest rung.

### Root cause

- The asserted peaks included the elements at kind-crossing junctions (web feet,
  cap–web crossings), where the shell model has a stress singularity: that peak
  rises under refinement for any load, so no threshold makes it a convergence
  check. The 2.0× tolerance set on 2026-09-25 (after SL-001) was calibrated on a
  load field that happened to sit under it.
- `*_interior` peaks excluded an end band of `ENDPOINT_BAND = 2` **elements**:
  its physical width shrinks with refinement, and it never excludes interior
  T-junctions (a cap crossing a web mid-strip) — which is where the finest-rung
  web peak was.

### Fix

`junction_masks`: an element is in a junction neighbourhood when its midpoint is
within a **physical** radius of any node owned by more than one component kind
(panel end or interior). `sweep_cascade_convergence` fixes the radius for the
whole ladder at `ENDPOINT_BAND` element lengths of the coarsest rung, so every
rung reports the same region. The tied-convergence test asserts the junction-free
peaks (`peak_abs_n12_interior`, `peak_mid_sig_interior`) and keeps the raw peaks
reported.

### Verification

Tied |N12| away from junctions: skin 65.6 / 64.3 / 66.0 kN/m (1.03×), web 39.9 /
37.7 / 38.0 kN/m (1.06×), cap 465 / 554 / 452 kN/m (1.22×).

## Cap-tie / recovery exactness slice (2026-09-25)

Package-local graphify: `constraints/cap_tie/` (before: 37 nodes, god
`project_onto_polyline` / `build_cap_skin_ties`; after FEA-CT-001: 47 nodes,
god `build_cap_skin_ties` / `_locate_on_bonds` — membership via bonds);
`processing/static/results/` (369 nodes; god `solve_station_clpt_shell`,
FI / FSDT recovery); loading and diagnostics already closed as FEA-L-001 /
FEA-L-002.

<a id="fea-ct-001"></a>
## FEA-CT-001 — Cap ties re-project onto the skin polyline

**Status:** Fixed — 2026-09-25
**Area:** `processing/static/constraints/cap_tie/geometry.py`,
`calculix_ccx/cap_skin_ties.py`

### Symptom / evidence
`build_cap_skin_ties` picked the nearest skin polyline segment via
`project_onto_polyline` for every exclusive cap node. section_geometry emits
`ShellMidlineStrip.bonds` (SG-017) and Channel C uses them as identity
`WallBond`s; FE re-derived the host by distance.

### Root cause
Cap-tie builders took only mesh `panel_maps` / `node_meta` and rediscovered
membership by proximity instead of reading producer bonds on
`ShellMeshInputs.midlines`.

### Fix
- `build_cap_skin_ties` requires `shell_midlines`; locates each exclusive cap
  node on that cap's `StripBond.points`, reads the aligned `host_points` foot,
  and builds masters only on the named host skin panel.
- Unbonded caps / missing midlines under `skin_offset` raise (no nearest-skin
  fallback). `project_onto_polyline` remains for named-polyline location and
  diagnostic gap notes only.
- Same membership rule in CalculiX `build_cap_skin_tie_surfaces`.
- Midlines threaded through compose / assembly / runner / CCX mesh build.

### Verification
- `pytest …/tests/test_cap_tie_constraints.py` (bond host labels; refuse empty
  bonds; rigid-body residual).
- Package-local graphify after: god `_locate_on_bonds` / `bonds_by_cap_panel_label`
  alongside `build_cap_skin_ties`; `project_onto_polyline` no longer the
  membership hub.
- Recovery FrameTag adoption remains a follow-on (item 8 remainder).

### Recovery survey
Results tree: `except Exception=0`, `nearest=0`, `fallback=16` (documented
policy fallbacks). FrameTag not imported on results surfaces — follow-on to
tag pierce / ply-frame fields with `coordinate_frames` where they cross package
boundaries (item 2 harness already exists).
