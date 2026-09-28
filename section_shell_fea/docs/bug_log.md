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
- Recovery FrameTag adoption closed as [FEA-R-001](#fea-r-001); demo-strength
  silent Hashin closed as [FEA-FI-001](#fea-fi-001).

### Recovery survey
Results tree (2026-09-25 re-graphify, 409 nodes): gods include
`ShellPlyResults` / `ShellElementalResults` / `solve_station_clpt_shell`.
`except Exception=0`, `nearest=0` on the results tree; documented policy
`fallback` sites remain. Traction nearest lives under FEA-L-003 (loading).

---

<a id="fea-r-001"></a>
## FEA-R-001 — Recovery containers lacked validated FrameTag fields

**Status:** Fixed — 2026-09-25
**Area:** `containers/ply_results.py`, `containers/elemental_results.py`,
`pre_processing/containers/recovery_types.py` (`ShellPanelResultants`)

### Symptom / evidence
Package-local graphify on `processing/static/results/` (409 nodes): recovery
god nodes (`ShellPlyResults`, `ShellElementalResults`, `solve_station_clpt_shell`)
carried E/M only in prose. Cross-package consumers could not validate frames
(ARCH-MUST-016). Survey follow-on from FEA-CT-001.

### Fix
- `ShellPanelResultants.frame` — required `FrameTag`, must be `FRAME_E`
- `ShellElementalResults.resultant_frame` — `FRAME_E`
- `ShellPlyResults.mid_surface_frame` / `ply_material_frame` — `FRAME_E` / `FRAME_M`
- `StationCLPTShellResult.mid_surface_frame` / `ply_material_frame` — same E/M pair
- `SectionShellRecoveryBundle.pierce_frame` / `ThinWallResult.pierce_frame` —
  `FRAME_B` for elastic-centroid `y_e`/`z_e` (section pierce plane)
- Tertiary orchestrator meta stamps the same tags

### Verification
Import/construct smoke; tertiary suite; refuse wrong tags in `__post_init__`
(where enforced). Bundle pierce tag covered in `test_recovery_frame_tags.py`.

---

<a id="fea-fi-001"></a>
## FEA-FI-001 — Governing Hashin silently used demo skin strengths

**Status:** Fixed — 2026-09-25
**Area:** `compute_quaternary/compute_governing_element.py` (`max_hashin_fi_from_bundle`)

### Symptom / evidence
Graphify surprising edge: `max_hashin_fi_from_bundle` → `default_skin_strengths_pa`.
When `strengths=None`, production used hard-coded glass/carbon demo allowables
instead of the panel laminate's ply strengths (unlike `select_governing_hashin_element`).

### Fix
Default path uses laminate-native `hashin_fi_per_ply_from_row` (same as the
governing-element selector). Explicit `strengths=` remains for demos/tests only;
missing laminate raises with FEA-FI-001. `default_skin_strengths_pa` docstring
marks demos-only.

### Verification
`tests/test_quaternary_orchestrator.py` and related FI suites.

---

<a id="fea-ccx-001"></a>
## FEA-CCX-001 — CalculiX SFV→N11/N12 used nearest-YZ / kind fallback

**Status:** Fixed — 2026-09-25
**Area:** `calculix_ccx/sfv_panel_loads.py`

### Symptom / evidence
`sfv_to_n11_n12` preferred samples by strip label, then fell back to
`sfv.by_kind(panel.kind)` and finally **all** samples; within the chosen pool
it picked the nearest YZ sample per mesh station. Same identity defect class as
FEA-L-001 / FEA-CT-001 — loads from another strip could bleed onto a panel.

### Root cause
Adapter re-derived membership by proximity / kind instead of requiring strip
label identity and arc placement on the panel polyline
(`COINCIDENCE_TOL_M = 1e-9`).

### Fix
- Membership: exact label match only (`skin:a` / bare strip `a` forms).
- Placement: locate each matched sample on the panel polyline; interpolate
  `f_x`/`f_s` in arc-s onto mesh stations.
- No kind-wide or global-nearest fallback — unmatched panels get zeros.

### Verification
`calculix_ccx/tests/test_sfv_panel_loads.py` (label+arc, bare strip, refuse
kind bleed, empty → zeros).

---

<a id="fea-ccx-002"></a>
## FEA-CCX-002 — Cap ``*EQUATION`` re-picked nearest masters

**Status:** Fixed — 2026-09-25
**Area:** `calculix_ccx/cap_skin_ties.py`, `section_export.py`,
`ccx_cascade_mesh_convergence.py`

### Symptom / evidence
After FEA-CT-001, ``*TIE`` membership was StripBond-correct, but
``cap_tie_export="equation"`` still equated each slave to the **nearest** master
element node by XYZ. Cascade silently switched tie → equation on any TIE
failure, masking the defect.

### Root cause
Equation path rediscovered masters by proximity instead of reusing the bond-foot
segment/weights already computed for ``*TIE`` surfaces (same class as MITC4
``CapTieLink``).

### Fix
- `CcxCapEquationLink` + `CcxCapSkinTieSurface.equation_links` recorded at bond
  locate time (named-host polyline masters + interpolating weights).
- `_emit_cap_skin_ties` equation mode emits weighted multi-master ``*EQUATION``;
  refuses empty links (no nearest rediscovery).
- Cascade no longer auto-falls back to equation; callers must pass
  `cap_tie_export="equation"` explicitly.

### Verification
`calculix_ccx/tests/test_joint_validation.py::test_equation_fallback_export`
(requires `equation_links`; deck has ``*EQUATION``).

---

<a id="fea-c-001"></a>
## FEA-C-001 — Elastic centroid silently skipped failed panels

**Status:** Fixed — 2026-09-25
**Area:** `validation/section_centroid.py`

### Symptom / evidence
`compute_elastic_centroid_from_panels` caught broad ``Exception``, warned, and
``continue``d; empty A11 sum returned ``(0.0, 0.0)``. Pierce axis identity was
masked when ABD failed or all panels were skipped.

### Fix
Raise with panel label on ABD failure or short geometry; refuse zero A11 sum
(no silent origin pierce).

### Verification
`validation/tests/test_section_centroid.py` (raise paths).

---

<a id="fea-r-002"></a>
## FEA-R-002 — Mesh / load / Vlasov pierce coords untagged

**Status:** Fixed — 2026-09-25
**Area:** `pre_processing/mesh/shell_mesh_map.py` (`GlobalNodeMeta`),
`containers/load_channel_types.py` (`SectionExternalLoadRequest`),
`loading/thin_wall/section_vlasov.py` (`SectionVlasovResult`)

### Symptom / evidence
FEA-R-001 tagged recovery DTOs; mesh pierce `yz`, traction outline points, and
Vlasov shear-centre coords still lacked validated `FrameTag` (ARCH-MUST-016).

### Fix
- `GlobalNodeMeta.frame = FRAME_B` (required)
- `SectionExternalLoadRequest.traction_pierce_frame = FRAME_B`
- `SectionVlasovResult.pierce_frame = FRAME_B`

### Verification
`tests/test_recovery_frame_tags.py` (mesh + load request refuse wrong tags).

---

<a id="fea-fi-002"></a>
## FEA-FI-002 — Interlaminar allowable silently used S12 for missing S13

**Status:** Fixed — 2026-09-25
**Area:** `compute_quaternary/compute_hashin_fi.py`
(`interlaminar_allowable_pa` / `interlaminar_allowable_from_lam`)

### Symptom / evidence
Scalar interlaminar path substituted in-plane ``S12`` whenever ``S13`` was
absent, under/over-stating interface FI without provenance. Authoritative
`interface_shear_screening_from_tau` already gated S12 behind
`allow_s12_fallback`.

### Fix
Default: return ``0.0`` (caller skips FI) when S13 missing. ``S12`` only when
`allow_s12_fallback=True` (explicit legacy/demo).

### Verification
`tests/test_interlaminar_s13_exactness.py`.

---

<a id="fea-l-003-pkg"></a>
## Cross-ref: FEA-L-003 traction nearest (loading)

Logged under
[`pre_processing/loading/docs/bug_log.md#fea-l-003`](../pre_processing/loading/docs/bug_log.md#fea-l-003);
**Fixed** 2026-09-26 — refuse proximity; Channel B via labelled SFV / FEA-L-001.

<a id="fea-l-004-pkg"></a>
## Cross-ref: FEA-L-004 Vlasov nearest membership (loading)

Logged under
[`pre_processing/loading/docs/bug_log.md#fea-l-004`](../pre_processing/loading/docs/bug_log.md#fea-l-004);
**Fixed** 2026-09-25 — midlines / loops / coincidence (see loading log).

---

<a id="fea-ccx-003"></a>
## FEA-CCX-003 — CCX / diagnostics web kind substring bleed

**Status:** Fixed — 2026-09-25
**Area:** `calculix_ccx/thin_wall_loads.py`,
`processing/static/diagnostics/equilibrium_checks.py`

### Symptom / evidence
`"web" in label.lower()` treated labels like `skin:webbed_…` as webs (same
bleed FEA-L-004 removed from Vlasov).

### Fix
Unified `kind_of(label) == "web"` (cap_tie helper).

### Verification
`tests/test_kind_and_sc_exactness.py`.

---

<a id="fea-l-005"></a>
## FEA-L-005 — Vlasov silent singular Batho + fake ``I_ω``

**Status:** Fixed — 2026-09-25
**Area:** `loading/thin_wall/section_vlasov.py`

### Symptom / evidence
`LinAlgError` on Batho → `c_vec = 0` (open-section ω̂ for closed multi-cell);
`I_omega_E < 1e-60` → clamp to ``1.0`` inventing finite σ_ω/q_ω.

### Fix
Raise with FEA-L-005 on singular Batho and near-zero ``I_ω`` (no silent clamp).

### Verification
Existing Vlasov suites still pass on well-posed sections; raise paths covered by
docstring contract / unit probes as needed.

---

<a id="fea-r-003"></a>
## FEA-R-003 — Bundle / CLPT FrameTag defaults unenforced

**Status:** Fixed — 2026-09-25
**Area:** `station_analysis_types.py`, `local_clpt_shell.py`

### Symptom / evidence
FEA-R-001 added pierce / E/M defaults but bundle, `ThinWallResult`, and
`StationCLPTShellResult` did not `require_frame_tag` / refuse wrong tags.

### Fix
`__post_init__` validation mirroring FEA-R-001 / FEA-R-002.

### Verification
`tests/test_recovery_frame_tags.py` refuse wrong pierce / E/M tags.

---

<a id="fea-sc-001"></a>
## FEA-SC-001 — Opt-in SC axis silent ``t=0.01`` default

**Status:** Fixed — 2026-09-25 (multi-cell SC path closed as FEA-SC-002)
**Area:** `loading/shear_centre_loading_axis.py`

### Symptom / evidence
`thickness_m … or 0.01` invented wall thickness when missing; SC was
outer-skin-only until FEA-SC-002.

### Fix
Raise when `thickness_m` missing/≤0 (FEA-SC-001). Full SC via
`shear_centre_from_midlines` when loops present is [FEA-SC-002](#fea-sc-002)
(also Fixed).

### Verification
`tests/test_kind_and_sc_exactness.py`; `tests/test_eq_sc_exactness.py`.

---

<a id="fea-jn-001"></a>
## FEA-JN-001 — MITC4 re-clustered junctions the strip mesh already emitted

**Status:** Fixed — 2026-09-25
**Area:** `pre_processing/mesh/connected_panel_topology.py`

### Symptom / evidence
`build_connected_panel_maps` called `cluster_points(endpoints, endpoint_tol)`
after consuming `section_strip_mesh`, ignoring producer
``SectionStripMesh.clusters``. CalculiX already used
`clusters_from_section_strip_mesh` — MITC4 rediscovered membership by proximity.

### Fix
Prefer producer clusters; validate every panel endpoint appears in exactly one
cluster; refuse FE re-clustering (``endpoint_tol`` unused).

### Verification
`tests/test_jn_t_bk_exactness.py`; assembly / buckling / hybrid suites.

---

<a id="fea-t-001"></a>
## FEA-T-001 — Isotropic laminate invents ``t=1e-3``

**Status:** Fixed — 2026-09-25
**Area:** `element_library/formulation_factory.py`
(`_isotropic_to_laminate_definition`)

### Symptom / evidence
Missing zone/lam thickness fell through to silent ``1e-3`` m (same class as
FEA-SC-001).

### Fix
Raise with FEA-T-001 when thickness missing/≤0.

### Verification
`tests/test_jn_t_bk_exactness.py`.

---

<a id="fea-bk-001"></a>
## FEA-BK-001 — Buckling panel inputs invent thickness / width

**Status:** Fixed — 2026-09-25
**Area:** `simulation_runner/section_linear_buckling_runner.py`,
`api/linear_buckling.py`

### Symptom / evidence
Missing thickness → ``1e-3``; missing width → ``1.0`` m into eigenbuckling.

### Fix
Require positive thickness (from ``panel_thickness_m`` or row) and positive
``panel_widths_m`` (or arc-derived from ``panel_maps.s_nodes``); raise otherwise.

### Verification
`tests/test_jn_t_bk_exactness.py`; `test_section_linear_buckling_runner.py`.

---

<a id="fea-eq-001"></a>
## FEA-EQ-001 — Diagnostic junctions by proximity; fallbacks green-pass

**Status:** Fixed — 2026-09-25
**Area:** `processing/static/diagnostics/equilibrium_checks.py`

### Symptom / evidence
`_build_endpoint_clusters_raw` rediscovered interfaces by geometric proximity
(same class as FEA-JN-001). ``field-fallback`` / ``centroid-fallback`` N12
modes could still set ``resultant_pass`` True. Degenerate panels invented a
``(1,0)`` end tangent.

### Fix
- Optional ``section_strip_mesh`` / ``producer_clusters`` on
  ``check_panel_equilibrium`` / ``check_cluster_equilibrium``.
- ``resultant_pass`` authoritative only for ``traction-vector-strict``;
  ``resultant_metric_ok`` keeps the numeric gate for informational modes.
- ``_panel_end_tangent`` raises on degenerate geometry (FEA-EQ-001).

### Verification
`tests/test_eq_sc_exactness.py`; equilibrium suite in
`test_section_shell_model.py`.

---

<a id="fea-sc-002"></a>
## FEA-SC-002 — Opt-in SC ignored ``cell_wall_loops`` when present

**Status:** Fixed — 2026-09-25
**Area:** `loading/shear_centre_loading_axis.py`

### Symptom / evidence
Even with ``ShellMeshInputs.cell_wall_loops``, SC used outer-skin sectorial
estimate only (multi-cell wrong vs thin_wall_section).

### Fix
When midlines + loops present, call ``shear_centre_from_midlines``. Else keep
outer-skin path (FEA-SC-001 thickness refuse). Skin selection also accepts
``kind_of(label) == "skin"``.

### Verification
`tests/test_eq_sc_exactness.py`.

**Note:** Vlasov Batho / panel map identity closed as
[FEA-L-004](../pre_processing/loading/docs/bug_log.md#fea-l-004). Sectorial-ω̂
in `thin_wall_section` closed as
[TW-002](../../../geometry_utilities/section_thin_wall_model/docs/bug_log.md#tw-002).

---

<a id="fea-ccx-004"></a>
## FEA-CCX-004 — CCX recovery invented unit edge length

**Status:** Fixed — 2026-09-25
**Area:** `calculix_ccx/results.py`

### Symptom / evidence
Zero-length panel edges used ``length = … or 1.0``, inventing a 1 m tangent
basis for stress → resultant conversion (masking degenerate mesh geometry).

### Fix
Raise with FEA-CCX-004 on non-positive edge length (three call sites: element
membrane recovery paths and panel-station helper).

### Verification
Raise contract in `results.py`; covered by CCX recovery paths that always
emit positive edge lengths on valid meshes.

---

<a id="fea-v-001"></a>
## FEA-V-001 — Validation interlaminar ref invented ``t=1e-3``

**Status:** Fixed — 2026-09-25
**Area:** `validation/section_integrals.py`
(`physical_interlaminar_shear_from_sigma_gradient`)

### Symptom / evidence
``zones[0].lam.t … or 1e-3`` invented wall thickness for the spanwise
σ-gradient interlaminar reference (same class as FEA-T-001 / FEA-SC-001).

### Fix
Require positive ``lam.t``; raise FEA-V-001 otherwise.

### Verification
Raise contract in `section_integrals.py`.


---

<a id="fea-eq-002"></a>
## FEA-EQ-002 — Section carries no axial membrane force: bending moment never reaches N11

**Status:** Open — found 2026-09-27, not fixed
**Area:** `pre_processing/loading/section_force_vector_to_nodal_load.py` (axial load split),
`processing/static/constraints/junction/supports.py` (`build_clamp_supports`),
`calculix_ccx/section_export.py` (`_cload_entries`, `_support_node_ccx_ids`)
**Root log:** [BL-044](../../../../../../docs/bug_log.md#bl-044)

### Symptom / evidence

Found while building a MITC4 ↔ CalculiX cross-check for the CDFAM deck. Station:
`2D-F`, NACA 2412, chord 1 m, catalogue materials 13 / 17 / 13, realised gauges
4.0 / 12.155 / 5.0 mm, production chain `SectionGeometryRunner` →
`SectionBindRunner` → `build_section_strip_mesh` → `SectionLoadsRunner`
(thin-wall, Channel C) → `SectionLinearStaticRunner(load_channel="thinwall",
cap_tie_mode="skin_offset")`. Recovered elemental resultants
(`secondary_results.elemental_results.resultant_rows`) integrated over the strip
mesh (`Σ N · ds`, `Σ N · y · ds` per element, `mesh.panel_meshes[i].yz_nodes`):

| Applied (pure) | ∫N11 ds | ∫N11·y ds | Where the load went |
|---|---|---|---|
| N = 1.0e5 N | **3.3e-3 N** | 2.4e-3 N·m | RBM pin `u_x` reaction at nodes 0,1 = **−99 957 N** |
| M_flap = 8.0e4 N·m | 2.4e-2 N | **−1.5e-2 N·m** | ∫N12·y ds = −3.27e4 N·m (in-plane shear) |
| V_flap = 1.0e4 N | 13 N | 6 N·m | shear path; axial as expected ≈ 0 |

The load itself is right: the SFV reproduces the applied resultants to 1e-12 and
`Σ f[U_X] = N` exactly. The response is wrong: the section does not stretch, so
N11 — the membrane force that carries bending in every blade section — is
essentially zero, and the moment is carried (partly) by in-plane shear instead.

CalculiX fed the same SFV (`sfv_to_n11_n12` → `solve_global_coupled_ccx`) recovers
∫N11·y ds ≈ −3.25e4 N·m of the applied 8.0e4 — also short, differently.

### Root cause (suspected; to confirm before fixing)

The slice has **no end face that can react axial load**:

1. **Supports.** `build_clamp_supports` clamps `(W, BETA_S, BETA_X)` at panel
   endpoints on both layers and pins `u_x` at **two nodes only** (RBM pin). The
   CCX export mirrors this (`_support_node_ccx_ids`: `U_Z, TH_X, TH_Y` at
   endpoints + `U_X, U_Y` at two anchor nodes).
2. **MITC4 load split.** `section_force_vector_to_nodal_load` puts
   `0.5 · f_axial` on the bottom node **and** `0.5 · f_axial` on the top node of
   every pair — same sign on both end faces. A slice loaded identically on both
   faces in `x` translates or warps; it is not stretched, so ε11 ≈ 0 and the
   whole net axial force drains to the two-node pin.
3. **CCX load split.** `_cload_entries` loads the **top layer only**. With the
   bottom face free in `x` except the two anchors, the axial load reaches the
   anchors through a mixture of membrane and shear paths — hence ~41 %, not 0 %
   and not 100 %.

A beam slice under moment needs equal and **opposite** axial tractions on its two
end faces (or one face reacted in `u_x`), so that N11 = σ t is uniform along x.

### Consequences

- Every MITC4 Hashin FI omits bending stress: section-selection station FIs are
  1e-6 … 3.5e-3 on the passing 2D-F run `2026-09-26_04-38-13`, so the FE FI gate
  cannot bind — the same *vacuous gate* class as root BL-009.
- `tests/test_layup_analytic_matches_fe.py` passes for the wrong reason: the
  analytic FI is evaluated on the FE's own `resultant_rows`, so it inherits the
  missing N11 rather than checking it.
- The existing hard gate (LOAD_APPLICATION_DOUBLE_COUNTING.md) reconstructs R from
  the **load vector** `f`, never from the **solved response**, so it cannot see
  this.
- The MITC4 ↔ CalculiX comparison validates neither: both share the support set.
- Slide 36 of the CDFAM deck ("peak Hashin FI 0.51") inherits this; slide 39
  states it as an open item.

### Reproduce

Scratch probe `fe_equilibrium_probe.py` (session scratchpad, 2026-09-27): the
three pure cases above through the production runners, printing ∫N·ds, ∫N·x ds,
∫N·y ds and the RBM-pin reaction.

### Proposed fix direction (not implemented)

- Decide the slice contract explicitly: reacted root face (`u_x` clamped on the
  bottom layer, loads on the top) **or** self-equilibrated antisymmetric end
  tractions (−½ bottom, +½ top) with a generalised-plane-strain constraint.
- Add a **response** equilibrium gate after solve: ∫N11 ds = N,
  ∫N11·(y, x) ds = (M_flap, M_edge) and the shear resultants within 1e-6 —
  refuse the result otherwise, per the package's no-masking rule.
- Re-run the CCX cross-check only after both models share a correct contract.
