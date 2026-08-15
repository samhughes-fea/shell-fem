# Bug log — section_shell_fea

Canonical package bug / frame-canon log. Docs index: [`README.md`](README.md).

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
| 8 | `shell_midline_export.build_shell_midline_strips` + `ShellMeshInputs` | **Blocking — two frames inside one DTO.** See below. |
| 9 | `compute_tertiary/examples/plot_x3_recovery_demo.py` | Not converted to the plot contract. Its geometric heuristics (`cap_xy[:,0].min()/max()` as chordwise, `chord = te[0]-le[0]`, `argmin(xy[:,1])` for the lower cap) all assume the swapped frame and need re-deriving for canonical B. Demo only — no importers besides `run_all_result_demos.py`. |
| 10 | Section-plane axis naming | `y`/`z` still name the section-plane axes throughout (`about_yz`, `nodes_yz`, `traction_points_yz`, `GlobalNodeMeta.yz`, `Iyz`, …): 522 occurrences over 47 files, plus shared DTO fields in `section_contract` and `mesh_utilities`. Canon names them \(X_B\) / \(Y_B\), with \(Z_B\) **spanwise** — so `z` for an in-plane axis collides with the canon spanwise axis. The arithmetic is self-consistent (`My = Σfx·dz` = edgewise, `Mz = -Σfx·dy` = flapwise, matching `blade_resultants_to_shell_channels`); this is a legibility defect, but it is the labelling that let #1 hide. `equilibrium_checks.py` also states the section plane is \((Y_B, Z_B)\), which is wrong on canon. |
| 11 | `mesh_utilities/section_mesh/inputs/panel_x3_orientation.py` | Adjacent, out of this package: the cap fallback tests `n[1] > tol` for `surface == "upper"` and uses `ref = [0, 1]`, i.e. "+x3 toward the TE". Under canonical B an upper cap faces \(+X_B\) (column 0). Only fires when a section has no skin strip, but it decides FE `+x3`. |

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
| `leading_edge_b` | `(0.0000, -0.3330)` | col 1 — canonical B, \(-f_{pivot}c\) |
| `trailing_edge_b` | `(-0.0000, +0.6667)` | col 1 — canonical B, \((1-f_{pivot})c\) |
| `midline_b` | col0 extent ~thickness, col1 extent ~chord | **col 1** — canonical B via `chord_to_blade` |

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
| `midline_b` (emitted) | 0.1186 | 0.9880 | **col1 — canonical B** |

`chord_to_blade` is `naca_to_S ∘ rotate_S_to_B`, so its input must be **raw
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

Item 8 stays **open** for the `panel_x3_orientation` concern (item 11) — the
dot-product across frames is unaffected by the above and still needs its own
check.

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

4. **Still open — the search commits to a DV before it has any attribution.**
   With (3) fixed, `test_sizing_handoff_characterization` goes green but the
   FI-only smoke (`1B-CN`, catalog composites, `EI_req=0`) still mis-converges.
   Traced trajectory:

   ```
   FI#01  t_skin=0.0020 t_spar=0.0030 t_web=0.0020  max_fi=None   by={}
   FI#03  t_skin=0.0020 t_spar=0.0068 t_web=0.0020  max_fi= 7.142  by={skin 7.14, spar 0.0, web 6.82}
   FI#08  t_skin=0.0020 t_spar=0.0500 t_web=0.0020  max_fi=13.014  by={skin 13.01, spar 0.0, web 7.15}
   FI#12  t_skin=0.0020 t_spar=0.0471 t_web=0.0020  max_fi=15.502  by={skin 15.50, spar 0.0, web 8.09}
   FI#17  t_skin=0.0030 t_spar=0.0500 t_web=0.0020  max_fi= 6.335  by={skin 5.03, spar 0.0, web 6.34}
   FI#18+ t_skin>=0.0045 t_spar=0.0500              max_fi=None    (cavity exhausted)
   ```

   The first evaluation returns `max_fi=None` (a 3 mm spar snaps to an
   unusable stack), so `fi_by_component` is `{}`, `worst` stays `0.0`, and the
   governing DV falls back to **tuple order** — `t_spar`. Fourteen evaluations
   then grow the one component whose FI contribution is identically zero, while
   skin governs at 7–15, and FI gets *worse* (7.14 → 15.50). By the time the
   loop corrects to `t_skin` at FI#17, `t_spar` is pinned at its 0.05 cap and
   the cavity can no longer accept skin growth, so every later trial fails.

   Two coupled faults: no defined behaviour when attribution is unavailable,
   and no exhaustion fallback when the chosen DV saturates
   (`_at_thickness_caps` requires *all* DVs at cap, so it never trips).
   Deliberately **not** patched in place — the search is being moved out of BSP
   into a `section_layup` worker that designs each zone against its own
   criterion, which removes the heuristic rather than tuning it.
