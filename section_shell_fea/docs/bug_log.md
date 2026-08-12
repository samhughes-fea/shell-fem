# Bug log — section_shell_fea

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
| 8 | `geometry_utilities/section_geometry/interface/shell_midline_export.build_shell_midline_strips` | **Blocking.** Emits `midline_b` via `rotate_chord_to_blade` — a bare rotation that leaves the array chordwise-first with its origin at the LE (the rotated-NACA adapter frame, as that function's own comment says), not canonical B. Both `tests/test_section_frame_canon.py::test_chord_spans_Y_and_thickness_spans_X` and the pre-existing `mesh_utilities/section_mesh/tests/test_section_plane_canon.py` fail on it. Fix is `chord_to_blade` (swap + pivot shift + rotation) plus the matching `leading_edge_b` / `trailing_edge_b` emit. Everything downstream in this package assumes canonical B. |
| 9 | `compute_tertiary/examples/plot_x3_recovery_demo.py` | Not converted to the plot contract. Its geometric heuristics (`cap_xy[:,0].min()/max()` as chordwise, `chord = te[0]-le[0]`, `argmin(xy[:,1])` for the lower cap) all assume the swapped frame and need re-deriving for canonical B. Demo only — no importers besides `run_all_result_demos.py`. |
| 10 | Section-plane axis naming | `y`/`z` still name the section-plane axes throughout (`about_yz`, `nodes_yz`, `traction_points_yz`, `GlobalNodeMeta.yz`, `Iyz`, …): 522 occurrences over 47 files, plus shared DTO fields in `section_contract` and `mesh_utilities`. Canon names them \(X_B\) / \(Y_B\), with \(Z_B\) **spanwise** — so `z` for an in-plane axis collides with the canon spanwise axis. The arithmetic is self-consistent (`My = Σfx·dz` = edgewise, `Mz = -Σfx·dy` = flapwise, matching `blade_resultants_to_shell_channels`); this is a legibility defect, but it is the labelling that let #1 hide. `equilibrium_checks.py` also states the section plane is \((Y_B, Z_B)\), which is wrong on canon. |
| 11 | `mesh_utilities/section_mesh/inputs/panel_x3_orientation.py` | Adjacent, out of this package: the cap fallback tests `n[1] > tol` for `surface == "upper"` and uses `ref = [0, 1]`, i.e. "+x3 toward the TE". Under canonical B an upper cap faces \(+X_B\) (column 0). Only fires when a section has no skin strip, but it decides FE `+x3`. |
