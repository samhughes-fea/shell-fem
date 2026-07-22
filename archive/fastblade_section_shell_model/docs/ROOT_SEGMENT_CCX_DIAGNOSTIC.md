# Root-segment CCX diagnostic (clamped multi-row shell)

**Status:** research / diagnostic only. Not registered in production theory
dispatch (`station_theory_dispatch`, `shell_solve_registry`, or the optimiser
FI path).

**Production root path:** the **prismatic unit-length periodic RVE**
(`run_section_periodic_bc`, \(L_x = 1\,\mathrm{m}\)) at station 0 — the same
unit cell as every other station. Only the applied resultant
\(R(z_{\mathrm{root}})\) differs. This diagnostic never gates or replaces
production FI at the root.

## Goal

Quantify warping-restraint / Saint-Venant end effects at the blade root that a
periodic unit cell cannot represent. Solve a short **clamped** shell segment of
the first \(\sim 2\)–\(3\) root diameters of span in CalculiX (S4R), recover wall
resultants at the root-adjacent element row, and compare to the periodic RVE at
station 0 (peak membrane resultants and relative Hashin FI).

## Layout

| Piece | Path |
|-------|------|
| Multi-row lofted mesh | [`calculix_ccx/root_segment_mesh.py`](../calculix_ccx/root_segment_mesh.py) |
| Deck writer + solve driver | [`calculix_ccx/root_segment_export.py`](../calculix_ccx/root_segment_export.py) |
| Resultant / decay mapping | [`calculix_ccx/results.py`](../calculix_ccx/results.py) (`map_root_segment_stresses_to_resultants`, `parse_ccx_dat_node_reactions`) |
| Benchmark CLI | [`benchmarks/run_ccx_root_segment_diag.py`](../benchmarks/run_ccx_root_segment_diag.py) |
| Tests | [`calculix_ccx/tests/test_root_segment_ccx.py`](../calculix_ccx/tests/test_root_segment_ccx.py) |

## Model sketch

```text
x = 0 (ROOT_RING)                         x = L (OUTBOARD_RING)
   ●─────────●─────────●─────────●─────────●
   │  row 0  │  row 1  │  ...    │ row M-1 │
   ●─────────●─────────●─────────●─────────●
   clamp DOF 1–6                      *CLOAD on OUTBOARD_RING
                                      (statically equivalent N,V,M,T)
```

- Contour discretisation matches the existing strip mesh builders (shared junction
  nodes within each span layer).
- Default segment length \(L = 2.5\,D\) where \(D\) is the YZ extent of the root
  panels (`estimate_root_diameter_m`). Configurable via `segment_length_m` /
  `diameter_multiples`.
- Optional `station_panels` loft: guide sections along the span are linearly
  interpolated onto the node layers (prismatic when omitted). Contour mapping
  uses a normalised arc parameter so guide stations with different panel lengths
  still loft end-to-end.

## Load-application caveat

CalculiX 2.23 expands shells to 3D continuum and **rejects** both nodal
`*DISTRIBUTING` surfaces and `*RIGID BODY` MPCs on shell nodes. Beam resultants
are therefore applied as statically equivalent `*CLOAD` on the outboard ring
(`outboard_cloads_from_resultants`: weight `t·ds` for forces; `y`/`z` couples for
bending; tangential pattern for torsion).

This distribution is an **approximation** of Saint-Venant cut loads. The nodal
pattern near the cut is not unique and can pollute local stress within roughly
one section diameter of the outboard boundary.

**Trust region:** only the root-adjacent row (and the spanwise decay toward the
mid-segment) is compared to the periodic RVE. If the decay profile does not
settle before the root when \(L\) is increased, the coupling artefact is too
close — lengthen the segment or refine the outboard load transfer before drawing
conclusions.

**Root-matched moments:** the diagnostic benchmark targets the same root
resultants \(R_\mathrm{root}\) used by the periodic RVE. Outboard face loads are

\[
M_y^\mathrm{out} = M_y^\mathrm{root} - V_z L,\qquad
M_z^\mathrm{out} = M_z^\mathrm{root} + V_y L
\]

with forces unchanged, so a prismatic free body of length \(L\) has root-face
moments matching \(R_\mathrm{root}\). Without this correction, applying
\(R_\mathrm{root}\) unchanged at the outboard cut would overstate the root
bending moment by the shear lever arm \(V\cdot L\).

## FI trust model

Same as other CCX paths: mid-surface `.dat` stress → membrane / shear wall
resultants → CLT ply stresses → Hashin. Bending moments default to zero with
trusted diagnostic provenance unless FRD through-thickness stresses are merged
later. Relative FI ratios are therefore membrane-dominated unless FRD upgrade
is wired for the multi-row mesh.

## How to run

```bash
# Offline deck/mesh tests (no ccx)
pytest blade_precompute/section_shell_model/calculix_ccx/tests/test_root_segment_ccx.py -q

# Live ccx-gated solve test
pytest blade_precompute/section_shell_model/calculix_ccx/tests/test_root_segment_ccx.py -m ccx -q

# Full diagnostic printout (requires ccx on PATH)
python -m blade_precompute.section_shell_model.benchmarks.run_ccx_root_segment_diag
```


## Validation interpretation (no promotion path)

CCX validates the production path only. This diagnostic never becomes a production
root theory or an optimiser override: the prismatic unit-length periodic RVE remains
the sole production FI path at station 0. Use the checks below to characterise how
well that RVE holds against a clamped multi-row segment on representative root
geometries and tip-integrated \(R(z_{\mathrm{root}})\):

1. Root-face force equilibrium: \(\lvert \sum RF_{\mathrm{root}} + F_{\mathrm{applied}}\rvert / \lvert F\rvert \lesssim 5\%\).
2. Spanwise decay of \(\max\lvert N_x\rvert\) (and \(\lvert N_{xy}\rvert\)) settles to within
   \(\sim 10\%\) of the mid-segment value at least one diameter inboard of the
   outboard cut for the chosen \(L\).
3. Peak Hashin (or peak \(\lvert N_x\rvert\)) at the root-adjacent row vs the periodic
   RVE — record amplification; a sustained \(> 15\%\) difference on governing
   skin/web panels flags where the unit cell under-represents warping restraint
   (still not a promote trigger).
4. FRD bending coverage matters only when interpreting bending-governed relative FI.
5. Runtime is diagnostic-only (not an optimiser budget).

This path stays quarantined alongside the other CalculiX diagnostics
(`docs/PERIODIC_CCX_HOMOGENIZATION.md`, reaction-K6 benchmarks).

## First live run (2026-07-14)

Benchmark CLI defaults (`l_section` + `naca`, \(L/D = 2.5\) and \(5.0\), root-matched
outboard bending). Primary \(V_z\) equilibrium is excellent
(\(\lvert\sum RF_z + V_z\rvert / \lvert V_z\rvert \sim 10^{-7}\)–\(10^{-8}\)).

| Fixture | L/D | tip/mid \|Nx\| | mid CV | max root/RVE \|Nx\| | max Hashin ratio | Root > RVE (1.15)? |
|---------|-----|----------------|--------|---------------------|------------------|--------------------|
| l_section | 2.50 | 0.346 | 0.134 | 0.664 | 0.440 | no |
| l_section | 5.00 | 0.592 | 0.074 | 1.878 | 3.528 | yes |
| naca | 2.50 | 2.607 | 0.044 | 0.156 | 0.025 | no |
| naca | 5.00 | 4.140 | 0.027 | 0.264 | 0.070 | no |

**Interpretation (validation only):**

- On the L-section at \(L/D = 5\), root-adjacent peak \(\lvert N_x\rvert\) and relative
  Hashin exceed the periodic RVE by a material amount while midspan flatness
  improves vs \(L/D = 2.5\) — a flag that the unit cell under-represents warping
  restraint on this synthetic geometry under this load.
- On NACA, tip/mid \(\lvert N_x\rvert \gg 1\) shows the outboard \(\texttt{*CLOAD}\) cut still
  dominates the field; root-vs-RVE ratios are therefore not trustworthy until the
  load-transfer artefact is reduced (longer segment and/or finer outboard pattern).
- No change to the production root path (prismatic periodic RVE).
