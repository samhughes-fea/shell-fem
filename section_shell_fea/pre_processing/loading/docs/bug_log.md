# bug_log — section_shell_fea / pre_processing / loading

↑ [package log](../../../docs/bug_log.md)

Entries are `FEA-L-nnn`. Each records symptom, evidence, root cause and — once
fixed — fix and verification. Related: section_loads
[`SL-nnn`](../../../../../load_utilities/section_loads/docs/bug_log.md).

## Index

| ID | Title | Status |
|---|---|---|
| [FEA-L-001](#fea-l-001) | Channel C samples are lumped onto the nearest node pair with re-derived weights | **Fixed** 2026-09-25 — consistent nodal loads |

---

## FEA-L-001

**Channel C samples are lumped onto the nearest node pair with re-derived weights**

**Status:** Fixed 2026-09-25
**Found:** 2026-09-25 (tracing section_loads SL-007 to the FE)
**Area:** `section_force_vector_to_nodal_load.py` (production path:
`SectionLinearStaticRunner` with `load_channel="thinwall"`)

### Symptom

The external load the section FE solves under is not a consistent discretisation
of the Channel C force field.

### Root cause

`section_force_vector_to_nodal_load`:

- weights each sample with `section_loads.sample_arc_weights`, which re-derives
  lengths from coordinates and over-weights strip ends, drops closed-strip
  wrap-around segments, and gives a lone sample 10⁻⁶ m (SL-009);
- puts each sample's **whole** force on the single node pair nearest by
  coordinate (`_skin_node_pairs` — despite the name, every shell node pair).
  The mesh is finer than the strip vertices the samples sit on, so the load is
  concentrated on the nodes nearest those vertices instead of being distributed;
  at junctions, where skin, web and cap node pairs coincide, "nearest" is
  ambiguous;
- applies the sample's tangential density along the chosen node's own `u_s`
  direction, which is only the sample's tangent when that node belongs to the
  sample's strip.

The mesh is built from the same labelled strips, so each node's strip and arc
position are known — a sample-to-node mapping does not need to be guessed.

### Required resolution

Consistent nodal loads: interpolate each strip's density field (piecewise-linear
σ, piecewise-constant q — SL-009) onto that strip's own mesh elements and
integrate with the element shape functions, so ∑ nodal loads reproduces the
field's resultants exactly and no load crosses to another strip.

### Fix

`section_force_vector_to_nodal_load(node_meta, sfv, *, panel_maps, zones, ...)`
integrates each strip's field against **that strip's own** element shape
functions:

- the field is rebuilt from the strip's samples — Channel C's start / mid / end
  triplets (Simpson weights, detected from `ds`) as an exact quadratic per
  segment; other producers as a linear field through their samples;
- every sample must lie on its strip's polyline (1 nm), and its arc position is
  located there; a sample off its strip, or for a strip with no panel, is refused;
- each element is split at the field's breakpoints and `∫ f N ds` is integrated
  exactly on every piece;
- loads go 50/50 to the bottom/top pair, and the in-plane force vector is
  resolved onto each node's `(t, normal_2d(t))` — the basis
  `applied_force_resultant_yz` inverts, so the applied force equals the field's.

### Verification

`tests/section_loads/unit/test_section_force_vector_to_nodal.py`: a linear
density 100 → 50 N/m on one element gives nodal loads `(2·100 + 50)/6` and
`(100 + 2·50)/6`, summing to the field's integral. FE suites pass with Channel C
through the runner. The tied-convergence guard then exposed that its peak metric
included junction singularities — section_shell_fea FEA-L-002.

