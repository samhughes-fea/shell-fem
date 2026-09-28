# bug_log — section_shell_fea / pre_processing / loading

↑ [package log](../../../docs/bug_log.md)

Entries are `FEA-L-nnn`. Each records symptom, evidence, root cause and — once
fixed — fix and verification. Related: section_loads
[`SL-nnn`](../../../../../load_utilities/section_loads/docs/bug_log.md).

## Index

| ID | Title | Status |
|---|---|---|
| [FEA-L-001](#fea-l-001) | Channel C samples are lumped onto the nearest node pair with re-derived weights | **Fixed** 2026-09-25 — consistent nodal loads |
| [FEA-L-003](#fea-l-003) | Traction face loads place outline forces on the nearest skin node-pair | **Fixed** 2026-09-26 — refuse proximity; Channel B via labelled SFV |
| [FEA-L-004](#fea-l-004) | Vlasov Batho / panel map rediscovers web feet by outline nearest | **Fixed** 2026-09-25 — midlines / loops / coincidence |
| [FEA-L-005](#fea-l-005) | Vlasov silent singular Batho + fake ``I_ω`` clamp | **Fixed** 2026-09-25 |

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

---

<a id="fea-l-003"></a>
## FEA-L-003

**Traction face loads place outline forces on the nearest skin node-pair**

**Status:** Fixed — 2026-09-26
**Found:** 2026-09-25 (exactness item 8 recovery / loading survey)
**Area:** `traction_face_load.py` (`traction_face_loads`);
`section_linear_static_runner.py`; `station_theory_dispatch.py`

### Symptom / evidence
Each TractionField panel force was assigned by `argmin` distance to
`_skin_node_pairs` midline locations. Channel B FE overlay rediscovered
membership by proximity.

### Root cause
``traction_face_loads`` consumed an unlabelled ``TractionField`` and therefore
could not place loads on a named host the way FEA-L-001 / Channel C does from
labelled SFV samples. Dead runner/dispatch kwargs still accepted the overlay
even after production apply had moved to SFV.

### Fix
- ``traction_face_loads`` raises (FEA-L-003); keep `_skin_node_pairs` /
  `applied_force_resultant_yz` for legacy helpers / audits.
- Runner no longer accepts `traction_field` / `traction_points_yz` or
  `used_traction` bookkeeping.
- Dispatch: usable load-field traction selects Channel B and **requires** a
  pre-shaped labelled SFV; no overlay kwargs forwarded.
- Channel B FE apply is only FEA-L-001 (`section_force_vector_to_nodal_load`).

### Verification
`tests/test_linear_static_traction.py` (refuse; Channel B SFV parity);
`test_station_theory_dispatch.py` (SFV required; no overlay kwargs).

---

<a id="fea-l-004"></a>
## FEA-L-004

**Vlasov Batho / panel map rediscovers web feet by outline nearest**

**Status:** Fixed — 2026-09-25
**Found:** 2026-09-25 (item-8 masking survey after FEA-CCX-002)
**Area:** `thin_wall/section_vlasov.py` (`compute_section_vlasov`,
`_web_split_indices`, `_panel_global_s`, open-outline helpers)

### Symptom / evidence
Web feet and panel arc origins used `np.argmin` distance to outline vertices.
Substring `"web" in label.lower()` kind-bleeds vs `kind_of(label)`. Same
identity class as FEA-L-001 / FEA-CT-001 / BL-040: wrong cell / wrong panel when
junctions coincide. Competed with `thin_wall_section` (TW-001).

Fixture evidence (NACA 0012, spar at 0.35c): web feet from `webs_geom` sat
~2–6 cm from the airfoil open-outline vertices used by Batho — nearest was not
a coincidence snap. Producer midlines place the same feet within ~1e-17 m of
skin vertices.

### Fix
- Multi-cell Batho **requires** `shell_midlines` + `cell_wall_loops` (SG-014);
  refuses `webs_geom`-only / airfoil nearest rediscovery.
- Open outline built from skin walls named by cell `wall_labels`, chained by tip
  coincidence (`COINCIDENCE_TOL_M = 1e-9`); suction/pressure sides when tagged.
- Web feet from web midlines; `_web_split_indices` / `_panel_global_s` raise if
  off outline vertices beyond the coincidence tol.
- Shear centre from `shear_centre_from_midlines` when loops present (with
  [FEA-SC-002](../../../docs/bug_log.md#fea-sc-002)).
- Caps skipped for outline ω̂ placement (bond into skin — TW-001); webs use ω̂
  at the two outline feet.
- Kind checks use `kind_of(label)` (no substring bleed).

### Residual (cleared 2026-09-26)
[TW-002](../../../../geometry_utilities/section_thin_wall_model/docs/bug_log.md#tw-002)
closed: ``sectorial_warping`` on the wall graph; FEA ``compute_section_vlasov``
consumes it when midlines + loops are present.

### Verification
`tests/test_section_shell_model.py` Vlasov suite (NACA midlines/loops; box /
diamond Batho; refuses webs without midlines).

---

<a id="fea-l-005"></a>
## FEA-L-005

**Vlasov silent singular Batho + fake ``I_ω`` clamp**

**Status:** Fixed — 2026-09-25
**Found:** 2026-09-25 (item-8 masking survey)
**Area:** `thin_wall/section_vlasov.py` (`_batho_correction`,
`compute_section_vlasov`)

### Symptom / evidence
Singular Batho → zero correction (open-section ω̂); near-zero ``I_ω`` clamped
to ``1.0``.

### Fix
Raise with FEA-L-005; no silent fallback / clamp.

---

