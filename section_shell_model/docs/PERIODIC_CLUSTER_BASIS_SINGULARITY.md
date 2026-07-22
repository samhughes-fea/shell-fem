# Periodic homogeniser × cluster-basis MPC singularity

> **Non-product / historical.** Channel-A periodic product APIs are removed (Phase 0).
> Production FI uses contour-supported linear-static with `shared` junctions.
> Spike 1B (`transformed_basis` K6) remains permanently parked in
> [DEFERRED_RESEARCH.md](DEFERRED_RESEARCH.md).

**Status:** partially addressed; a deeper bug found and deferred. The default
production path (`shared` junctions) is correct and unaffected throughout.

## Spike 1A confirmation (2026-07-14)

Re-ran the closed N-gon discriminator (`_ngon_panels`, ne=4) and locked the failure
signature in `test_spike_1a_ngon_wrong_k6_signature`:

| N | axial K6 ratio (xf/shared) | torsion ratio | worst-mode reldiff |
|---|----------------------------|---------------|--------------------|
| 24 | **0.8775** | ~259 | ~388 |
| 48 | **0.8775** | ~918 | ~1376 |
| 96 | **0.8775** | ~3830 | ~5745 |

Under unit axial macro: `transformed_basis` has max `|u_s|≈0.19`, `|w|≈0.35`;
`shared` has `|u_s|`/`|w|` ~ machine zero. Bending diagonals match to relative 1e-4.
`SHELL_MODEL.md` wording updated: the path is **non-singular but wrong**, not singular.
Constraint-manifold diagnosis stands; next work is Spike 1B reformulation candidates.

## Spike 1B result (2026-07-14) — park permanently

Tried Approach-B / clamp variants on the N=48 N-gon (ratios = xfmd/shared diag):

| Candidate | axial | ky/kz | torsion | shear | notes |
|-----------|-------|-------|---------|-------|-------|
| Baseline (current) | 0.8775 | 1.00 | ~918 | ~0.64 | breathing `|u_s|~0.19`, `|w|~0.35` |
| Clamp cluster `W` only | **1.00** | 1.00 | ~918 | ~0.64 | kills breathing; torsion still diverges |
| Clamp `W`+`BETA_S`+`BETA_X` | 1.00 | ~1.00 | ~918 | ~0.66 | same class |
| Clamp `US`+`W` | 1.00 | ~1.12 | ~953 | ~1.39 | breaks bending/shear |
| Clamp all cluster-master DOFs | 1.00 | ~1.45 | ~953 | ~1.65 | Approach B full condense |

No candidate meets the promote exit criteria (axial/bending/**torsion** within ~1% of
`shared` on N=96 with mesh-stable shear). Clamping `W` restores axial without the
bending damage of full condense, but **torsion remains ~N² divergent** — the
formulation gap is not closable by free-set clamping heuristics.

**Policy:** Will **not** feature a production periodic `transformed_basis` K6 path.
Keep `shared` for periodic FI; keep `xfail(strict)` N-gon oracles +
`test_spike_1a_ngon_wrong_k6_signature` as regression locks; use direct (non-periodic)
`transformed_basis` assembly for sharp-junction diagnostics only. Defect K remains an
accepted limitation on the periodic FI path (see [SHELL_MODEL.md](SHELL_MODEL.md)).

## Status update (2026-06)

The original `Factor is exactly singular` crash is **fixed** — `compute_k6_periodic_bc`
now composes the cluster-basis MPCs with periodic pairing (Approach A, see below), and
the periodic `transformed_basis` solve runs for both MITC4 and HSDT. Verified: mesh
layer element-agnostic (ndof 6 vs 8 identical), axial/bending K6 match `shared`.

But a **second, deeper bug** was then found and is **NOT yet fixed**: the periodic
`transformed_basis` K6 is *wrong* for the torsion/transverse-shear (and axial) modes.
A closed N-gon discriminator (where `shared` reproduces the analytic beam stiffness
`axial = A11·perimeter` exactly) shows `transformed_basis` does **not** converge to
`shared` as junction angle → 0: axial is a fixed 0.877× of EA (12% soft) and torsion
**diverges** with mesh/N. Root cause (diagnosed, see below): the virtual-junction
representation admits a **spurious cross-section distortion mechanism** under the
periodic macro modes — radial "breathing" under axial (Poisson-like), which `shared`'s
rigid junction forbids.

This is a **formulation-level** issue, not a coding bug:
- An independent reduction through the trusted `reduce_propagate_solve` reproduces the
  same wrong K6 → the bespoke transform is algebraically correct; the constraint
  manifold itself is wrong.
- Simple fixes were ruled out: clamping rot `BETA_S` (what the *direct* solver does) or
  tying the rot hinge has **no effect**; clamping the cluster-main `W`+rotations fixes
  axial exactly but breaks the shear modes and leaves torsion divergent.

A correct fix is a research-level reformulation (constrain the cross-section to deform
only via beam-admissible modes while keeping the rotated-basis compatibility). **Deferred**
— production uses `shared` (provably correct here) and does not need the periodic
`transformed_basis` path. Two `xfail(strict)` discriminator tests in
`tests/test_periodic_cluster_basis.py` lock the broken behaviour as the regression
target so it cannot silently regress and a future fix has a ready oracle. Pick this up
only if periodic shear-lag at genuinely sharp skin–web corners becomes a near-term need;
the *direct* `transformed_basis` assembly (validated within 1%) serves non-periodic
sharp-junction analysis meanwhile.

---

The remainder of this document is the original singularity write-up and the Approach-A
plan that fixed the crash; the Step 3 validation below is what surfaced the deeper bug.

## Summary

The resultant-driven section solve (`compute_k6_periodic_bc`) and the cluster-basis
(`transformed_basis`) interface-constraint scheme were built independently and never
integrated. Running the periodic solve with `transformed_basis` fails with
`Factor is exactly singular`. This blocks shear-lag validation at sharply
non-collinear junctions (e.g. skin–web corners) under resultant loading.

This is **pre-existing** — it is not introduced by the HSDT cluster-basis port (#3).
The HSDT port is validated through the *direct* nx/nxy assembly path, which has no
periodic-pairing step; only the periodic (homogenisation) path hits the singularity.

## The two mechanisms that don't compose

### 1. Periodic homogeniser — pairs nodes by physical position

The production resultant solve treats the section as a unit-length prismatic cell and
makes it **periodic**: every inboard cut-face node (x=0) is tied to its geometric twin
on the outboard face (x=L), so a macro strain (ε0, κ_y, κ_z, torsion, the two shears)
can be imposed across the cell.

Pairing is by physical coordinate, and **NaN-coordinate nodes are dropped**:

| Concern | Location |
|---------|----------|
| KD-tree pairing on `(y,z)`, NaN filter `physical = ~np.isnan(yz[:, 0])` | [homogenisation.py:644-669](../engine/homogenisation.py#L644-L669) |
| Topological pairing **raises** on NaN-coord paired node | [homogenisation.py:700-703](../engine/homogenisation.py#L700-L703) |
| Unit macro-mode matrix built from each node's `(y,z)` | `_build_unit_mode_matrix` ([homogenisation.py:131](../engine/homogenisation.py#L131)), used at [homogenisation.py:228](../engine/homogenisation.py#L228) |

### 2. Cluster-basis (transformed_basis) MPCs — introduce virtual nodes with NaN coords

For a sharply non-collinear junction (skin meeting a web at an angle) endpoint nodes
cannot simply be made coincident. Instead a **virtual cluster reference node** is
introduced at the junction with *no element stiffness of its own*; the real panel
endpoint DOFs are tied to it through a 2D rotation. These virtual reference nodes
deliberately carry **NaN `(y,z)` coordinates** — they are algebraic MPC masters, not
mesh points.

`collect_global_node_meta` ([shell_mesh_map.py:172](../engine/shell_mesh_map.py#L172))
only iterates `panel_maps` to populate `yz`/`is_top`/`tangent`. Cluster reference/rot
nodes are allocated separately from panel nodes — in fact *ahead* of them, taking the
low ids `0 .. 2·n_layers−1` ([connected_assembly_core.py:105-121](../engine/connected_assembly_core.py#L105-L121)) —
and are never visited, so they keep `yz = NaN`, `is_top = False`. (The ordering is
immaterial; being unvisited is what leaves them NaN.)

### Why it goes singular

The virtual cluster nodes are **master DOFs** in the constraint system but are invisible
to the periodic homogeniser: the NaN filter excludes them from inboard/outboard pairing,
and nothing else in the periodic path constrains them. After MPC reduction the retained
master DOFs are neither periodically paired nor supported → zero-energy free modes →
the reduced stiffness is rank-deficient → factorization fails (`Factor is exactly
singular`).

The same failure reproduces with `run_section_periodic_bc_mitc4` + `transformed_basis`,
confirming it is a shared production limitation, not HSDT-specific.

## Practical consequence

| Configuration | Status |
|---------------|--------|
| Collinear / near-collinear junctions, `shared` mode (production default) | works |
| Non-collinear junctions, direct nx/nxy assembly, `transformed_basis` | works (validated in #3) |
| Non-collinear junctions, **periodic resultant solve**, `transformed_basis` | **singular** |

The gap is specifically: non-collinear junctions under resultant loading via the
periodic solve — the configuration needed to validate shear-lag stress transfer at the
skin–web corner (the deferred #3 headline validation, relevant to the delamination
constraint in #4).

## Architectural boundary — element-agnostic mesh

The fix must respect the project-wide seam: **the meshing/constraint infrastructure is
element-formulation-agnostic, and each element (FSDT/MITC4 ndof=6, HSDT ndof=8) plugs
into the same mesh modularly.** That seam already exists and the fix must stay inside it:

| Layer | Element-agnostic? | Where | Fix touches it? |
|-------|-------------------|-------|-----------------|
| Geometry/topology (`yz`, `is_top`, `tangent`, `pair_group`, panel maps) | **yes** — pure geometry | `shell_mesh_map.py`, `connected_assembly_core.py::assign_global_nodes_cluster_basis` | **yes (core of A)** |
| DOF-semantics convention (local idx 0–5 = `U_X,U_S,W,BETA_S,BETA_X,BETA_ZET`; extras via `identity_extra_dofs`) | **yes** — `ndof`-parameterised | `connected_assembly_core.py` (constraints/supports) | **yes (clamp step)** |
| K6 extraction / macro-mode jump (`compute_k6_periodic_bc`, `_build_unit_mode_matrix`) | **yes** — `ndof`-parameterised, relies only on the 0–5 convention | `homogenisation.py` | no change needed |
| Resultant recovery (`compute_resultants_from_modes` → `mitc4_resultants`) | **no** — hardcodes `_NDOF_NODE` / MITC4 | `homogenisation.py:410-611` | **out of scope** (see below) |

Consequences for the plan:

- **This is why (A) beats (B).** (A)'s core is *pure geometry* (`yz`/`pair_group`/`is_top`
  on cluster nodes) and lands in the agnostic layers, so the same fix serves MITC4 and
  HSDT unchanged. (B) "special-cases the homogeniser" — the element-coupled path — which
  is the wrong side of the seam. Under the modularity goal (A) is effectively required.
- **The contract the fix depends on:** the shared **0–5 DOF convention** is the mesh↔element
  interface. The macro-mode jump (`d_0`, [homogenisation.py:796-812](../engine/homogenisation.py#L796-L812))
  assumes DOF 0 = u_x, DOF 1 = u_s for *every* formulation — true today per
  `connected_assembly_core` — so the plan must hold that invariant, not reintroduce
  per-element DOF indexing.
- **The one remaining element-bound seam is deliberately out of scope:** resultant recovery
  hardcodes `_NDOF_NODE`/`mitc4_*`. A future element plugs into the periodic mesh by
  swapping **only** the recovery function — never the meshing/pairing/constraint code.
  Flag it; do not "fix" it here.

## Recommendation — steps forward

### Step 0 — Lock the target with a failing test

Add a test running `run_section_periodic_bc_mitc4` (and the HSDT twin) with
`transformed_basis` on the NACA+spar section that currently raises
`Factor is exactly singular`. This makes the regression target explicit and proves any
fix on the **production** (MITC4) path, not just HSDT.

Add, in the same test, an **element-agnosticism assertion**: the geometric/topological
artifacts produced for `ndof=6` and `ndof=8` on the identical section must match exactly
— `node_meta.pair_group`, the cluster node `(y,z)`/`is_top`, and the `T_pbc` sparsity
pattern. This is the executable guarantee that the mesh layer stayed element-agnostic
(per the architectural-boundary section); a future element formulation must reproduce it.

### Step 1 — Decide the approach

- **(A) Give cluster reference nodes a geometric position so they pair like physical
  nodes — _recommended_.** Each junction sits at a real `(y,z)` centroid and exists on
  both the inboard (x=0) and outboard (x=L) faces. Assign the cluster main/rot nodes
  that centroid and duplicate them per face (mirroring how panel nodes are split
  inboard/outboard); then `_build_node_pairs` pairs them automatically (no NaN) and
  `_build_unit_mode_matrix` gives them the correct macro-mode displacement. Physically
  correct: the junction frame translates/rotates consistently with the imposed mode.
  Most invasive — touches `assign_global_nodes_cluster_basis` (assign coords + per-face
  duplication), `collect_global_node_meta` (populate `yz`/`is_top`/`tangent` for cluster
  nodes), and the rot-node/gamma DOFs (clamp the non-physical ones, as already done in
  direct assembly).

  **Caveat — pairing collisions.** Once cluster nodes carry a real `(y,z)`, they enter
  `_build_node_pairs`, which builds a KD-tree and asserts a *unique* nearest match within
  `max dist < 1e-9` ([homogenisation.py:665-667](../engine/homogenisation.py#L665-L667)).
  A junction centroid will sit coincident with — or arbitrarily close to — the real panel
  endpoint nodes it constrains, so naive coordinate matching risks an ambiguous nearest
  neighbour or a count mismatch on the inboard/outboard assert. Note also there are
  **three** pairing strategies (`_build_node_pairs` by coordinate, `_build_node_pairs_topological`,
  and `_build_node_pairs_by_group` via `pair_group` ids at
  [homogenisation.py:708](../engine/homogenisation.py#L708)); the group-id path is likely
  the cleaner hook for (A) than coordinate matching, since it pairs by explicit id and
  sidesteps the centroid/endpoint collision entirely.

  **Decision (committed):** target the **`pair_group`** path. It pairs by explicit
  geometric id rather than coordinate proximity, so it (i) dodges the centroid/endpoint
  collision above and (ii) keeps pairing in the element-agnostic geometry layer. Assign
  each junction's inboard/outboard cluster nodes a shared `pair_group` id in
  `collect_global_node_meta` alongside their `(y,z)`/`is_top`; route the periodic solve
  through `compute_k6_periodic_bc(..., topological=True)` so it uses
  `_build_node_pairs_by_group`.

  **Decision (committed):** the cluster rot/gamma clamp added to the periodic path must be
  **`ndof`-parameterised** — reuse `build_cluster_basis_supports` semantics (clamp
  `BETA_S`, `BETA_X`, `*identity_extra_dofs` on the cluster refs), *not* a MITC4 hardcode.
  Note the periodic path currently applies **no** supports (`compute_k6_periodic_bc`
  clamps nothing), so this is a *new* support-application step in the periodic path, gated
  on `transformed_basis`; with `ndof` threaded through, HSDT (ndof=8) is covered with no
  extra code.

- **(B) Condense the virtual masters out before factorization — _fallback_.** Cluster
  reference nodes carry no element stiffness, so statically eliminate them (or directly
  prescribe their displacement from the macro mode at the junction `(y,z)`) inside the
  periodic solve so they never reach factorization as free DOFs. Less geometry churn,
  but special-cases the homogeniser and the rotation-DOF math is fiddlier. Use only if
  (A) proves messy.

### Step 2 — Implement (A) behind the existing mode switch

No change to `shared` behavior; the new path activates only for `transformed_basis`.
Guard the production default with the existing shared-mode regression suite so the
FSDT/MITC4 production routing is provably untouched.

### Step 3 — Validate, in increasing strength

1. **Consistency:** on a *near-collinear* junction, `transformed_basis` K6/K7 must match
   `shared` K6/K7 to round-off (as junction angle → 0 the two schemes converge). The
   cleanest correctness check.
2. **Equilibrium:** recovered section resultants (`∫Nx·ds`, moments, torsion) match the
   direct-assembly result validated in #3, within the few-% local-redistribution band.
3. **Element-agnosticism:** the cross-`ndof` identity check from Step 0 must still hold
   after the fix — same `pair_group`, cluster coords, and `T_pbc` structure for ndof=6
   and ndof=8. Proves the fix landed in the agnostic mesh layer, not in element-specific
   code. (Cheap; pairs with 3.1.)
4. **Headline (#3 deferred goal):** skin–web **shear-lag** — check traction continuity /
   stress transfer across the non-collinear corner under a transverse-shear resultant,
   vs the equilibrium / beam-theory expectation.

### Step 4 — Flip production non-collinear junctions to `transformed_basis`

Currently forced to `shared`. Document closure in
[project_hsdt_shell](../../../../.claude/projects/c--Users-s1834431-Projects-fastblade/memory/project_hsdt_shell.md).

### Follow-on (separate ticket) — HSDT periodic resultant recovery

The *MITC4* periodic recovery (`compute_resultants_from_modes`) is MITC4-bound
(`_NDOF_NODE`/`mitc4_*`, see the architectural-boundary table). The HSDT periodic path
does **not** reuse it — it already has its own element-correct recovery
`compute_resultants_from_modes_hsdt` (`element_library/hsdt/utilities/hsdt_homogenisation.py`, using
`hsdt_resultants` + `element_library/hsdt/utilities/hsdt_recovery.py` interlaminar), wired into
`run_section_periodic_bc_hsdt`. **So the "swap the recovery fn" item is already done.**

The genuinely open HSDT-recovery gap is different (and is *gap #2* from
[[project_hsdt_shell]]): a **single prismatic periodic cell cannot produce transverse
shear** (it needs a spanwise `dM/dx` gradient), so single-section HSDT interlaminar
`tau` ≈ 0 — physically meaningless for delamination. The fix is an **HSDT tapered
(two-station) path**, analogous to the existing `run_section_tapered_periodic_bc_mitc4`:
add `panels_outboard` support to `solve_global_coupled_hsdt_quad4` (mirror the MITC4
`L_s` averaging + outboard `collect_global_node_meta`) and a
`run_section_tapered_periodic_bc_hsdt` bridge fn using `compute_resultants_from_modes_hsdt`.
Track that as its own item; the single-section limitation is asserted by
`test_periodic_cluster_basis.py::test_hsdt_single_section_interlaminar_is_zero`.

### Suggested first slice

Step 0 (incl. the cross-`ndof` agnosticism assertion) + Step 1(A) + Step 3.1
(near-collinear equivalence) is the high-value core (~1 day) and de-risks the shared
production code. Steps 3.4 / 4 are the payoff and depend on the core landing cleanly.

## Is it worth doing now?

Production has run on `shared` junctions throughout; for collinear/near-collinear
geometry that is accurate. This matters only where there are genuinely sharp skin–web
corners *and* resolved junction shear-lag / delamination there is of interest (implied
by the delamination-constraint work in #4). If that is near-term, close it; otherwise it
is legitimately fine as documented debt.

## References

- Periodic solve: `engine/homogenisation.py` — `compute_k6_periodic_bc`,
  `_build_node_pairs`, `_build_unit_mode_matrix`.
- Node metadata: `engine/shell_mesh_map.py` — `collect_global_node_meta`,
  `cluster_points`, `cluster_is_collinear`.
- Cluster-basis MPCs (HSDT port): `engine/connected_assembly_core.py`
  (`assign_global_nodes_cluster_basis`, `build_cluster_basis_constraints`,
  `build_cluster_basis_supports`); `element_library/hsdt/assembly.py`.
- Production MITC4 path: `api/periodic.py::run_section_periodic_bc_mitc4`.
