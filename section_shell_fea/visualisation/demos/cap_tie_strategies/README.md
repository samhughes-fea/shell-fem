# Spar-cap tie strategies (BL-003)

Spar caps were **structurally disconnected** in the shell FE. Junction clustering
keys on `panel_endpoints` (first/last node only); a cap's endpoints sit
mid-skin-segment and its web crossings are interior to both strips, so no cluster
forms. Confirmed at assembly level: **0** cap global node ids were shared with any
other panel (5D-F: 230 cap nodes, 2D-CN: 120 — both zero).

Consequence: cap in-plane shear diverged under mesh refinement while skin and web
converged, and cap `fi_spar` was never physically meaningful. EI was unaffected —
caps enter through the geometry property integration, a separate path.

**Status: fixed.** Strategy S2 is implemented in the assembler
([cap_tie_constraints.py](../../processing/static/operations/cap_tie_constraints.py))
and reachable as `SectionLinearStaticRunner(cap_tie_mode="skin_offset")`. It is
**off by default** (`"none"`), so nothing changes until a caller opts in.

## Files

- `strategies.py` — four candidate tie rules as pure geometry functions, each
  returning a `TieProposal`. Decision aid only; the assembler does not use them.
- `compare_cap_tie_strategies.py` — coverage sweep across every catalogue layout
  with spar caps × root/mid/tip sections × thin/nominal/thick thicknesses.
- `cap_tie_convergence.py` — the FE test that coverage could not do: refine the
  mesh with ties off vs on and watch cap `|N12|`.

## Coverage results (261 cases)

| strategy | fail % | ties/case | offset mm (min/med/max) |
|---|---|---|---|
| S1 cap↔web crossings | **56.3 %** | 2.6 | 0.000 / 0.000 / 0.001 |
| S2 cap↔skin offset | 0.0 % | 56.8 | 0.000 / 2.745 / 7.504 |
| S3 cap endpoints only | 0.0 % | 6.8 | 0.000 / 0.000 / 7.463 |
| S4 merge into skin laminate | 0.0 % | 0 (topology change) | — |

A *fail* means at least one cap was left with no load path at all.

**S1 is disqualified as a standalone rule.** Caps in the entire B family
(`0B`, `1B-*`, `2B-*`, `3B-*`, …) never cross a web, so more than half the
catalogue would keep floating caps — exactly the layout-dependence the
robustness requirement exists to catch.

### Correction: S2 does not fail 10.7 %

An earlier run scored S2 at 10.7 % failures on `3D-*` / `4C-*` / `4D-*` and
guessed the `tol_frac` outlier gate was too tight. Half right — the gate was the
problem, but not because it needed loosening. It rejected any node whose
cap→skin separation differed from `(t_skin + t_cap)/2` by more than 60 %, and
that prediction was wrong for **two independent reasons**.

**1. The cap midline was on the wrong locus.** `_airfoil_offset_arc` placed the
cap mid-surface at `t_skin + t_cap/2` measured **vertically** from the outer
surface, while the cap *solid* — `Slab(Offset(poly, -t_skin), cap_height)` — and
the skin midline are both true **normal** offsets. A vertical drop of `d` lands
at normal depth `d·cosθ`, so the midline ran shallow wherever the surface is
steep, in the worst measured case by **86 %**, putting it inside the skin band
rather than in the cap. **This is now fixed** (see
`test_spar_cap_midline_offset.py`); max depth error across 784 cap instances
went from 0.859 to 6.7e-5, and the measured normal gap is now
`(t_skin + t_cap)/2` to within 1e-3.

**2. Requested thicknesses are not the thicknesses used.** The section builder
clamps and rescales `skin_thickness` / `cap_height` to fit the local cavity — for
example a requested `skin_thickness=0.001` becoming `0.0025`. Any prediction built from
the *caller's* requested values is wrong by whatever the clamp did, no matter how
good the geometry is.

The earlier "ratio spans 0.00–2.42" figure quoted here conflated the two: it
compared a measured separation against a prediction using the requested skin
thickness and the actual cap thickness. Reason 1 was real and is fixed; the
specific spread was inflated by reason 2. Measured against the thicknesses
actually used, the normal gap is `(t_skin + t_cap)/2` at p5, p50 and p95 alike.

Measuring the offset off the mesh removes the gate, is immune to both problems,
and takes S2 to 0 % coverage failure. That is what `strategies.py` and the
assembler both do.

**One genuine zero-separation case remains.** About 7.6 % of cap nodes sit
exactly on the skin polyline, entirely within the C family (`1C`–`5C`) at roughly
8 nodes per cap regardless of cap size — the discrete-cap endpoints, which lie on
the skin by construction. That is the original "cap endpoints sit mid-skin-segment"
topology, not an offset error, and the tie handles a zero-length link correctly.

## FE result: the divergence is fixed

`cap_tie_convergence.py`, 5D-F at root, torsion-dominated load, peak `|N12|` over
five refinement levels (332 → 1268 nodes). **`thinwall` is the production channel
and the only one that applies load to the caps at all** — `equipollent` drops
every strip whose kind is not skin/web, so under it a cap is moved purely by
compatibility with the skin:

| channel | cap_tie_mode | skin drift | web drift | cap drift |
|---|---|---|---|---|
| `thinwall` | `none` | 1.12× | 1.09× | **30.65×** |
| `thinwall` | `skin_offset` | 1.14× | 1.17× | **1.13×** |
| `equipollent` | `none` | 1.03× | 1.03× | **20.51×** |
| `equipollent` | `skin_offset` | 1.08× | 1.07× | **1.04×** |

Untied, cap shear peaks at 4.6e8 against 6.7e4 for skin — four orders of
magnitude out, wandering non-monotonically. That is the signature of a component
that is not structurally connected at all: its in-plane DOFs sit in the solver's
null space, so the number is conditioning noise rather than a stress. It is worse
on `thinwall` (30.65×) precisely because that channel does apply load to the
disconnected cap.

Tied, cap drift falls to 1.13× — the same scale as skin (1.14×) and web (1.17×),
which are the control: they are connected either way and set what "converged"
means here.

Skin and web peaks also rise once ties are on (6.7e4 → 8.5e4, 4.8e4 → 6.8e4 on
thinwall). That is the point, not a side effect: with the cap participating, load
redistributes, so the untied model was mis-reporting skin and web too.

## FE vs load channel: they disagree about the cap

`cap_fe_vs_load_channel.py`, 176 caps across the cap-bearing catalogue at root and
mid. The comparison is a **dimensionless share**, not a magnitude:

- **load channel** — `stacked_cap_gt_share`, the cap's `Gt` fraction of the
  stacked cap+skin wall. With the default `G_by_kind` (caps and skin share a `G`)
  this is a pure thickness ratio.
- **tied FE** — at each cap node, `|Nxy_cap| / (|Nxy_cap| + |Nxy_skin|)` against
  the skin point directly beneath, median over the cap.

| | p5 | p50 | p95 |
|---|---|---|---|
| tied FE | 0.735 | **0.962** | 0.982 |
| load channel | 0.453 | **0.667** | 0.667 |
| FE − channel | +0.068 | **+0.295** | +0.528 |

**The bonded FE gives the cap substantially more shear than the `Gt` split does,
and the sign never flips** (p5 of the difference is +0.068). So the stacked-wall
split is not what a bonded cap actually does. S2 and S4 are therefore *not*
interchangeable: S4 would change how the spar is sized, not merely how it is
reported.

Two honesty notes on this measurement:

- **A raw `N12`-vs-`q` comparison does not work**, and the skin/web control is
  what proves it. The channel's `q` is applied flow while the FE's `N12` is
  developed resultant; comparing peaks gave skin 0.08, web 0.08 and cap 0.15 —
  all offset by the same order, i.e. measuring the boundary conditions rather
  than the cap model. Only the share comparison is apples-to-apples.
- **The FE share saturates near 0.98** on several layouts, so this establishes
  direction and rough magnitude, not a calibrated number. Pinning it down wants a
  shear-only load case and element-level rather than nodal `N12`.

## Sizing delta: untied caps were sizing the blade on a fiction

`cap_tie_sizing_delta.py` runs the real station sizing loop twice per case,
`cap_tie_mode` `"none"` then `"skin_offset"`, over 16 cases (8 layouts × 2
stations) with the FI gate on at `allow=1.0`:

| quantity | on/off ratio p5 | p50 | p95 |
|---|---|---|---|
| `max_fi_hashin` | 0.006 | **0.030** | 0.883 |
| `fi_spar` | 1.959 | **5.698** | 269.1 |
| `fi_skin` | 0.002 | 0.020 | 0.883 |
| `t_skin` | 0.119 | 1.000 | 1.000 |

- **`pass_flag` changed in 6/16 cases**, every one `False → True`. Untied, those
  stations reported `max_fi_hashin` of 9–17 against a gate of 1.0, so the loop
  grew skin to its bound and still soft-failed. Tied, FI lands at 0.06–0.4 and
  the station passes.
- **The governing DV flipped in 4/16 cases** (`2C-CN`, `5C-F`, both stations),
  from `t_skin` to `t_web`. That is the load-bearing result: ties change *which
  subcomponent the optimiser thickens*, not merely what it reports.
- `fi_spar` rises (median 5.7×) because the cap now carries real load, while
  `fi_skin` collapses (median 0.02×) because it is no longer absorbing the
  disconnected cap's garbage.

So the untied model was not merely under-reporting the spar — it was inflating
skin FI enough to drive the whole sizing decision. `cap_tie_mode` still defaults
to `"none"`; these numbers are what turning it on would move, and they are the
argument for doing so.

## Why a rigid link and not a shared node

A cap sits inboard of the skin, so the midlines are separated (2.1 mm on 5D-F
root). Collapsing them onto one node moves the cap onto the skin midline and
throws away the lever arm that makes a cap worth having. The tie is therefore an
offset rigid link — emitted as ordinary master/slave MPCs on the existing
constraint path, so no new solver machinery is involved. Link lengths came out
mesh-invariant across the sweep (2.077/2.125/2.150 mm at the coarsest,
2.081/2.129/2.153 mm at the finest), as a geometric property should.

The defining correctness property is that a rigid link reproduces rigid-body
motion exactly, or it injects strain energy with no physical origin. That is
tested directly in
[test_cap_tie_constraints.py](../../tests/test_cap_tie_constraints.py) over five
rigid-body modes, plus the degenerate zero-offset case where the link must
reduce to plain shared-node compatibility.

## What is still open

**S4 has not been ruled out, and it is not the same kind of choice.** S4 drops
the cap panel and thickens the skin laminate over its span, which is what the
*load* channel already assumes — `apply_stacked_cap_shear` sets
`t_shear = t_skin + t_cap` and hands the cap only `q × Gt_share`. S2 makes the
cap a real, separately reported component that carries load through a bonded
interface; S4 makes the FE agree with the load model at the cost of the cap
ceasing to be a separate panel, so per-component spar FI would have to come from
ply ranges within the stacked laminate. Note S4 cannot be ranked by the sweep
above — it removes the very panel the sweep measures. The FE/load-model
contradiction S4 would resolve is still there.

**Caps remain untied to webs.** S2 ties caps to skin only. Where a cap crosses a
web (the D family) that junction is still not shared; S1 showed those crossings
exist in under half the catalogue, so this is a refinement, not a gap in the load
path.

**Production default:** `SectionLinearStaticRunner` defaults to
`cap_tie_mode="skin_offset"`. Pass `"none"` only for the untied BL-003
discriminator. Turning ties on changes skin and web results as well as cap
results versus historical untied sizing, so designs sized against untied
numbers need revalidation. Ties remain incompatible with
`interface_constraint_mode="transformed_basis"`, where skin endpoint DOFs are
themselves constraint slaves; the builder raises rather than let the reduction
silently drop those masters.
