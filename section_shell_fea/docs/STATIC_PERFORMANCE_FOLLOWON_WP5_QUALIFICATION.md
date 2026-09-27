# Static performance follow-on WP5 — qualification

**Date:** 2026-09-27  
**Programme:** Static processing performance follow-ons (post `7f537a6a`).

## Exactness contract (retained)

- No constitutive, constraint, recovery, FI, or solve-quality policy changes.
- No tolerance widening, diagnostic suppression, mesh coarsening, or tier omission.
- Cache keys refuse incomplete identities; changed `K` always rebuilds `K_reduced` / SuperLU.
- Governing provenance, trust flags, and diagnostic availability unchanged.

## D5.1 — Focused parity suites

| Suite | Result |
|-------|--------|
| `test_static_performance_followon_pins` | green (factor identity, solve_many edge, primary bitwise, shear coeffs, interface-shear batch≡scalar, cache refusal) |
| `test_static_processing_performance_pins` | green |
| `test_solve_quality` | green |
| `test_cap_tie_constraints` / `test_constraint_composer` / `test_runner_four_tier_pipeline` | green |
| `test_evaluation_caches` (BSP) | green |

## D5.2 — Full quality gates

| Gate | Result |
|------|--------|
| `section_shell_fea/tests` | **463 passed**, 1 failed |
| Pre-existing failure | `test_shell_midline_strip_contract::test_solver_and_strip_mesh_agree_on_panel_element_counts` — CCX `build_cap_skin_tie_surfaces` missing `shell_midlines` (FEA-CT-001). Unrelated to WP1–WP4; same failure class as prior refactor closeout. |
| New xfails / tolerance changes / skipped checks | **none** |

## D5.3 — Performance evidence

### Operation-level (retained)

| Family | Change | Exactness | Gain |
|--------|--------|-----------|------|
| Solution | Fail-closed SuperLU identity; `solve_many` keeps first `d` | pin suite | Correctness + observability |
| Production timing | `compose_constraints` + constrain/reduce/solve/reconstruct on `execute()` / `solve_constrained_system` | DiagnosticsBin schema preserved | Complete phase coverage |
| Constraint/reduction | Exact-key `ConstraintReductionCache` (no `K_rr`/LU) | fresh≡cached constraints/`T` under same identity | Cross-trial reuse when topology/modes match |
| Primary | Single `K@d` for `r_full` / residual / reactions | **bitwise** equal to triple matvec | −2 sparse matvecs / cascade |
| Tertiary | Batched ply E→M recovery; panel-static shear coeffs; vectorized canonical stress pack | existing tolerances / shape / provenance | Shear coeff reuse warm median **3.98 ms → 0.23 ms** (200 Q-pairs) |
| Quaternary | Panel-batched interface-shear screening (scalar oracle retained) | batch≡scalar values + provenance | Shared laminate-static strength table (~11% on 400-row microbench) |
| Cap-tie | Exact-key geometry cache (`CapTieGeometryCache`) | FEA-CT-001 / SG-017 unchanged | Skip rebuild when mesh/bonds identical |

Microbench driver: `examples/bench_static_performance_followon.py`.

### 16-station 2D-F `--no-plots` integration pin

Command (from repo root):

```text
python src/blade_structure_precompute/examples/section_selection/e2e_section_selection.py \
  --no-plots --rotors new_3.2m_rotor --layout 2D-F
```

Log path for this closeout attempt: `docs/runs/e2e_section_selection/wp5_followon_2D-F_no_plots.log`.

**Status at closeout:** pin **launched and in flight** (pass-1 station-level FE for all 16 stations after upstream Scale/IF/Feasibility/Stiffness). Append warm `stage_timings_s.section_selection`, acceptance, digests, FE calls, cache counters, and governing locations/modes to this file when the job finishes. Do not treat a contested/partial run as a performance oracle.

Compare against WP0 freeze and prior `docs/runs/e2e_section_selection/wp_s5_2D-F_run*.log` when complete.

## D5.4 — Delivery slices (Conventional Commits)

1. `fix(section-shell-fea): fail-closed SuperLU identity and production phase timing`
2. `feat(section-selection): exact-key constraint/reduction cache across trials`
3. `perf(section-shell-fea): single primary matvec and tertiary panel-static recovery`
4. `perf(section-shell-fea): batch interface-shear screening and cache cap-tie geometry`
5. `docs(section-shell-fea): WP0/WP5 static performance follow-on qualification`

## Rejected optimizations

| Idea | Why rejected |
|------|----------------|
| Auto-assign current `K` identity to injected SuperLU | Silent stale-factor reuse (WP1 hole) |
| Cache `K_reduced` / SuperLU under topology-only keys | Gauge/ABD changes would reuse wrong factors |
| Widen `rtol`/`atol` or suppress residual checks for parity | Violates exactness contract |
| Drop quaternary interface provenance fields | Governing/diagnostic drift |

## Exit gate

Focused exactness pins green; production phases observable; retained opts have operation-level evidence; full suite green aside from the documented pre-existing CCX midline failure; 2D-F exclusive pin queued for warm timing capture.
