# Numerical stability and mesh independence (GCI)

Status for the section FE worker package (`section_shell_fea`): primary solve
quality, ASME V&V 20 / Roache fine-grid GCI campaigns, and **root-cause-first**
follow-up for quantities that are not yet mesh-independent.

**Related:** [SHELL_MODEL.md](SHELL_MODEL.md) (cascade / GCI harness overview),
[`coordinate_frames/README.md`](../../../geometry_utilities/coordinate_frames/README.md)
(E-frame), [`cap_peak_sampling.py`](../processing/static/diagnostics/cap_peak_sampling.py)
(peak localization tags).

---

## 1. Purpose

Two different questions are often conflated:

| Question | Metric | Meaning |
|----------|--------|---------|
| **Numerical stability** | Reduced solve residual, `cascade_trust_ok`, mesh QA | Did this rung solve correctly? |
| **Mesh independence** | Last-two drift, fine-grid GCI on kind peaks / `governing_fi` | Do scalars stop changing under refinement? |

A trusted solve on a coarse mesh can still fail GCI. Conversely, large
*unreduced* residuals on tied layouts do **not** mean the reduced SuperLU solve
failed (see §4).

Campaign context: 40 `SYSTEM_TYPE_KEYS`, Cotide `new_3.2m_rotor` + `NACA_63-415.dat`,
five-rung ladder `frac ∈ {0.024, 0.016, 0.012, 0.008, 0.005}`, MITC4 production
cascade vs CalculiX membrane oracle.

---

## 2. E-frame contract (1–2–3)

Mid-surface resultants are stored in element / laminate frame **E**
(\(x_1, x_2, x_3\)) on [`ShellPanelResultants`](../pre_processing/containers/recovery_types.py):

| Component | Meaning |
|-----------|---------|
| \(N_{11}, M_{11}\) | strip / laminate longitudinal |
| \(N_{22}, M_{22}\) | contour tangent |
| \(N_{12}, M_{12}\) | in-plane shear |
| \(Q_1, Q_2\) | FSDT transverse shear |
| \(\tau_{13}, \tau_{23}\) | interlaminar (tertiary) |

GCI peak columns use the same indices: `peak_abs_n11`, `peak_abs_n22`,
`peak_abs_n12`, `peak_abs_m11`, … (see
[`cascade_mesh_convergence.py`](../processing/static/diagnostics/cascade_mesh_convergence.py)).

**Do not** introduce new public `Nx` / `Nxy` resultant names. Legacy study CSVs
(if present) map as:

`Nx→N11`, `Ny→N22`, `Nxy→N12`, `Mx→M11`, `My→M22`, `Mxy→M12`, `Qx→Q1`, `Qy→Q2`.

---

## 3. Harness

| Piece | Location |
|-------|----------|
| Ladder extract (M+P→S→T→Q) | `processing/static/diagnostics/cascade_mesh_convergence.py` |
| Multi-key GCI study | `processing/static/diagnostics/grid_independence_study.py` |
| Roache GCI | `processing/static/diagnostics/asme_gci.py` |
| Peak tags (junction / tie / endpoint) | `processing/static/diagnostics/cap_peak_sampling.py` |
| MITC4 CLI | `benchmarks/run_grid_independence_study.py` |
| CCX CLI | `benchmarks/run_ccx_grid_independence_study.py` |
| Outputs | `visualisation/output/gci_study_mitc4/`, `gci_study_ccx/` |

Default study fracs: `STUDY_REFINEMENTS = (0.024, 0.016, 0.012, 0.008, 0.005, 0.0035)`.

Kind-peak registry covers secondary membrane/bending/shear, tertiary mid-σ /
`|τ13|`/`|τ23|`, and quaternary Hashin in-plane / interlaminar, each with raw and
endpoint-excluded interior series.

---

## 4. Primary stability findings (completed campaign)

On the fine rung (`frac=0.005`), MITC4:

| Check | Result |
|-------|--------|
| `solve_residual_rel` (reduced) | median ~\(4\times10^{-12}\); max ~\(2\times10^{-10}\); **0/40** above \(10^{-6}\) |
| `cascade_trust_ok` / `mesh_ok` | **40/40** layouts, all rungs |
| `‖d‖_∞` last-two | median ~1.004×; p90 ~1.43× |
| `residual_rel_unreduced` | often O(1)–O(10) on tied caps |

**Interpretation:** trust and CI must use the **reduced** residual. Unreduced
`‖f−Kd‖/‖f‖` includes MPC / support reaction imbalance on the full DOF basis and
is not a failed solve indicator (same note in [SHELL_MODEL.md](SHELL_MODEL.md)).

CCX rungs do not populate primary residual columns; trust is coverage / parse based.

---

## 5. Convergence scorecard (40-key Cotide, E-frame names)

Fine-grid GCI ≤ 10% on skin/web (unless noted). Figures from the **post
FI-ownership + Defect-K composition** MITC4 refresh (`STUDY_REFINEMENTS` incl.
`0.0035`, production `shared` + `skin_offset`, continuum-owned Hashin/gov).

**Machine + human scorecard:** after any study, or offline from an existing
`gci_summary.csv`:

```powershell
python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.run_stability_scorecard --study-dir visualisation/output/gci_study_mitc4
```

Writes `stability_scorecard.json` + `stability_scorecard.md`. Fails are split into
**magnitude** (|GCI| > limit) vs **Roache applicability** (`gci_non_monotonic`
with |GCI| ≤ limit). Day-to-day and package confidence use magnitude counts.

### Family confidence bands (MITC4 shared, magnitude lens)

| Band | Families / keys | CI membrane (N12+mid-σ ≤10%) | Notes |
|------|-----------------|------------------------------|-------|
| **A** | A, B, C (non-watch) | Pass on production path | Smoke CI covers representatives |
| **B** | D non-watch (`2D-*`, `3D-F`, …) | Pass CI magnitude | Study Hashin/gov may still watch some keys |
| **C** | Cap kind peaks | Informational | Defect K / `skin_offset` |
| **D-watch** | `CI_WATCH_KEYS` | **Known residual** | Not a regression; see §6 |

### MITC4

| Tier | Quantity | Status | Evidence (skin unless noted) |
|------|----------|--------|------------------------------|
| S | `\|N12\|` | **CI freeze (watch)** | ≤10%: **35/40** (was 39); trust/mesh 40/40 — watch list §6 |
| S | `\|N11\|`, `\|N22\|` | **Mostly converged / study** | ≤10%: **32/40** skin/web — not CI; residual C/D keys secondary to `\|N12\|` watch |
| S | `\|M11\|`, `\|M22\|` | **Study-reported (deferred)** | ≤10%: **23/40** skin (`\|M11\|`) — no consumer CI; see §7 |
| S | `\|M12\|`, `\|Q1\|` | **Report-only (true null)** | campaign load; see §9 |
| S | `\|Q2\|` | **Study-reported (deferred)** | ≤10%: **28/40** skin — not CI until a consumer requires it |
| T | mid-σ | **CI freeze (watch)** | ≤10%: **35/40** (was 39) — same watch keys as `\|N12\|` (§6) |
| T | `\|τ13\|` | Report-only | tracks null `\|Q1\|` |
| T | `\|τ23\|` | Study-reported | tracks `\|Q2\|` |
| Q | Hashin in-plane | **Study gate** | ≤10%: **34/40** (was 32); not CI |
| Q | Hashin interlaminar | **Weak / study** | study CSV |
| Q | `governing_fi` | **Study gate** | ≤10%: **34/40** (was 28); not CI |
| Cap | kind peaks | Informational | `skin_offset` Defect K / tie policy |

Phase B2 **kept `shared`** after dual-mode outlier study (see
[GCI_OUTLIER_DIAGNOSIS.md](GCI_OUTLIER_DIAGNOSIS.md) / SHELL_MODEL Phase B2).
`transformed_basis` + `skin_offset` is runnable but not production default.
Study-only `hybrid_kind_crossing` is the generalised joint spike (Phase B);
promote only after scorecard CI membrane **40/40 magnitude** and the Phase B2
Hashin/gov bar.

### CCX (membrane)

Housekeeping refresh (`--keys all --gates ccx`, shared + `skin_offset`):
`visualisation/output/gci_study_ccx/`. Trust/mesh **40/40**.

| Quantity | Status |
|----------|--------|
| `\|N12\|` | **Oracle-ready** — ≤10%: **39/40** (was 38); med GCI ~0.30%; last-two all pass. Sole fine-GCI fail: `4D-F` |
| `\|N11\|`, `\|N22\|` | **Report-only** — SFV load-channel / peak metric limitation; do not chase with mesh |

Formal `gci_ok` often fails with `gci_non_monotonic` even when |GCI| is already
small (especially CCX `|N12|`). Treat that as Roache **applicability** only when
|GCI| and last-two are already acceptable—not as a free pass when drift is large.

---

## 6. Outlier layouts / CI watch list

Root-cause notes and before/after GCI: [GCI_OUTLIER_DIAGNOSIS.md](GCI_OUTLIER_DIAGNOSIS.md).

### Current CI watch (post-ownership 40-key, fine GCI > 10%)

Frozen in code as `CI_WATCH_KEYS` / `STUDY_FI_WATCH_KEYS` in
`grid_independence_study.py` (`--keys watch` resolves to the CI set).

CI still gates raw skin/web `peak_abs_n12` + `peak_mid_sig` (`CI_KIND_METRICS`).
Keys are **not** dropped from CI and gates are **not** switched to interior peaks.
A full `--keys all --gates ci` run is **expected to fail exactly
`CI_WATCH_KEYS`** (magnitude) until Phase B clears them — **known residual / not
a regression**. Day-to-day CI uses `--keys smoke` (disjoint from the watch set).

| Field | Layouts failing ≤10% fine GCI |
|-------|-------------------------------|
| `\|N12\|` / mid-σ (CI) | `3D-CN`, `4D-CN`, `4D-F`, `5D-CN`, `5D-F` (`CI_WATCH_KEYS`) |
| `governing_fi` (study only) | `2D-CN`, `3B-CN`, `3D-CN`, `4D-CN`, `5D-CN`, `5D-F` (`STUDY_FI_WATCH_KEYS`) |

### CI watch ownership (cap-peak diagnosis)

`cap_peak_sampling` on the five CI keys
(`visualisation/output/cap_peak_sampling_ci_watch/`): skin `|N12|` / mid-σ peaks
are **junction + endpoint** on finest rungs for `3D-CN`, `4D-CN`, `4D-F`,
`5D-CN` (raw/interior ~1.5–1.65× on CN keys; `4D-F` ~1.09×). `5D-F` migrates
off the junction on the finest rung but still fails fine GCI. Full tag table:
[GCI_OUTLIER_DIAGNOSIS.md](GCI_OUTLIER_DIAGNOSIS.md) § “CI watch ownership”.

Geometry + CI ladder composites for the watch set (by structural family):
`visualisation/output/gci_study_mitc4/plots/ci_watch_layouts/{A|B|C|D}/{layout}_convergence.png`
(full 40-key gallery: `plots/layouts/{A|B|C|D}/`). Spike evidence dirs:
`gci_study_mitc4_junction_band/`, `gci_study_mitc4_tb_ci_watch/` (same plot
layout). Regen without re-solve:

```powershell
python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.plot_gci_study_outputs --study-dir visualisation/output/gci_study_mitc4 --keys 3D-CN,4D-CN,4D-F,5D-CN,5D-F --out-subdir plots/ci_watch_layouts --layouts-only
```

### Historical clusters (pre-ownership campaign notes)

| Cluster | Typical fails |
|---------|----------------|
| **4D-F**, **5D-F** | Skin `\|N12\|` / Hashin / mid-σ late jump on finest rungs |
| **0A**, **4D-CN**, **3B-F**, **3B-CN** | Hashin / `governing_fi` high fine GCI |
| **2D-F**, **2C-F**, **1C-F**, **2B-*** | Hashin-IL / `\|M12\|` / shear noise |

Diagnostic procedure (do **not** “fix” by gating these away):

1. Tag the peak element (`junction` / `tie_slave` / `endpoint_band`) per rung.
2. Compare raw vs interior series — flat interior + jumping raw ⇒ interface localization.
3. Check whether peak panel/station index migrates with `frac` (knot-merge / uniform strip).

---

## 7. Root-cause-first forward work

**Principle:** non-convergence is a defect or unresolved singularity until proven
otherwise. Success means continuum-owned peaks improve under the same study GCI
metrics—not switching acceptance to interior-only.

### Done (this campaign)

1. E-frame naming; junction-local \(h\); `n_cap_samples=120`; recovery audit (§9).
2. Full 40-key refresh; Cluster A membrane late jumps fixed on raw `|N12|`.
3. CI freeze on proven `peak_abs_n12` + `peak_mid_sig` (see §8).
4. FI sampling ownership: Hashin / `governing_fi` exclude **kind-crossing**
   junction nodes and cap-tie-slave elements (Defect K / `skin_offset`
   continuum ownership; same-kind strip interfaces stay eligible).
5. `skin_offset` × `transformed_basis` composition (cluster MPCs + master
   expansion); Phase B2 **keep `shared`** after outlier dual-mode study
   (Hashin/gov win on 4/6 keys only — see GCI_OUTLIER_DIAGNOSIS).

### Remaining / locked decisions

1. ~~CI watch `cap_peak_sampling`~~ — done: four of five keys junction-owned on
   finest skin peaks; `5D-F` migrates (see §6).
2. ~~Junction-band densify spike (`n_band=6`, half-`h`)~~ — **failed** CI success
   bar (see GCI_OUTLIER_DIAGNOSIS). Production stays
   `JUNCTION_BAND_ELEMENTS=3` / `JUNCTION_BAND_STEP_OVER_H=0.5`. Study overrides:
   `--junction-band-elements` / `--junction-band-step-over-h`.
3. ~~TB CI-watch dual-run (measurement)~~ — clears raw `|N12|`/mid-σ ≤10% on
   4/5 keys; residual **`5D-CN`**. Soft cusp / relative-slip penalties stay
   deferred.
4. ~~`5D-CN` precursor + `hybrid_collinear` spike~~ — TB relocates peak
   `SS:bay_0 j-e` → `PS:bay_3 --e` and amplifies; hybrid study mode trust OK
   but skin GCI **17.9% → 29.1%** (fail).
5. ~~Bay-foot densify (`SS:bay_0` start, n=6)~~ — cluster is non-collinear
   3-way `SS:TE`+`SS:bay_0`+`web_4`; label/start-scoped band leaves skin GCI
   **~17.9% flat** and peak still `j-e`. **Keep `shared` / band=3**; do not
   fold. Mesh pins at that foot alone do not clear MITC4 membrane CI.
6. **Phase B joint spike (`hybrid_kind_crossing`):** study-only TB on
   kind-crossing ∩ non-collinear; caps `skin_offset`. Watch scorecard
   (`gci_study_mitc4_hkc/`) → membrane magnitude **1/5** (`3D-CN` only);
   several keys regress vs shared. **Entry criteria not met** → keep `shared`,
   mode stays study-only, no densify, no default flip. Next candidate must clear
   watch then 40/40 magnitude and meet the Phase B2 Hashin/gov bar
   (equal-or-better on residual set incl. `4D-F`/`2D-F`).
7. `|M11|/|M22|/|Q2|` stay **study-reported, not CI** — no consumer has required
   bending/shear CI; when one does, treat as junction bending ownership first.
8. ~~CCX `|N12|` 40-key oracle refresh~~ — **39/40** (sole fail `4D-F`).

### Do not (masking)

- Widen `ENDPOINT_BAND` or switch production/study **acceptance** to interior /
  interface-excluded peaks while continuum peaks still jump.
- Treat large-|GCI| `gci_non_monotonic` as a Roache quirk and move on.
- Chase CCX peak `|N11|`/`|N22|` with mesh alone (load-channel / metric definition).
- Re-open `|M12|/|Q1|/|τ13|` GCI without a load case that excites those channels.

Interior / tag-excluded series remain **diagnostics**, not a substitute
acceptance criterion.

---

## 8. Policy (gates and oracle)

| Surface | Policy |
|---------|--------|
| MITC4 CI | **Frozen:** raw skin/web `peak_abs_n12` + `peak_mid_sig` drift + fine GCI ≤10% + mesh/trust. Cap informational. Hashin / `governing_fi` **not** CI. Full 40-key CI expects the five §6 watch keys to fail until a later campaign; use smoke/subset for day-to-day |
| MITC4 study | Stricter drift/GCI on raw skin/web N12 / mid-σ / Hashin + `governing_fi`; cap **interior** last-two reported (Defect K) |
| CCX | **`peak_abs_n12` only** (`gates="ccx"`); `|N11|`/`|N22|` report-only |
| Noise / deferred | `|M12|`, `|Q1|`, `|τ13|` report-only (true null); `|M11|/|M22|/|Q2|` study-reported not CI |

### Runbook (package surface)

```powershell
# 1) Day-to-day CI smoke (must PASS; excludes CI_WATCH_KEYS)
python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.run_grid_independence_study --keys smoke --gates ci

# 2) Full CI (expected to fail exactly CI_WATCH_KEYS until Phase B)
python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.run_grid_independence_study --keys all --gates ci

# 3) Study gates + scorecard (auto-written under --out-dir)
python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.run_grid_independence_study --keys all --gates study
python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.run_stability_scorecard --study-dir visualisation/output/gci_study_mitc4

# 4) Watch-only / Phase B spike
python -m ...run_grid_independence_study --keys watch --gates ci --interface hybrid_kind_crossing --out-dir visualisation/output/gci_study_mitc4_hkc

# CCX membrane oracle
python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.run_ccx_grid_independence_study --keys all --gates ccx
```

**Phase B entry criteria:** scorecard CI membrane **40/40 magnitude** (all
`CI_WATCH_KEYS` cleared) before any promote discussion.

---

## 9. Recovery audit — `|M12|`, `|Q1|`, `|τ13|`

Completed-study ladder magnitudes (layout `0A`, skin, fine rungs) show:

| Field | Order of magnitude | Verdict |
|-------|--------------------|---------|
| `|N12|` | \(5\times10^4\) N/m | physical |
| `|Q2|` / `|τ23|` | \(10^3\) N/m / \(10^6\) Pa | physical |
| `|M12|` | \(10^{-12}\)–\(10^{-11}\) N·m/m | **true null** (machine noise) |
| `|Q1|` | \(10^{-9}\)–\(10^{-7}\) N/m | **true null** under campaign load |
| `|τ13|` | \(10^{-5}\)–\(10^{-4}\) Pa | tracks `|Q1|` (null) |

Secondary FSDT recovery populates `Q1`/`M12` with trusted provenance when
thickness / \(G_\mathrm{eff}\) exist; values are not NaN-filled zeros. Tertiary
`τ13` follows the null `|Q1|` channel under the Cotide study load
(`T`, `Vy`, `Mz`). **Conclusion:** report-only is justified by physics of the
load case, not a silent recovery bug. Re-open if a load case with non-zero
`Q1`/`M12` kinematics appears.

## 10. Mesh / junction / FI ownership remediation (applied)

| Change | Location |
|--------|----------|
| Junction-band `required_s` (`JUNCTION_BAND_ELEMENTS=3`, half-`h`; study overrides OK) | `section_mesh/strip/strip_mesh.py` |
| Skin/web/cap `KnotConstrainedUniform` so pins are honoured | same |
| D-family `n_cap_samples=120` | `cascade_mesh_convergence._build_shell` |
| Study ladder densified with `0.0035` | `STUDY_REFINEMENTS` |
| FI sampling ownership (skip kind-crossing junction + tie-slave elems for Hashin/gov) | `cascade_mesh_convergence._fi_sampling_excluded` |
| Cluster MPCs attached for `transformed_basis` / `shared_rotated` | `section_assembly.assemble_section_from_zones` |
| Cap-tie master expansion through junction slaves | `cap_tie_constraints._expand_master_terms` |
| CI freeze on N12 + mid-σ | `grid_independence_study.CI_KIND_METRICS` |

Membrane / mid-σ peaks remain unfiltered. Interior series stay diagnostic.
`|M11|/|M22|/|Q2|` stay study-reported (not CI) until a consumer requires them.

## 11. Change log

| Date | Note |
|------|------|
| 2026-08 | Initial doc from 40-key MITC4/CCX campaign + root-cause-first plan (E-frame naming, anti-masking). |
| 2026-08 | Junction-local h + recovery audit + E-frame cleanup; finer ladder `0.0035`. |
| 2026-08 | Post-refresh scorecard; CI freeze N12/mid-σ; FI junction/tie ownership; M11/Q2 deferred. |
| 2026-08 | Defect K: skin_offset×transformed_basis composition; Phase B2 keep shared (4/6 Hashin/gov). |
| 2026-08 | Post-ownership 40-key refresh (shared): N12/mid 35/40; Hashin/gov 34/40; M11 23/40. |
| 2026-08 | Housekeeping: CI watch list (§6) for five D-family N12/mid fails; CCX N12 oracle refresh → **39/40** (fail `4D-F`). |
| 2026-08 | Per-layout geometry + CI ladder PNGs (`plots/layouts/`, `plots/ci_watch_layouts/`); offline `plot_gci_study_outputs`. |
| 2026-08 | CI-watch cap-peak diagnosis: junction-owned skin N12/mid on 3D-CN/4D-CN/4D-F/5D-CN; keep shared; bending stays deferred. |
| 2026-08 | Junction-band spike `n_band=6` failed CI bar (regress `5D-F`); TB CI-watch clears 4/5 membrane keys, residual `5D-CN`; keep shared / band=3. |
| 2026-08 | 5D-CN precursor (TB relocates peak); `hybrid_collinear` study mode regresses skin GCI 17.9→29.1%; keep shared; no Phase B2 promote. |
| 2026-08 | Bay-foot densify on `SS:bay_0` start (n=6) flat at ~17.9% skin GCI; 3-way non-collinear cusp; keep shared/band=3. |
| 2026-08 | Stability package: `CI_WATCH_KEYS` + `--keys watch`, scorecard JSON/MD (magnitude vs Roache), smoke CI contract, family confidence bands; study-only `hybrid_kind_crossing`. |
| 2026-08 | `hybrid_kind_crossing` CI-watch scorecard 1/5 magnitude (`3D-CN` cleared); keep shared; no Phase B promote. |
| 2026-08 | Residual-4 deep-dive (HKC peak migration vs Defect K); `shared_cluster_frame` watch scorecard 0/5 — keep shared. |
