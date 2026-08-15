# GCI outlier root-cause notes

Diagnosis from the completed MITC4 ladder (legacy CSV names → E-frame:
`nxy→n12`, `mx→m11`, `qy→q2`) plus live `cap_peak_sampling` tags
under junction-local \(h\).

**Policy:** raw peaks remain the acceptance metric. Interior / interface-excluded
series are evidence only.

## Current CI / study watch (post-ownership 40-key refresh)

Source: `visualisation/output/gci_study_mitc4/gci_summary.csv` (shared +
`skin_offset`, continuum-owned Hashin/gov). Fine-grid GCI > 10%:

| Gate | Layouts |
|------|---------|
| CI: skin `\|N12\|` / mid-σ | `3D-CN`, `4D-CN`, `4D-F`, `5D-CN`, `5D-F` |
| Study: `governing_fi` | `2D-CN`, `3B-CN`, `3D-CN`, `4D-CN`, `5D-CN`, `5D-F` |

CI metrics unchanged (`peak_abs_n12`, `peak_mid_sig`). Full `--keys all --gates ci`
is expected to fail the five CI keys until a later joint/mesh campaign; smoke /
subset CI remains day-to-day. Sections below keep historical campaign notes.

**CCX `|N12|` oracle (housekeeping refresh):** ≤10% fine GCI **39/40**; trust/mesh
40/40; sole fail `4D-F`. See NUMERICAL_STABILITY §5 CCX row.

## CI watch `|N12|` / mid-σ ownership (`cap_peak_sampling_ci_watch`)

Live ladder: `visualisation/output/cap_peak_sampling_ci_watch/`
(`--keys 3D-CN,4D-CN,4D-F,5D-CN,5D-F`, study fracs incl. `0.0035`,
`shared` + `skin_offset`). Tag shorthand: `j` = kind-crossing junction,
`t` = tie_slave, `e` = endpoint_band, `-` = absent.

| Layout | Skin `\|N12\|` tags (coarse→fine) | Finest raw/interior | Verdict |
|--------|-----------------------------------|---------------------|---------|
| **3D-CN** | `---` … then `j-e` `j-e` | ~1.52× | Junction-owned late; interior last-two flat (~1.00) while raw last-two ~1.09 |
| **4D-CN** | `---` … then `j-e` `j-e` | ~1.65× | Same pattern; raw last-two ~1.20, interior ~1.00 |
| **4D-F** | `j-e` from mid-ladder onward | ~1.09× | Persistently junction+endpoint; raw≈interior on finest |
| **5D-CN** | `---` … then `j-e` `j-e` | ~1.62× | Junction-owned late; raw last-two ~1.31 vs interior ~1.00 |
| **5D-F** | migrates `j-e`/`--e` → finest `---` | ~1.00× | Peak leaves junction on finest rung; interior last-two still ~1.54 (migration / non-mono) |

Web peaks on these keys are bulk or endpoint-only with raw≈interior (~1.00–1.01×);
the CI fine-GCI fails are **skin** junction-localized (except `5D-F` migration).

**Decision:** residual CI watch is **junction-owned** on four of five keys → next
physics path is a local joint formulation, **not** further global `frac`
densification and **not** interior-only gates. Phase B2 **keep `shared`**;
`transformed_basis` promote bar unchanged (prior dual-mode still fails
equal-or-better on `4D-F` / `2D-F`). No soft-contact / CZM scope.

**Plots:** section geometry + CI ladder composites for these five keys live under
`visualisation/output/gci_study_mitc4/plots/ci_watch_layouts/{family}/` (e.g.
[`D/4D-F_convergence.png`](../visualisation/output/gci_study_mitc4/plots/ci_watch_layouts/D/4D-F_convergence.png)).

## 5D-CN precursor — why global TB regresses membrane CI

Live: `visualisation/output/cap_peak_sampling_5d_cn/` (`precursor_summary.json`,
fine fracs `0.005` / `0.0035`, `skin_offset`).

Under **shared**, finest skin `|N12|` sits on `skin:outer_skin:SS:bay_0` **e0**
with tags **`j-e`** (kind-crossing + endpoint); raw/interior ≈ **1.62×**; peak
~3.5e4 N/m. Under **transformed_basis**, the peak **relocates** to
`skin:outer_skin:PS:bay_3` (endpoint-band only, **not** junction), magnitude
~**6.3e4** N/m (nearly 2×), and cascade `residual_rel` worsens (~6 → ~11). TB
does not amplify the same `SS:bay_0` cusp — it moves ownership to a PS bay
endpoint singularity. That motivated study-only **`hybrid_collinear`** (below).

## SS:bay_0 cluster traction (shared, frac=0.0035)

Dump: `visualisation/output/cap_peak_sampling_5d_cn/ss_bay0_cluster_traction.json`.

Owning cluster for peak station `skin:outer_skin:SS:bay_0:start` is **cluster_id=1**,
**non-collinear** 3-way: `SS:TE:end` + `SS:bay_0:start` + `web:web_4:start`.
Trust/mesh OK; skin `|N12|` ≈ 3.52e4. Live `reaction_dN12_rel` / edge-traction
samples are not populated on the production cascade path (topology collinear +
membership are authoritative here). Confirms a skin–web T-junction foot, not a
bulk strip station.

## Bay-foot densify spike (`SS:bay_0` start only) — do not fold

Study: `visualisation/output/gci_study_mitc4_bay_foot_5d_cn/` +
`cap_peak_sampling_5d_cn/bay_foot/`. CLI:
`--junction-band-labels skin:outer_skin:SS:bay_0 --junction-band-which start
--junction-band-elements 6 --junction-band-step-over-h 0.5` (other panels stay
production band=3).

| Mode | Skin `|N12|` fine GCI | Finest locus | raw/interior |
|------|----------------------|--------------|--------------|
| shared (baseline) | **17.92%** | `SS:bay_0` e0 `j-e` | ~1.62× |
| bay-foot n=6 start | **17.86%** | `SS:bay_0` e0 `j-e` | ~1.62× |

Trust/mesh OK. Peak ownership unchanged; GCI essentially **flat**. CI success bar
not met → **no** five-key watch expansion, **no** Phase B2 smoke, **do not** fold
label-scoped band into production. Keep `shared` / `JUNCTION_BAND_ELEMENTS=3`.
Residual is Defect K at the `SS:TE`–`SS:bay_0`–`web_4` cusp under shared — mesh
pins at that foot alone do not clear MITC4 membrane CI.

## `hybrid_collinear` spike (5D-CN) — do not promote

Study mode: shared numbering on collinear clusters; cluster-basis MPCs only on
non-collinear clusters (`assign_global_nodes_hybrid_collinear`). CLI:
`--interface hybrid_collinear`. Evidence:
`visualisation/output/gci_study_mitc4_hybrid_5d_cn/`,
`cap_peak_sampling_5d_cn/hybrid_collinear/`.

| Mode | Skin `|N12|` fine GCI | Finest peak locus | raw/interior |
|------|----------------------|-------------------|--------------|
| shared | **17.9%** | `SS:bay_0` e0 **`j-e`** | ~1.62× |
| TB | 21.6% | `PS:bay_3` `--e` | ~1.58× (mag ↑) |
| **hybrid_collinear** | **29.1%** | `SS:bay_0` e0 `--e` | ~2.45× |

Trust/mesh OK. Hybrid clears the kind-crossing tag at the shared peak station
but **worsens** raw fine GCI vs shared — CI success bar not met; full five-key
watch expansion and Phase B2 Hashin/gov smoke **skipped**. **Keep production
`shared`**; leave `hybrid_collinear` as study-only diagnostic (not a promote
candidate).

## `hybrid_kind_crossing` spike (CI watch) — do not promote

Study mode: TB only on **kind-crossing ∩ non-collinear** clusters; other
junctions stay shared; cap ties `skin_offset`. CLI:
`--keys watch --gates ci --interface hybrid_kind_crossing`. Evidence + scorecard:
`visualisation/output/gci_study_mitc4_hkc/` (`stability_scorecard.md`).

| Layout | Skin `|N12|` shared → HKC | Skin mid-σ shared → HKC | Verdict |
|--------|---------------------------|-------------------------|---------|
| 3D-CN | 10.1% → **3.0%** | 10.2% → **3.1%** | cleared |
| 4D-CN | 20.2% → **30.3%** | 20.1% → **38.1%** | regress |
| 4D-F | 12.4% → **10.7%** | 11.0% → **3.0%** | mid-σ OK; `|N12|` still >10% |
| 5D-CN | 17.9% → **29.8%** | 18.0% → **29.6%** | regress |
| 5D-F | 21.6% → **21.7%** | 12.4% → **11.8%** | flat / fail |

Trust/mesh **5/5**. Scorecard CI membrane magnitude **1/5** — Phase B entry
(watch cleared → 40/40) **not met**. Study Hashin/gov on residual keys also
regressed on several layouts (e.g. `5D-CN` gov 19.6% → 157%). **Keep production
`shared`**; leave `hybrid_kind_crossing` study-only; do not densify; do not flip
default. Next joint physics must beat this scorecard without Hashin/gov
regressions on the Phase B2 residual set.

## Residual-4 deep-dive (why HKC helped only `3D-CN`)

Evidence: `visualisation/output/residual4_diagnosis/residual4_summary.json`
(cap-peak at `frac∈{0.005,0.0035}` under `shared` vs `hybrid_kind_crossing`,
plus GCI matrix from shared / HKC / TB watch studies).

### Peak ownership (finest rung `frac=0.0035`)

| Layout | Shared skin `|N12|` locus | HKC locus | Notes |
|--------|---------------------------|-----------|-------|
| 4D-CN | `SS:bay_2` e63 **j+e** (raw/int ~1.65×) | `SS:bay_0` e0 **--e** (raw/int ~2.40×) | Peak **migrates**; singularity amplifies |
| 4D-F | `SS:bay_0` e0 **j+e** (raw/int ~1.04×) | `SS:bay_0` e0 **--e** | Same bay-foot; junction tag cleared |
| 5D-CN | `SS:bay_3` e53 **j+e** (raw/int ~1.41×) | `SS:bay_0` e0 **--e** (raw/int ~2.20×) | Migrates to bay-foot; mag ↑ |
| 5D-F | `SS:bay_0` e2 **non-junction** (raw=int) | same panel e2 non-junction | Continuum peak; HKC flat |

### Owning shared clusters (start of peak panel, shared numbering)

| Layout | Cluster members (at peak panel) | collinear | kind-crossing |
|--------|----------------------------------|-----------|---------------|
| 4D-CN | `SS:bay_1`+`SS:bay_2`+`web_1` | false | true |
| 4D-F | `SS:TE`+`SS:bay_0`+`web_3` | false | true |
| 5D-CN | `SS:bay_2`+`SS:bay_3`+`web_1` | false | true |
| 5D-F | n/a for finest peak (not junction-owned) | — | — |

### GCI matrix (skin `|N12|` / mid-σ / Hashin / gov)

| Layout | shared N12 | HKC N12 | TB N12 | shared gov | HKC gov |
|--------|------------|---------|--------|------------|---------|
| 4D-CN | 20.2% | **30.3%** | 6.6% | 12.3% | **62.4%** |
| 4D-F | 12.4% | 10.7% | 4.2% | 3.3% | 4.8% |
| 5D-CN | 17.9% | **29.8%** | 21.6% | 19.6% | **157%** |
| 5D-F | 21.6% | 21.7% | 0.7% | 96.4% | 52.2% |

### Formulation implication

Residual is **mixed**:

1. **Defect-K shared-frame mismatch** at kind-crossing ∩ non-collinear cusps
   dominates `4D-CN` / `4D-F` / `5D-CN` under production `shared` (junction +
   endpoint band ownership).
2. **Peak migration / singularity relocation** under TB-style hybrids (HKC)
   clears the junction tag but often **amplifies** raw GCI and Hashin/gov —
   so HKC is not a safe promote path.
3. **`5D-F`** on the finest rung is already a **non-junction** continuum peak;
   kinematics flips do not address its `|N12|` residual the same way.

Therefore the next spike should keep **shared node IDs** (avoid TB peak
relocation) while correcting the **cluster-frame mismatch** at kind-crossing ∩
non-collinear cusps only → study-only **`shared_cluster_frame`**.

## `shared_cluster_frame` spike (CI watch) — do not promote

Study mode: shared node numbering + Ke congruence into cluster ``t_ref`` on
kind-crossing ∩ non-collinear cusps only. CLI:
`--keys watch --gates ci --interface shared_cluster_frame`. Evidence:
`visualisation/output/gci_study_mitc4_scf/` (`stability_scorecard.md`).

| Check | Result |
|-------|--------|
| Trust / mesh | **5/5** |
| CI membrane magnitude | **0/5** (no watch key cleared) |
| Notable regress | `5D-CN` mid-σ skin **130%**; `3D-CN` mid-σ web **92%** |

Phase B entry **not met**. Hashin/gov also fail study magnitude on several keys.
**Keep production `shared`**; leave `shared_cluster_frame` study-only; do not
densify; do not flip default. Endpoint Ke-frame congruence alone is insufficient
for the residual-4 set — further shared-compatible work needs a different
mechanism (still without soft penalties / interior gates / densify-as-fix).

## Junction-band densify spike (CI watch, 2026-08) — do not fold

Study: `visualisation/output/gci_study_mitc4_junction_band/`
(`JUNCTION_BAND_ELEMENTS=6`, `JUNCTION_BAND_STEP_OVER_H=0.5`, production
`shared` + `skin_offset`, `--gates ci`). CLI overrides:
`--junction-band-elements` / `--junction-band-step-over-h`.
Quarter-`h` spacing (`step_over_h=0.25`) was rejected — it violates mesh QA
`el_min/h` (pins clamped to ≥0.25).

| Layout | Skin `\|N12\|` GCI shared→band6 | Skin mid-σ | Verdict |
|--------|----------------------------------|------------|---------|
| 3D-CN | 10.1% → **11.9%** | 10.2% → 12.0% | regress |
| 4D-CN | 20.2% → 20.1% | 20.1% → 19.9% | flat |
| 4D-F | 12.4% → **11.1%** | 11.0% → 10.4% | slight help; still >10% |
| 5D-CN | 17.9% → 18.1% | 18.0% → 18.2% | flat/regress |
| 5D-F | 21.6% → **36.4%** | 12.4% → 11.1% | strong `\|N12\|` regress |

Trust/mesh stayed OK. **Success bar not met** (need ≤10% on all five) → **do not**
raise production `JUNCTION_BAND_ELEMENTS` (stays 3 / half-`h`). Escalate to
kinematics measurement below.

## Kinematics escalate — TB CI-watch dual-run (measurement only)

Study: `visualisation/output/gci_study_mitc4_tb_ci_watch/`
(`--interface transformed_basis`, production band, `--gates ci`).

| Layout | Skin `\|N12\|` GCI shared→TB | Skin mid-σ | ≤10%? |
|--------|------------------------------|------------|-------|
| 3D-CN | 10.1% → **2.2%** | 10.2% → **2.2%** | yes |
| 4D-CN | 20.2% → **6.6%** | 20.1% → **6.6%** | yes |
| 4D-F | 12.4% → **4.2%** | 11.0% → **4.8%** | yes |
| 5D-CN | 17.9% → **21.6%** | 18.0% → **22.0%** | **no** |
| 5D-F | 21.6% → **0.7%** | 12.4% → **0.7%** | yes |

TB clears raw fine GCI ≤10% on **4/5** watch keys; residual CI membrane fail is
`5D-CN`. Gate CSV may still list `gci_non_monotonic` rows with |GCI|<10% (Roache
applicability — see NUMERICAL_STABILITY §5). **Phase B2 promote bar unchanged:**
prior Hashin/gov dual-mode still fails equal-or-better on `4D-F` / `2D-F` →
**keep production `shared`**. Soft cusp / relative-slip penalties stay deferred
([DEFERRED_RESEARCH.md](DEFERRED_RESEARCH.md)); next joint work targets
`5D-CN` under a non-promoting kinematics experiment, not band densify.

## Historical Cluster A — interface / endpoint peak migration (`4D-F`, `5D-F`, `2D-F`)

> **Historical (pre-ownership campaign).** Current CI watch keys are listed above;
> do not treat this Cluster A table as the live fail set.

| Layout | Symptom | Evidence |
|--------|---------|----------|
| **4D-F** | Skin `\|N12\|` / Hashin late jump | Raw drift ~2.5× / 3.8×; **interior ~1.02× / 1.34×** |
| **5D-F** | Same pattern (weaker `\|N12\|`) | Raw Hashin ~3.8×; interior ~1.11× |
| **2D-F** | Hashin / `\|N12\|` interface-local | Raw Hashin ~3.5×; interior ~1.02× |

**Root cause:** peaking element sits in the junction / endpoint band; knot-merge
and global `frac` move which station owns the max. **Not** bulk strip physics.

**Fix applied:** junction-band `required_s` + `KnotConstrainedUniform` on
skin/web; `n_cap_samples=120` for D-family geometry sampling. Success =
raw peaks improve on re-study (not switching gates to interior).

## Historical Cluster B — bulk FI climb (`0A`, `3B-*`, `4D-CN`)

> **Historical.** Post-ownership study watch for `governing_fi` is in the table above.

| Layout | Symptom | Evidence |
|--------|---------|----------|
| **0A** | Hashin / `governing_fi` high GCI | Raw and interior Hashin both ~3.8× |
| **3B-F / 3B-CN** | Hashin / gov FI | Interior can be *worse* than raw → not pure endpoint mask |
| **4D-CN** | Hashin partial; `\|M11\|` climbs | Interior Hashin flatter; bending still refining |

**Root cause:** design FI still tracks refining stress peaks (junction-adjacent
or mid-panel) under the short/medium ladder. Governing selection now records
`panel_index` / `element_index` / `criterion` on the ladder CSV for ownership
checks after mesh fixes.

## Historical Cluster C — bending (`|M11|/|M22|`) across outliers

> **Historical.** Bending remains study-reported / not CI (see NUMERICAL_STABILITY §5).

Raw and interior `|M11|` both climb (~8–12× on the 5-rung campaign). Junction
densification helps localization of membrane/FI peaks; bending needs the
finer ladder (`0.0035`) to confirm asymptote — not an interior-gate workaround.

## Near-null fields

`|M12|`, `|Q1|`, `|τ13|` remain ~machine-zero under the campaign load — see
§9 of [NUMERICAL_STABILITY_AND_GCI.md](NUMERICAL_STABILITY_AND_GCI.md).

## Cap-peak tag study (post junction-local \(h\))

Live output: `visualisation/output/cap_peak_sampling/`
(`cap_peak_ladder.csv`, `cap_peak_locations.json`, `cap_peak_drift_summary.csv`).

Tag shorthand: `j` = junction, `t` = tie_slave, `e` = endpoint_band, `-` = absent.

| Layout | Skin `\|N12\|` tags (6 rungs) | Skin Hashin tags | Notes |
|--------|-------------------------------|------------------|-------|
| 4D-F | mostly bulk / endpoint; migrates | endpoint-heavy | Fine-rung raw≈interior for `\|N12\|` after fix |
| 5D-F | junction+endpoint | junction+endpoint | Classic Cluster A |
| 0A | mix endpoint/bulk | mix endpoint/bulk | Cluster B — interior still climbs |
| 3B-F / 3B-CN | junction+endpoint dominate | same | FI ownership at skin–web joints |
| 4D-CN | junction+endpoint | junction+endpoint | Same |
| 2D-F | junction+endpoint | junction+endpoint | Same |

**Hashin / gov FI follow-up:** peaks still sit on junction/endpoint stations after
the first junction densification pass. Ladder now records
`governing_panel_index` / `governing_element_index` / `governing_criterion`.
Junction band tightened to `JUNCTION_BAND_ELEMENTS=3` with half-`h` spacing;
raw GCI (not interior gates) remains the success metric.

## Historical before / after (outlier subset, fine GCI fraction)

> Campaign archaeology (junction densification / ownership filter). Live CI watch
> keys are in the table at the top of this file.

Re-study: `visualisation/output/gci_study_mitc4_outliers/` (6-rung ladder incl. `0.0035`).

| Layout | `|N12|` skin GCI | Hashin skin GCI | gov FI GCI |
|--------|------------------|-----------------|------------|
| 4D-F | **1.20 → 0.005** | 0.35 → 1.09 | 0.91 → 1.10 |
| 5D-F | **0.77 → 0.007** | **1.36 → 0.33** | 0.45 → 0.97 |
| 0A | 0.0004 → 0.0004 | **1.23 → 0.001** | **1.23 → 0.001** |
| 3B-F | 0.003 → 0.007 | **0.42 → 0.004** | **0.22 → 0.004** |
| 3B-CN | 0.009 → 0.018 | 0.34 → 0.19 | 0.58 → 0.19 |
| 4D-CN | 0.011 → 0.044 | **0.92 → 0.26** | 0.32 → 0.37 |
| 2D-F | 0.054 → 0.138 | 0.11 → 0.13 | **1.11 → 0.32** |

Cluster A membrane success criterion met on **4D-F / 5D-F** (raw `|N12|`) after
junction densification. Hashin/gov cleared on **0A / 3B-F** in that pass;
D-family Hashin remained junction-owned until the ownership filter below.

### After FI ownership filter (kind-crossing + tie-slave)

Same outlier keys, continuum-owned Hashin / `governing_fi` (cap Hashin → 0 under
`skin_offset`). Finest-rung `governing_*` lives on skin/web panels only.

| Layout | Hashin skin GCI | gov FI GCI | gov last-two | Notes |
|--------|-----------------|------------|--------------|-------|
| 4D-F | 0.640 | **0.068** | 1.06 | Gov improved vs 1.10; skin Hashin still climbing |
| 5D-F | **0.051** | 0.529 | 1.22 | Hashin improved; gov non-monotonic (`gci_non_monotonic`) |
| 0A | 0.033 | 0.033 | 1.03 | Still study-OK on last-two; full-ladder Hashin drift |
| 3B-F | 0.125 | 0.125 | 1.12 | Continuum-owned; residual junction-adjacent climb |
| 3B-CN | 0.158 | 0.241 | 1.15 | Same |
| 4D-CN | 0.705 | 0.358 | 1.16 | Still Defect-K / joint-owned asymptote |
| 2D-F | **0.011** | **0.0015** | 1.00 | Cleared under ownership filter |

**Verdict:** tie_slave ownership is fixed (tied cap FI no longer governs).
Kind-crossing exclusion moves gov FI onto continuum strips; **2D-F** and **4D-F
gov** meet the “raw peaks improve” success bar. Remaining D/B fails are joint-
model work (Defect K kinematics)—not further global `frac` densification or
`ENDPOINT_BAND` widening. Hashin / gov stay **study gates**, not CI.

## Defect K dual-mode (shared vs transformed_basis)

After composing `skin_offset` with cluster-frame MPCs, outlier ladders were
re-run under both junction modes (`gci_defect_k_shared/`, `gci_defect_k_tb/`).

| Layout | Hashin shared→TB | gov shared→TB | Notes |
|--------|------------------|---------------|-------|
| 4D-CN | 0.64 → **0.13** | 0.12 → **0.08** | Strong TB win |
| 5D-F | 0.04 → **0.01** | 0.96 → **0.01** | Clears non-monotonic gov |
| 4D-F | 0.05 → 0.09 | 0.03 → 0.05 | TB regress |
| 3B-F | 0.09 → **0.08** | 0.09 → **0.08** | Slight TB win |
| 3B-CN | 0.21 → **0.15** | 0.21 → **0.15** | TB win |
| 2D-F | 0.03 → 0.08 | 0.002 → 0.08 | Control regress under TB |

**Phase B2 decision: keep `shared`.** Trust OK both modes; `|N12|` improves under
TB everywhere; Hashin/gov only 4/6 equal-or-better — promote bar not met. See
[SHELL_MODEL.md](SHELL_MODEL.md) Phase B2.

## Classification → fix (D-family Hashin / gov FI)

| Class | Layouts / tags | Action |
|-------|----------------|--------|
| **tie_slave** | Cap peaks under `skin_offset` (`t=1`) | Exclude tie-slave elements from Hashin / gov FI peak ownership — continuum peak lives on master skin (tied cap Hashin → 0) |
| **junction / Defect K** | Skin Hashin `j-e` / `j--` on 4D-*/5D-*/2D-F / 3B-* | Exclude **kind-crossing** junction nodes only (skin–cap / skin–web / cap–web); same-kind strip interfaces stay eligible |
| **bulk** | 0A (post mesh fix) | Cleared under study GCI; no further mesh work |

Implemented in `cascade_mesh_convergence._fi_sampling_excluded` +
`_kind_owner_counts` (Hashin + `governing_fi` only; membrane/mid-σ unchanged).
`cap_peak_sampling` junction tags use the same kind-crossing definition.
Re-study outlier subset under `--gates study` after the ownership filter.

## Bending / shear (deferred)

`|M11|/|M22|/|Q2|` improved but remain **study-reported, not CI**. If a consumer
later requires gates, treat as junction bending ownership—not global ladder
densification or interior-only acceptance.
