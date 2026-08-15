# `section_shell_fea` — documentation index

Thesis-facing map of deep docs for the FE worker. Package README (L1):
[`../README.md`](../README.md).

**Permanent layout:** `pre_processing/` and `processing/` are the staging homes —
do not flatten or relocate them.

## Reader questions

| Question | Where |
|----------|--------|
| What is solved? | [../README.md §4](../README.md#4-governing-equations), [SECTION_SOLVE_STAGES.md](SECTION_SOLVE_STAGES.md) |
| Which cascade tier holds Hashin FI? | **Quaternary** — [processing/static/results/docs/README.md](../processing/static/results/docs/README.md), [compute_quaternary](../processing/static/results/compute_quaternary/docs/README.md) |
| Which figure proves mesh / GCI? | [`ladder_governing_fi.png`](../visualisation/output/gci_study_mitc4/plots/ladder_governing_fi.png) — [NUMERICAL_STABILITY_AND_GCI.md](NUMERICAL_STABILITY_AND_GCI.md) |
| MITC4 / FSDT assumptions; frames **C ≡ E** | [SHELL_MODEL.md](SHELL_MODEL.md), [TAXONOMY.md](TAXONOMY.md) |
| Non-ownership (geometry / mesh / bind / SFV shaping) | [../README.md §1](../README.md#1-purpose--ownership), coexistence doc below |

---

## Taxonomy & solve stages

| Doc | Role |
|-----|------|
| [TAXONOMY.md](TAXONOMY.md) | Station → section → strip → MaterialZone → element |
| [SECTION_SOLVE_STAGES.md](SECTION_SOLVE_STAGES.md) | Prepare → assemble → constrain → reduce(`T`) → solve → P→S→T→Q |
| [SHELL_MODEL.md](SHELL_MODEL.md) | MITC4/FSDT model, BC modes, acceptance metrics, handoff contract |
| [ELEMENT_AGNOSTIC_ARCHITECTURE.md](ELEMENT_AGNOSTIC_ARCHITECTURE.md) | Kernel-agnostic cascade vs formulation home |
| [LINEAR_STATIC_MAIN_TRACK_PHASE0.md](LINEAR_STATIC_MAIN_TRACK_PHASE0.md) | Production FI track (Channel-A removed) |

---

## GCI / numerical stability

| Doc | Role |
|-----|------|
| [NUMERICAL_STABILITY_AND_GCI.md](NUMERICAL_STABILITY_AND_GCI.md) | Residual trust vs mesh independence; harness; E-frame peaks |
| [GCI_OUTLIER_DIAGNOSIS.md](GCI_OUTLIER_DIAGNOSIS.md) | Outlier triage on GCI campaigns |
| Cap-peak sampling | [`../processing/static/diagnostics/cap_peak_sampling.py`](../processing/static/diagnostics/cap_peak_sampling.py); CSV SoT [`../visualisation/output/cap_peak_sampling/`](../visualisation/output/cap_peak_sampling/); CI-watch [`../visualisation/output/cap_peak_sampling_ci_watch/`](../visualisation/output/cap_peak_sampling_ci_watch/) |

**Proof PNGs (committed):**

- Governing FI ladder: [`../visualisation/output/gci_study_mitc4/plots/ladder_governing_fi.png`](../visualisation/output/gci_study_mitc4/plots/ladder_governing_fi.png)
- Displacement ladder: [`../visualisation/output/gci_study_mitc4/plots/ladder_d_inf.png`](../visualisation/output/gci_study_mitc4/plots/ladder_d_inf.png)
- Fine Hashin peaks: [`../visualisation/output/gci_study_mitc4/plots/gci_fine_peak_hashin_by_layout.png`](../visualisation/output/gci_study_mitc4/plots/gci_fine_peak_hashin_by_layout.png)
- Layout geometry + CI ladder (watch example): [`../visualisation/output/gci_study_mitc4/plots/ci_watch_layouts/D/4D-F_convergence.png`](../visualisation/output/gci_study_mitc4/plots/ci_watch_layouts/D/4D-F_convergence.png)
- Defect-K shared ladder: [`../visualisation/output/gci_defect_k_shared/plots/ladder_governing_fi.png`](../visualisation/output/gci_defect_k_shared/plots/ladder_governing_fi.png)

---

## CalculiX quarantine

[`../calculix_ccx/README.md`](../calculix_ccx/README.md) — external `ccx` S4R oracle only.
Production theory is MITC4; opt-in via `register_calculix_ccx_backend()`.
Related: [CALCULIX_CCX_ELEMENT_LIBRARY_RESTRUCTURE.md](CALCULIX_CCX_ELEMENT_LIBRARY_RESTRUCTURE.md),
[ROOT_SEGMENT_CCX_DIAGNOSTIC.md](ROOT_SEGMENT_CCX_DIAGNOSTIC.md),
[PERIODIC_CCX_HOMOGENIZATION.md](PERIODIC_CCX_HOMOGENIZATION.md) (non-product).

---

## Stage homes (equations + I/O)

### `pre_processing/`

| Home | README |
|------|--------|
| Loading (apply SFV) | [`../pre_processing/loading/README.md`](../pre_processing/loading/README.md) |
| Element library (MITC4 `Ke`) | [`../pre_processing/element_library/README.md`](../pre_processing/element_library/README.md) |
| Constitutive (ABD) | [`../pre_processing/constitutive/README.md`](../pre_processing/constitutive/README.md) |
| Mesh / panel topology | [`../pre_processing/mesh/README.md`](../pre_processing/mesh/README.md) |
| Containers | [`../pre_processing/containers/README.md`](../pre_processing/containers/README.md) |
| Dispatch | [`../pre_processing/dispatch/README.md`](../pre_processing/dispatch/README.md) |
| Layer index | [`../pre_processing/README.md`](../pre_processing/README.md) |

### `processing/`

| Home | README |
|------|--------|
| Static assemble / cascade | [`../processing/static/README.md`](../processing/static/README.md) |
| Cascade tier index | [`../processing/static/results/docs/README.md`](../processing/static/results/docs/README.md) |
| Buckling | [`../processing/buckling/README.md`](../processing/buckling/README.md) |
| Layer index | [`../processing/README.md`](../processing/README.md) |

### Runners / viz / aux

| Home | README |
|------|--------|
| Simulation runners | [`../simulation_runner/README.md`](../simulation_runner/README.md) |
| Visualisation + demos | [`../visualisation/README.md`](../visualisation/README.md), [`../visualisation/demos/README.md`](../visualisation/demos/README.md) |
| Benchmarks (GCI / CCX campaigns) | [`../benchmarks/README.md`](../benchmarks/README.md) |
| Interface adapters | [`../interface/README.md`](../interface/README.md) |
| I/O payloads | [`../io/README.md`](../io/README.md) |
| Validation oracles | [`../validation/README.md`](../validation/README.md) |

---

## Demos wired to stages

| Demo / artefact | Stage it proves |
|-----------------|-----------------|
| [`section_centerline_mesh_v2/*.png`](../visualisation/output/section_centerline_mesh_v2/) | Mesh map / panel topology (prep) |
| Cap-tie strategies [`demos/cap_tie_strategies/`](../visualisation/demos/cap_tie_strategies/) | Junction / interface constraint behaviour |
| GCI ladders under [`visualisation/output/gci_study_mitc4/`](../visualisation/output/gci_study_mitc4/) | Static solve stability + mesh independence |
| Cap-peak CSVs [`visualisation/output/cap_peak_sampling/`](../visualisation/output/cap_peak_sampling/) | Peak localization diagnostics |
| Benchmark shell elements [`visualisation/output/benchmark_shell_elements/`](../visualisation/output/benchmark_shell_elements/) | Element / FI oracle comparison |
| Tier `compute_*/examples/outputs/` | Secondary / tertiary / quaternary cascade maths |

Run entrypoints: [`../visualisation/demos/README.md`](../visualisation/demos/README.md),
[`../benchmarks/`](../benchmarks/).

---

## Bug log & operational notes

| Doc | Role |
|-----|------|
| [bug_log.md](bug_log.md) | Package bug / frame-canon log (**canonical**) |
| Cascade tier bug logs | Under each `processing/static/results/compute_*/docs/bug_log.md` |
| [LOAD_APPLICATION_DOUBLE_COUNTING.md](LOAD_APPLICATION_DOUBLE_COUNTING.md) | Beam \(V\) vs Channel B exclusivity |
| [DEPRECATED_ALIASES.md](DEPRECATED_ALIASES.md) | Retired names |
| [DEFERRED_RESEARCH.md](DEFERRED_RESEARCH.md) | Parked topics |
| [NEXT_STEPS_PERF_AND_MAINTAINABILITY.md](NEXT_STEPS_PERF_AND_MAINTAINABILITY.md) | Perf backlog |
| [PERFORMANCE_OPTIMISATION_REPORT.md](PERFORMANCE_OPTIMISATION_REPORT.md) | Perf findings |

Architecture proposals (historical; layout is permanent):
[PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md](PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md),
[PROCESSING_ARCHITECTURE_PROPOSAL.md](PROCESSING_ARCHITECTURE_PROPOSAL.md).

---

## Upstream coexistence

[`../../docs/COEXISTENCE_IMPLICIT_SHELL.md`](../../docs/COEXISTENCE_IMPLICIT_SHELL.md) —
geometry → mesh → bind → loads → FE ownership ladder.
