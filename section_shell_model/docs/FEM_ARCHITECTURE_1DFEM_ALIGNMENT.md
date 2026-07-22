# `section_shell_model` vs 1d-fem: computational architecture alignment

**Status:** current (Phase 0 pure FE)  
**Last updated:** 2026-07-22

This document evaluates `section_shell_model`'s organisational philosophy against `1d-fem` (`C:\Users\s1834431\Projects\1d-fem`). 1d-fem is a reference for **how to decompose a solve pipeline**, not a line-for-line contract.

**Related docs**

| Doc | Role |
|-----|------|
| [LINEAR_STATIC_MAIN_TRACK_PHASE0.md](LINEAR_STATIC_MAIN_TRACK_PHASE0.md) | Authoritative product surface (two runners; Channel-A removed) |
| [ELEMENT_AGNOSTIC_ARCHITECTURE.md](ELEMENT_AGNOSTIC_ARCHITECTURE.md) | Layer rules + element factory |
| [SECTION_SOLVE_STAGES.md](SECTION_SOLVE_STAGES.md) | Staged static ops + assembly orchestra naming |
| [PROCESSING_ARCHITECTURE_PROPOSAL.md](PROCESSING_ARCHITECTURE_PROPOSAL.md) | Four-tier results |

---

## Principle: organisational philosophy, not a mirror

Verdict categories:

- **(A) Direct analogue** — same role / relative position.
- **(B) Blade-specific extension** — warranted; do not rename for parity.
- **(C) Drift to correct** — worth fixing.
- **(D) Deliberate departure** — diverging is correct.

---

## Verdict table (2026-07-22)

| `section_shell_model` item | Category | 1d-fem counterpart | Verdict |
|------------------------------|----------|---------------------|---------|
| `pre_processing/containers/` formulation DTOs | **A** | `element_library/gauss_point_data.py` (active) | In step |
| `processing/static/operations/solve_states.py` | **A** | `processing/static/results/containers/global_results.py` | **Closed** — relocated under `processing/` |
| `pre_processing/element_library/` + `resolve_element_factory` | **A / D (mechanism)** | `ElementFactory.ELEMENT_CLASS_MAP` | Landed; imperative register for CCX gating |
| `pre_processing/mesh/`, `pre_processing/loading/` | **B** | none | Keep |
| `processing/static/operations/` stage classes | **A** | `processing/static/operations/*` | Direct |
| `processing/static/diagnostics/` | **A** | same relative place | Direct |
| Four-tier static results (P→Q) | **B** | three tiers (P→T) | Quaternary FI is correct extension |
| `SectionLinearStaticRunner` | **A** | `LinearStaticSimulationRunner` | Sole linear-static orchestrator |
| `SectionLinearBucklingRunner` (full-section `K`/`Kg`) | **A** | `LinearBucklingSimulationRunner` / spectral backend | **Landed** — connected-section pencil with static `T`; panel classical optional |
| Channel-A / Saint-Venant periodic / `api.k6` | **removed** | n/a | **Deleted (Phase 0)** — not a pending cutover |
| `orchestration/precompute/` parallel stations | **B** | `workflow_orchestrator/` (single job) | Keep outside this package |
| Library-style runners (`from_cache`, typed results) | **D** | job-folder + results-tree runners | Intentional — this package is a library, not a CLI FEM |
| Multi-family breadth (eigen/harmonic/transient) | **D** | full analysis taxonomy | Shell stays static + buckling only |

---

## Where `section_shell_model` should keep diverging

1. **`pre_processing/mesh/` and `pre_processing/loading/`** — no 1-D-beam analogue.
2. **Quaternary results tier** — composite Hashin / Von Mises FI.
3. **`orchestration/precompute/`** parallel station orchestration — outside this package.
4. **Library-style runner API** — `from_cache` / typed dataclasses vs 1d-fem job results trees (**category D**; not unfinished drift).
5. **Registration mechanism** — imperative `register_element_factory()` for CCX opt-in gating.
6. **BC realisation** — elimination via `T` vs 1d-fem penalty + condensation (stages still separate).

Do **not** reintroduce Channel-A / periodic product surfaces for alignment theatre.

---

## Closed gaps

| Gap | Status |
|-----|--------|
| Solve-state DTOs under `pre_processing/` | **Done** → `processing/static/operations/solve_states.py` |
| Element factory registry | **Done** |
| Top-level `engine/` / package `api/` | **Done** — deleted |
| Legacy MITC4 clamped / HSDT monoliths | **Done** — deleted |
| Channel-A FI / `api.periodic` / `api.k6` | **Done** — removed (Phase 0) |
| Full-section geometric-stiffness buckling | **Done** — `SectionLinearBucklingRunner.from_static_solve` / default `eigen_track="section"` |

---

## Summary

`section_shell_model` follows 1d-fem's computational philosophy: `pre_processing` prepares → `processing` computes → `simulation_runner` sequences and owns result sets. Production surfaces are the two FE runners (`SectionLinearStaticRunner`, `SectionLinearBucklingRunner` with full-section `K`/`Kg` by default). Channel-A is gone. Remaining differences from 1d-fem are intentional domain responses, not backlog.
