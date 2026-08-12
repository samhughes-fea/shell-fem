# `section_shell_fea` — FE worker

**Package:** `blade_structure_utilities.fem_utilities.section_shell_fea`  
**Former name:** `fem_utilities.section_shell_model`  
**Role:** FE **worker** only — apply precomputed loads, assemble, solve, cascade.

This package has **no** knowledge of sizing vs fatigue, parallel vs serial, or
iterative vs single-shot. Callers decide when to invoke it and which cascade
tiers to read.

```text
section_geometry → section_contract → section_mesh
        → section_bind (MaterialZone)
        → section_loads (SectionForceVector)
        → section_shell_fea (apply SFV → f / Fe → Ke → solve → P→S→T→Q)  ← this worker
```

Canonical import: `blade_structure_utilities.fem_utilities.section_shell_fea`.

### Boundary

| | |
|---|---|
| **Product** | Apply SFV → `f`/`Fe`; `Ke`; assemble; BC; solve; P→S→T→Q + `DiagnosticsBin` |
| **Must not own** | Midline CSG, strip discretisation, layup *policy*, SFV *shaping*, TractionField construction |
| **Barrel** | `blade_structure_utilities.fem_utilities.section_shell_fea` |
| **Consumers** | Import the FE barrel / runner only; import `section_bind` / `section_loads` barrels for types — never deep home paths |

Ownership index: [`docs/COEXISTENCE_IMPLICIT_SHELL.md`](../../docs/COEXISTENCE_IMPLICIT_SHELL.md).

## Table of contents

1. [Purpose / ownership](#1-purpose--ownership)
2. [Ownership homes](#2-ownership-homes)
3. [Public API](#3-public-api)
4. [Contract](#4-contract)
5. [Provenance / locked boundaries](#5-provenance--locked-boundaries)
6. [Package layout](#6-package-layout)
7. [See also](#7-see-also)

---

## 1. Purpose / ownership

| Owns | Does **not** own |
|------|------------------|
| Apply `SectionForceVector` → nodal `f` / element `Fe` | SFV shaping ([`section_loads`](../../load_utilities/section_loads/README.md)) |
| MITC4 `Ke`, assemble, BC, solve \(F=KU\) | Midline CSG / strip Q4 ([`section_geometry`](../../geometry_utilities/section_geometry/README.md), [`section_mesh`](../../mesh_utilities/section_mesh/README.md)) |
| Cascade bin P→S→T→Q + `DiagnosticsBin` | Layup / thickness *policy* ([`section_bind`](../../material_utilities/section_bind/README.md)) |
| Optional CalculiX validation backend | Sizing / fatigue orchestration (BSP / BSF) |

| Actor | Role |
|-------|------|
| **Producer (upstream)** | Bind + loads — zones + precomputed SFV |
| **This worker** | Classical FEM on the section shell |
| **Consumers** | `section_selection`, `bv_methodology`, other use cases reading cascade tiers |

---

## 2. Ownership homes

| # | Home | Owns |
|---|------|------|
| 1 | [`pre_processing/loading/README.md`](pre_processing/loading/README.md) | Apply SFV → nodal `f` / `Fe` (product load path) |
| 2 | [`pre_processing/element_library/README.md`](pre_processing/element_library/README.md) | MITC4 / formulation factory (`Ke`) |
| 3 | [`processing/static/README.md`](processing/static/README.md) | Assemble / BC / solve + cascade index |
| 4 | [`simulation_runner/README.md`](simulation_runner/README.md) | `SectionLinearStaticRunner` / buckling runner |
| 5 | [`calculix_ccx/README.md`](calculix_ccx/README.md) | Quarantined CCX validation backend |
| 6 | [`visualisation/README.md`](visualisation/README.md) | Plot helpers + demos (not a math kernel) |

Deep stage taxonomy: [`docs/TAXONOMY.md`](docs/TAXONOMY.md),
[`docs/SECTION_SOLVE_STAGES.md`](docs/SECTION_SOLVE_STAGES.md),
[`docs/SHELL_MODEL.md`](docs/SHELL_MODEL.md). Cascade tier maths:
[`processing/static/results/docs/README.md`](processing/static/results/docs/README.md).

---

## 3. Public API

```python
from blade_structure_utilities.fem_utilities.section_shell_fea import (
    CascadeBin,
    DiagnosticsBin,
    SectionLinearStaticRunner,
)

bin: CascadeBin = (
    SectionLinearStaticRunner(panel_topology=..., resultants=...)
    .execute()
    .get_results()
)
# sizing example: bin.quaternary_results.hashin_fi...
# fatigue example: bin.tertiary_results...
# FE phase timings: bin.diagnostics.phase_totals_s
```

Also: `SectionLinearBucklingRunner` (prestress → eigen) for buckling campaigns.

Pass a precomputed `section_force_vector=` when the caller shaped SFV outside FE.
Do **not** shape loads or bind laminates inside this worker.

---

## 4. Contract

| | |
|---|---|
| **Inputs** | Bound `MaterialZone`s / panel topology + precomputed `SectionForceVector` + BCs |
| **Work** | Apply SFV → nodal `f`; solve \(F = KU\); fill the **P → S → T → Q** cascade bin |
| **Outputs** | Cascade bin (`CascadeBin` / `SectionLinearStaticResult`) including per-solve `diagnostics` |
| **Agnosticism** | No `purpose="sizing"|"fatigue"` modes; optional tier gates are compute-cost knobs only |

```text
bound zones + precomputed SFV
            │
            ▼
     section_shell_fea   ← this package (worker)
            │  F = KU
            ▼
     CascadeBin  P → S → T → Q + diagnostics
            │
            ├── section_selection     may read Q (FI) + nest diagnostics into sizing logs
            ├── bv_methodology        may read T (ply/shell stresses → rainflow)
            └── use case X            any tier
```

`DiagnosticsBin` holds FE phase wall times for **this** worker call only.
Campaign `.log` dumps live in orchestration (`section_selection`), not here.

---

## 5. Provenance / locked boundaries

| Claim | Today |
|-------|-------|
| Renamed from `section_shell_model` | Package path `fem_utilities/section_shell_fea` |
| Apply SFV only | `section_force_vector_to_nodal_load`; Channel B staging in `section_loads` |
| No orchestration modes | No sizing/fatigue APIs on the runner |
| CCX quarantined | Opt-in `register_calculix_ccx_backend()`; not production theory |
| Demos | [`visualisation/demos/`](visualisation/demos/README.md); artifacts under `visualisation/output/` |

Orchestration stays outside: `blade_structure_precompute.module.section_selection`,
`blade_structure_fatigue/...`, any future use case X.

---

## 6. Package layout

```text
section_shell_fea/
  pre_processing/
    loading/           # apply SFV (home 1)
    element_library/   # MITC4 Ke (home 2)
    dispatch/          # theory / solve registries
    containers/        # FE-local types
    mesh/              # FE-local panel topology helpers
  processing/
    static/            # assemble / solve / cascade (home 3)
    buckling/          # eigen track
  simulation_runner/   # public runners (home 4)
  calculix_ccx/        # CCX validation (home 5)
  visualisation/       # helpers + demos/ + output/ (home 6)
  benchmarks/          # CLI studies → visualisation/output/
  interface/           # thin adapters
  docs/                # taxonomy / stages / architecture
  README.md            # package index
```

---

## 7. See also

- [`section_loads`](../../load_utilities/section_loads/README.md) — shape SFV
- [`section_bind`](../../material_utilities/section_bind/README.md) — MaterialZone
- [`section_mesh`](../../mesh_utilities/section_mesh/README.md) — strip mesh inputs
- [`section_contract`](../../geometry_utilities/section_contract/README.md) — DTO ABI
- [`COEXISTENCE_IMPLICIT_SHELL.md`](../../docs/COEXISTENCE_IMPLICIT_SHELL.md)
- [`docs/TAXONOMY.md`](docs/TAXONOMY.md) · [`docs/SHELL_MODEL.md`](docs/SHELL_MODEL.md)
