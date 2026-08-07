# section_shell_model - FE kernel worker

**Role:** FE **worker** only - not an orchestration product.

This package has **no** knowledge of sizing vs fatigue, parallel vs serial, or
iterative vs single-shot. Callers decide when to invoke it and which cascade
tiers to read.

## Contract

| | |
|---|---|
| **Inputs** | Section geometry + material / panel laminate assignments + loads / BCs (DTOs) |
| **Work** | Solve \(F = KU\); fill the **P → S → T → Q** cascade bin |
| **Outputs** | Cascade bin (`CascadeBin` / `SectionLinearStaticResult`) including per-solve `diagnostics` (`DiagnosticsBin`); use cases read whichever tiers they need |
| **Agnosticism** | No `purpose="sizing"|"fatigue"` modes; optional tier gates (if any) are compute-cost knobs only |

```text
geometry + panel materials + loads
            │
            ▼
     section_shell_model   ← this package (worker)
            │  F = KU
            ▼
     CascadeBin  P → S → T → Q + diagnostics
            │
            ├── section_selection     may read Q (FI) + nest diagnostics into sizing logs
            ├── bv_methodology        may read T (ply/shell stresses → rainflow)
            └── use case X            any tier
```

`DiagnosticsBin` holds FE phase wall times for **this** worker call only (assemble /
solve / cascade tiers). It is structured data — not a campaign `.log` file writer.
Sizing trial traces and `SECTION_SELECTION_STATION_LOG` file dumps live in
`section_selection.processing.shell_station.station_sizing_timing_log`.

## Public entry

```python
from blade_structure_utilities.fem_utilities.section_shell_model import (
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

## Orchestration elsewhere

Control flow and interpretation live **outside** this package:

- `blade_structure_precompute.module.section_selection` — FI sizing loops, spanwise monotonicity, station parallelism (`processing/shell_station/`)
- `blade_structure_fatigue/examples/bv_methodology` — time-step / rainflow / Miner
- any future **use case X**

Thin adapters in those packages **call** this worker; they do not define it.

## Non-goals (not in this package)

- Rainflow counting, Miner accumulation
- Thickness bisection / optimiser loops / campaign I/O
- Use-case modes or “sizing vs fatigue” APIs
- Station sizing `.log` dumps (`SECTION_SELECTION_STATION_LOG`) — orchestration only

Hashin / von Mises **formulas** live in `material_utilities.failure`; the worker may populate Q by calling them. Whether a use case *requires* Q for decisions is outside the worker.

## Shared inputs (BSU)

| Package | Role |
|---|---|
| `geometry_utilities.section_geometry` | Midline / strip geometry |
| `material_utilities` | CLPT ABD, failure formulas, layup / panel adapters |
| `load_utilities.beam_to_shell_loads` | Beam → shell load mapping |

## Physical model (summary)

FSDT shell (MITC4 production kernel) with CLT laminate ABD per panel. Taxonomy:
station → section → subcomponent → strip → panel → element — see
[`docs/TAXONOMY.md`](docs/TAXONOMY.md). Stage detail:
[`docs/SECTION_SOLVE_STAGES.md`](docs/SECTION_SOLVE_STAGES.md),
[`docs/SHELL_MODEL.md`](docs/SHELL_MODEL.md).

Meshes each structural subcomponent mid-surface into strip shell elements and
recovers shell resultants / ply stresses / FI via the linear-static cascade.
Meshing, junction topology, and recovery are **element-agnostic**; CalculiX
thin-wall remains optional validation only.
