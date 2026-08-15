# demos — section_shell_fea visualisation catalog

Package-level demos and study drivers. Cascade-tier documentation figures remain
under `processing/static/results/compute_*/examples/` (committed PNGs linked from
tier deep docs).

↑ Parent suite: [`../README.md`](../README.md) · Docs index: [`../../docs/README.md`](../../docs/README.md)

## Artifacts SoT

[`../output/`](../output/) — GCI studies, mesh sweeps, cap-tie outputs, etc.
(formerly `examples/output/`).

## Wired to stages

| Script / artefact | Stage |
|-------------------|-------|
| `cap_tie_strategies/` | Mesh / interface constraints (shared vs tie coverage) — [`cap_tie_strategies/README.md`](cap_tie_strategies/README.md) |
| `../output/section_centerline_mesh_v2/` | Prep mesh map |
| `../output/gci_study_mitc4/` | Static assemble/solve + Q GCI |
| `../output/cap_peak_sampling/` | Static diagnostics (peak tags) |
| `../output/benchmark_shell_elements/` | Element library / FI comparison |
| Tier `compute_*/examples/` | Cascade P→S→T→Q maths demos |

## Scripts

| Script / directory | Role |
|--------------------|------|
| `cap_tie_strategies/` | BL-003 spar-cap tie coverage / FI response sweeps |
| `repro_section_centerline_mesh_v2.py` | Deprecated redirect → [`section_mesh` viz](../../../../mesh_utilities/section_mesh/visualisation/demos/section_centerline_mesh.py) |

## How to run

```bash
# Cap-tie coverage gate (geometry-only)
python blade_structure_utilities/fem_utilities/section_shell_fea/visualisation/demos/cap_tie_strategies/compare_cap_tie_strategies.py

# Benchmarks (write under visualisation/output/)
python blade_structure_utilities/fem_utilities/section_shell_fea/benchmarks/run_grid_independence_study.py
```

## See also

- [`../README.md`](../README.md)
- [`../../benchmarks/`](../../benchmarks/)
- Cascade tier demos: [`../../processing/static/results/docs/README.md`](../../processing/static/results/docs/README.md)
- Package L1 PNG proof: [`../../README.md#5-png-proof`](../../README.md#5-png-proof)
