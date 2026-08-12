# demos — section_shell_fea visualisation catalog

Package-level demos and study drivers. Cascade-tier documentation figures remain
under `processing/static/results/compute_*/examples/` (committed PNGs linked from
tier deep docs).

↑ Parent suite: [`../README.md`](../README.md)

## Artifacts SoT

[`../output/`](../output/) — GCI studies, mesh sweeps, cap-tie outputs, etc.
(formerly `examples/output/`).

## Scripts

| Script / directory | Role |
|--------------------|------|
| `cap_tie_strategies/` | BL-003 spar-cap tie coverage / FI response sweeps ([`cap_tie_strategies/README.md`](cap_tie_strategies/README.md)) |
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
