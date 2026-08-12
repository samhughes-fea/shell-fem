# visualisation — plot helpers and demos

Operational suite for the `section_shell_fea` worker: matplotlib helpers and
figure-producing demos. FE mathematics live in loading / element_library /
processing / simulation_runner.

↑ Parent: [`../README.md`](../README.md)

## Purpose / ownership

| Owns | Does **not** own |
|------|------------------|
| Mesh fringe / section-axes plot helpers | Cascade tier math (`processing/static/results/`) |
| Package demos under `demos/` | Nested tier doc PNGs under `processing/static/results/compute_*/examples/` |
| Generated artifacts under `output/` | SFV shaping plots ([`section_loads.visualisation`](../../../load_utilities/section_loads/visualisation/README.md)) |

## Directory map

| Path | Role |
|------|------|
| [`mesh_fringe.py`](mesh_fringe.py) / [`section_axes.py`](section_axes.py) | Shared helpers |
| [`demos/`](demos/README.md) | Former top-level `examples/` scripts + catalog |
| [`output/`](output/) | Generated PNGs / CSV / study artifacts (SoT) |

## How to run demos

```bash
python blade_structure_utilities/fem_utilities/section_shell_fea/visualisation/demos/cap_tie_strategies/compare_cap_tie_strategies.py
```

See [`demos/README.md`](demos/README.md). Benchmarks under `../benchmarks/` also
write into `visualisation/output/` by default.

## See also

- [`../README.md`](../README.md)
- Geometry viz pattern: [`../../../geometry_utilities/section_geometry/visualisation/README.md`](../../../geometry_utilities/section_geometry/visualisation/README.md)
- Loads viz: [`../../../load_utilities/section_loads/visualisation/README.md`](../../../load_utilities/section_loads/visualisation/README.md)
