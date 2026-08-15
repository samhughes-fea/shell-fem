# visualisation — plot helpers and demos

Operational suite for the `section_shell_fea` worker: matplotlib helpers and
figure-producing demos. FE mathematics live in loading / element_library /
processing / simulation_runner.

↑ Parent: [`../README.md`](../README.md) · Docs index: [`../docs/README.md`](../docs/README.md)

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

## Artefacts ↔ stages

| `output/` tree | Stage / claim |
|----------------|---------------|
| [`section_centerline_mesh_v2/`](output/section_centerline_mesh_v2/) | Prep mesh / panel topology |
| [`gci_study_mitc4/`](output/gci_study_mitc4/) | Static solve + Q under refinement (GCI) |
| [`gci_defect_k_shared/`](output/gci_defect_k_shared/), [`gci_defect_k_tb/`](output/gci_defect_k_tb/) | Defect-K junction studies |
| [`gci_study_ccx/`](output/gci_study_ccx/), [`gci_study_mitc4_vs_ccx/`](output/gci_study_mitc4_vs_ccx/) | Quarantined CCX validation |
| [`cap_peak_sampling/`](output/cap_peak_sampling/) | Cap-peak diagnostics (CSV/JSON) |
| [`benchmark_shell_elements/`](output/benchmark_shell_elements/) | Element / FI oracle comparison |

Docs link these in place — do not regenerate for documentation upgrades.

## How to run demos

```bash
python blade_structure_utilities/fem_utilities/section_shell_fea/visualisation/demos/cap_tie_strategies/compare_cap_tie_strategies.py
```

See [`demos/README.md`](demos/README.md). Benchmarks under `../benchmarks/` also
write into `visualisation/output/` by default.

## See also

- [`../README.md`](../README.md)
- [`../docs/README.md`](../docs/README.md)
- [`../docs/NUMERICAL_STABILITY_AND_GCI.md`](../docs/NUMERICAL_STABILITY_AND_GCI.md)
- Geometry viz pattern: [`../../../geometry_utilities/section_geometry/visualisation/README.md`](../../../geometry_utilities/section_geometry/visualisation/README.md)
- Loads viz: [`../../../load_utilities/section_loads/visualisation/README.md`](../../../load_utilities/section_loads/visualisation/README.md)
