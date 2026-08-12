# Tertiary recovery examples

> **Synthetic / helper-level demo — not orchestrator output.**  
> These scripts apply ABD / CLPT \(\sigma(x_3)\) to hand-chosen `[N;M]` and illustrate the production station product (per-plate outer / mid / inner via `x3_ply_faces`). They are **not** artefacts of `ShellTertiaryResultsOrchestrator.compute()`.  
> Committed PNGs under `outputs/` are documentation figures. Cascade policy: [`../../docs/README.md`](../../docs/README.md). Regenerate all tiers via [`../../examples/run_all_result_demos.py`](../../examples/run_all_result_demos.py).

Runnable demos for [`compute_tertiary`](../) through-thickness (laminate **axis 3**) stations and CLPT plane-stress profiles. Not part of pytest.

## How to invoke

From the repository root:

```bash
python blade_structure_utilities/fem_utilities/section_shell_fea/processing/static/results/compute_tertiary/examples/plot_x3_recovery_demo.py
```

Requires `numpy` and `matplotlib`. Optional: `matplotlib-label-lines` for inline `P{i}` on `panel_face_semantics.png`.

## Expected artefacts

Written under [`outputs/`](outputs/):

| File | Content |
|------|---------|
| `x3_stations_isotropic.png` | Isotropic monolayer `FACE_OUTER` / `FACE_MID` / `FACE_INNER` |
| `x3_stations_composite.png` | Composite per-plate face triples (`x3_ply_faces`) |
| `sigma11_x3_isotropic_bending.png` | \(\sigma_{11}(x_3)\) under pure \(M_{11}\) with face markers |
| `sigma11_x3_composite_bending.png` | \(\sigma_{11}(x_3)\) under pure \(M_{11}\) with per-plate faces |
| `free_surfaces_vs_mids_composite.png` | Mid faces (Hashin/VM) vs free surfaces \(\pm h/2\) (extreme-fibre) |
| `panel_face_semantics.png` | Canonical alias of `panel_face_semantics_2D-CN.png` |
| `panel_face_semantics_{layout}.png` | Same figure for layouts `1C-CN`, `2C-CN`, `2D-CN`, `2D-F`, `3D-CN` (NACA 2412); RGB midlines, face markers, inline `P{i}`, panel-id legend |

These figures are linked from [`../docs/README.md`](../docs/README.md).

## See also

- Deep doc: [`../docs/README.md`](../docs/README.md)
- Station API: `LaminateDefinition.ply_stack_x3_stations()` → `PlyStackX3Stations`
