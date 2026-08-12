# Secondary results examples

> **Synthetic / helper-level demo — not orchestrator output.**  
> These scripts build hand-made `ShellPanelResultants` rows and call helpers such as `nest_resultant_rows_by_panel`. They are **not** artefacts of `ShellSecondaryResultsOrchestrator.compute()` (which recovers flat `resultant_rows` from primary `U_e` + the element kernel and does **not** nest internally).  
> Committed PNGs under `outputs/` are documentation figures. Cascade policy: [`../../docs/README.md`](../../docs/README.md). Regenerate all tiers via [`../../examples/run_all_result_demos.py`](../../examples/run_all_result_demos.py).

Runnable demos for [`compute_secondary`](../) panel resultants and panel nesting. Not part of pytest.

## How to invoke

From the repository root:

```bash
python blade_structure_utilities/fem_utilities/section_shell_fea/processing/static/results/compute_secondary/examples/plot_secondary_demo.py
```

Requires `numpy` and `matplotlib`. The script bootstraps the repo root onto `sys.path`.

## Expected artefacts

Written under [`outputs/`](outputs/):

| File | Content |
|------|---------|
| `panel_nesting_order.png` | Flat row order before/after `nest_resultant_rows_by_panel` |
| `resultants_Nx_by_panel.png` | \(N_x\) vs `station_index`, colored by panel |
| `resultants_Mx_by_panel.png` | \(M_x\) vs `station_index`, colored by panel |
| `qx_qy_pass_through.png` | Optional Q1/Q2 presence (grey = absent) |

These figures are linked from [`../docs/README.md`](../docs/README.md).

## See also

- Deep doc: [`../docs/README.md`](../docs/README.md)
- Tertiary consumer: [`../../compute_tertiary/docs/README.md`](../../compute_tertiary/docs/README.md)
