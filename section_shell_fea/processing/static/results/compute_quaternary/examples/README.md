# Quaternary FI examples

> **Synthetic / helper-level demo — not orchestrator output.**  
> These scripts call criterion helpers (`von_mises_plane_stress_fi`, `hashin_fi`, `extreme_fibre_fi`, etc.) on hand-chosen stress grids and strengths. They are **not** artefacts of `ShellQuaternaryResultsOrchestrator.compute()` (which consumes tertiary `sig_ply_mat` + cache).  
> Captions distinguish yield \(\sigma_y\) (Von Mises) from ultimates \(\sigma_{\mathrm{uts}}/\sigma_{\mathrm{ucs}}/\tau_{\mathrm{us}}\) (extreme-fibre).  
> Extreme-fibre free surfaces read as fluid/void (skin), skin/void (spar cap), or **fore**/**aft** (web; LE/TE sides) — see tertiary [`panel_face_semantics.png`](../../compute_tertiary/examples/outputs/panel_face_semantics.png).  
> Committed PNGs under `outputs/` are documentation figures. Cascade policy: [`../../docs/README.md`](../../docs/README.md). Regenerate all tiers via [`../../examples/run_all_result_demos.py`](../../examples/run_all_result_demos.py).

Runnable demos for [`compute_quaternary`](../) failure indices. Not part of pytest.

## How to invoke

From the repository root:

```bash
python blade_structure_utilities/fem_utilities/section_shell_fea/processing/static/results/compute_quaternary/examples/plot_fi_demo.py
```

Requires `numpy` and `matplotlib`. The script bootstraps the repo root onto `sys.path`.

## Expected artefacts

Written under [`outputs/`](outputs/):

| File | Content |
|------|---------|
| `von_mises_fi_isotropic.png` | Von Mises FI vs \(\sigma_y\) (mid face) |
| `hashin_envelope_composite.png` | Hashin envelope FI (mid face, \(\tau_{12}=0\)) |
| `hashin_mode_breakdown.png` | Four Hashin mode FIs for one stress state |
| `extreme_fibre_fi_envelope.png` | Extreme-fibre FI envelope (uts/ucs; free surfaces) |
| `vm_vs_extreme_fibre_bending.png` | Mid-face VM vs free-surface extreme-fibre under bending-like \(\sigma_{11}\) |
| `governing_track_schematic.png` | Demo argmax across Hashin / interlaminar / Von Mises / extreme-fibre |

These figures are linked from [`../docs/README.md`](../docs/README.md).

## See also

- Deep doc: [`../docs/README.md`](../docs/README.md)
- Tertiary stress recovery (inputs to quaternary): [`../../compute_tertiary/docs/README.md`](../../compute_tertiary/docs/README.md)
