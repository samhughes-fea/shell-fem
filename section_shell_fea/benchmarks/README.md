# benchmarks — GCI / runtime / CCX campaigns (not a stage home)

↑ Parent: [`../README.md`](../README.md) · Computable set: [`../README.md` §2](../README.md#2-computable-set) ·
Glossary: [`../../../docs/CASCADE_GLOSSARY.md`](../../../docs/CASCADE_GLOSSARY.md) · Narrative:
[`../docs/NUMERICAL_STABILITY_AND_GCI.md`](../docs/NUMERICAL_STABILITY_AND_GCI.md)

Runnable study drivers (grid independence, defect-K sweeps, cap-peak sampling,
CCX quarantine validation, runtime A/B). Outputs often land under
`benchmarks/output/` and curated proof PNGs under
[`../visualisation/output/`](../visualisation/output/).

## Purpose / ownership

| Owns | Does **not** own |
|------|------------------|
| Campaign scripts (`run_grid_independence_study.py`, …) | Production sizing orchestration (BSP) |
| Local study artifacts | MITC4 formulation ([`../pre_processing/element_library/`](../pre_processing/element_library/README.md)) |

## Proof figures (committed)

| PNG | Claim |
|-----|-------|
| [`../visualisation/output/gci_study_mitc4/plots/ladder_governing_fi.png`](../visualisation/output/gci_study_mitc4/plots/ladder_governing_fi.png) | Governing FI vs refinement |
| [`../visualisation/output/gci_study_mitc4/plots/ladder_d_inf.png`](../visualisation/output/gci_study_mitc4/plots/ladder_d_inf.png) | Displacement ladder |
| [`../visualisation/output/gci_study_mitc4/plots/layouts/{A,B,C,D}/`](../visualisation/output/gci_study_mitc4/plots/layouts/) | Per-layout section geometry + CI ladder (40 keys, by family) |
| [`../visualisation/output/gci_study_mitc4/plots/ci_watch_layouts/{A,B,C,D}/`](../visualisation/output/gci_study_mitc4/plots/ci_watch_layouts/) | CI-watch subset composites |
| [`../visualisation/output/gci_study_ccx/plots/layouts/{A,B,C,D}/`](../visualisation/output/gci_study_ccx/plots/layouts/) | CCX membrane oracle geometry + `|N12|` ladders |

Regen from CSVs (no re-solve):

```powershell
python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.plot_gci_study_outputs --study-dir ../visualisation/output/gci_study_mitc4 --backend-label MITC4
python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.plot_gci_study_outputs --study-dir ../visualisation/output/gci_study_ccx --backend-label CCX --no-hashin
```

CalculiX paths stay quarantined under [`../calculix_ccx/`](../calculix_ccx/).

## See also

- [`../docs/README.md`](../docs/README.md)
- [`../visualisation/demos/README.md`](../visualisation/demos/README.md)
