# containers — FEA product DTOs

FE-local carriers (station / formulation / recovery / buckling / theory caps).

↑ Parent: [`../README.md`](../README.md) · Package: [`../../README.md`](../../README.md)

Prefer package-barrel imports from consumers.

## Purpose / ownership

| Owns | Does **not** own |
|------|------------------|
| `ShellPanelResultants`, provenance kinds | Cascade bin compute (processing results) |
| Station / recovery / buckling type bags | Upstream contract DTOs (`BeamResultantVector`, `ShellMidlineStrip`) |
| `ShellTheory` / capability registry tables | Element kernels / SFV shaping |

---

## Mathematics / contracts

Secondary resultants stored in laminate **E** (\(x_1,x_2,x_3\)):

\[
\mathbf{N}=[N_{11},N_{22},N_{12}],
\quad
\mathbf{M}=[M_{11},M_{22},M_{12}],
\quad
\mathbf{Q}=[Q_1,Q_2].
\]

Each field carries `FieldProvenance`; Hashin FI gates on
`provenance_is_trusted` (placeholders must not govern). Units: \(N,Q\) in N/m;
\(M\) in N·m/m.

---

## Exact I/O

| Path | Role |
|------|------|
| [`recovery_types.py`](recovery_types.py) | `ShellPanelResultants`, `FieldProvenance`, recovery point sets |
| [`station_types.py`](station_types.py) / [`station_analysis_types.py`](station_analysis_types.py) | Station geometry / analysis bags |
| [`buckling_types.py`](buckling_types.py) | Buckling input / assessment / station bundle |
| [`formulation_objects.py`](formulation_objects.py) | Formulation cache objects |
| [`shell_theory_capabilities.py`](shell_theory_capabilities.py) | `ShellTheory`, capability flags |
| [`fem_config.py`](fem_config.py) | FE config knobs |
| [`types.py`](types.py) | Compatibility facade re-export |
| [`load_channel_types.py`](load_channel_types.py) | Load-channel related FE locals |

| In | Assembly / recovery producers write these DTOs |
| Out | Typed handoffs into cascade tiers and buckling |

---

## PNG pointers

| Figure | Meaning |
|--------|---------|
| Secondary resultants by panel | [`../../processing/static/results/compute_secondary/examples/outputs/resultants_Nx_by_panel.png`](../../processing/static/results/compute_secondary/examples/outputs/resultants_Nx_by_panel.png) |
| E-frame GCI peaks | [`../../docs/NUMERICAL_STABILITY_AND_GCI.md`](../../docs/NUMERICAL_STABILITY_AND_GCI.md) |

---

## Provenance / what happens today

| Claim | Today |
|-------|-------|
| Production theory enum | MITC4 / FSDT supported; `CALCULIX_CCX` quarantined from regime select |
| Prefer leaf imports | Import from `recovery_types` etc.; `types.py` is a facade |

## See also

- [`../../docs/TAXONOMY.md`](../../docs/TAXONOMY.md)
- [`../../processing/static/results/docs/README.md`](../../processing/static/results/docs/README.md)
- [`../dispatch/README.md`](../dispatch/README.md)
