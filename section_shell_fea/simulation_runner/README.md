# simulation_runner — public FE entrypoints

Runners that orchestrate one worker call (not a sizing/fatigue campaign).

↑ Parent: [`../README.md`](../README.md) · Docs index: [`../docs/README.md`](../docs/README.md)

## Purpose / ownership

| Owns | Does **not** own |
|------|------------------|
| `SectionLinearStaticRunner` / `SectionLinearBucklingRunner` | Thickness bisection / spanwise monotonicity |
| `execute()` → cascade bin handoff | SFV shaping, laminate bind policy |
| Optional tier gates as compute-cost knobs | `purpose="sizing"|"fatigue"` modes |

---

## Mathematics

A linear-static run is the composition of homes 1–3:

\[
(\mathrm{zones},\,\mathrm{SFV},\,\mathrm{BC})
\;\xrightarrow{\mathrm{apply}}\;
\mathbf{f}
\;\xrightarrow{\mathrm{Ke}+\mathrm{assemble}}\;
\mathbf{KU}=\mathbf{F}
\;\xrightarrow{\mathrm{cascade}}\;
\texttt{CascadeBin}.
\]

Buckling runs add prestress → eigen extraction on the same mesh/constitutive
setup. No campaign state is stored inside the runner.

---

## Exact I/O

### `SectionLinearStaticRunner(...).execute().get_results() -> CascadeBin`

| In | `panel_topology` (mesh) + bound `MaterialZone`s, resultants and/or `section_force_vector`, BCs |
| Out | `CascadeBin` (`SectionLinearStaticResult`) including `diagnostics` |

| Path | Role |
|------|------|
| [`section_linear_static_runner.py`](section_linear_static_runner.py) | Production static runner |
| [`section_linear_buckling_runner.py`](section_linear_buckling_runner.py) | Buckling runner |
| [`api/`](api/) | Thin functional wrappers |

Barrel re-exports these from `section_shell_fea`.

---

## PNG pointers

| Figure | Meaning |
|--------|---------|
| Mesh input | [`../visualisation/output/section_centerline_mesh_v2/section_mesh_2C-F_i000.png`](../visualisation/output/section_centerline_mesh_v2/section_mesh_2C-F_i000.png) |
| Q Hashin | [`../processing/static/results/compute_quaternary/examples/outputs/hashin_envelope_composite.png`](../processing/static/results/compute_quaternary/examples/outputs/hashin_envelope_composite.png) |
| GCI governing FI | [`../visualisation/output/gci_study_mitc4/plots/ladder_governing_fi.png`](../visualisation/output/gci_study_mitc4/plots/ladder_governing_fi.png) |

## Provenance / what happens today

| Claim | Today |
|-------|-------|
| Public entry | Package `__init__.py` exports runners + `CascadeBin` |
| Agnosticism | No sizing/fatigue purpose flags |
| Diagnostics | Structured `DiagnosticsBin` only — not campaign logs |
| Hashin FI tier | Quaternary on static runner product |

## See also

- [`../README.md`](../README.md)
- [`../pre_processing/loading/README.md`](../pre_processing/loading/README.md)
- [`../processing/static/README.md`](../processing/static/README.md)
