# simulation_runner — public FE entrypoints

Runners that orchestrate one worker call (not a sizing/fatigue campaign).

↑ Parent: [`../README.md`](../README.md)

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

## Provenance / what happens today

| Claim | Today |
|-------|-------|
| Public entry | Package `__init__.py` exports runners + `CascadeBin` |
| Agnosticism | No sizing/fatigue purpose flags |
| Diagnostics | Structured `DiagnosticsBin` only — not campaign logs |

## See also

- [`../README.md`](../README.md)
- [`../pre_processing/loading/README.md`](../pre_processing/loading/README.md)
- [`../processing/static/README.md`](../processing/static/README.md)
