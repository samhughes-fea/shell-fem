# processing/static — assemble, solve, cascade

Linear-static operations and the P→S→T→Q results cascade.

↑ Parent: [`../../README.md`](../../README.md)

## Purpose / ownership

| Owns | Does **not** own |
|------|------------------|
| Assemble \(\mathbf{K}\), \(\mathbf{f}\); apply BCs; solve \(F=KU\) | SFV shaping / nodal projection policy details ([`../../pre_processing/loading/`](../../pre_processing/loading/README.md)) |
| Cascade bins primary → quaternary | Campaign sizing loops / rainflow |
| Per-solve `DiagnosticsBin` phase timings | Station log file writers (orchestration) |

---

## Mathematics

After element `Ke`/`Fe` are formed:

1. **Assemble** global \(\mathbf{K}\), \(\mathbf{f}\).
2. **Constrain** essential BCs.
3. **Solve** \(\mathbf{K}\mathbf{U}=\mathbf{F}\).
4. **Recover** through the cascade:

\[
\mathrm{P}\ (U_e)
\;\rightarrow\;
\mathrm{S}\ (N,M,Q)
\;\rightarrow\;
\mathrm{T}\ (\sigma_{\mathrm{ply}})
\;\rightarrow\;
\mathrm{Q}\ (\mathrm{FI}).
\]

Tier ownership and frame ladder:
[`results/docs/README.md`](results/docs/README.md).

---

## Exact I/O

| Path | Role |
|------|------|
| [`operations/`](operations/) | Assemble / constrain / solve |
| [`results/`](results/) | Cascade compute_* tiers + bin types |
| [`diagnostics/`](diagnostics/) | GCI, trust, loading/mesh plots, timings |

Public product is the cascade bin returned by
[`../../simulation_runner/`](../../simulation_runner/README.md).

---

## Provenance / what happens today

| Claim | Today |
|-------|-------|
| Cascade index | [`results/docs/README.md`](results/docs/README.md) |
| Tier demos | Nested `results/compute_*/examples/` (committed PNG docs) |
| Package demos | [`../../visualisation/demos/`](../../visualisation/demos/README.md) |

## See also

- [`../../README.md`](../../README.md)
- [`results/docs/README.md`](results/docs/README.md)
- [`../../docs/SECTION_SOLVE_STAGES.md`](../../docs/SECTION_SOLVE_STAGES.md)
- [`../buckling/`](../buckling/) — eigen / prestress track (sibling)
