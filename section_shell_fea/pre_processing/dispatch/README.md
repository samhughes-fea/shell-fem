# dispatch — theory / solve registries

Registration seam for shell theories and global backends.

↑ Parent: [`../README.md`](../README.md) · Package: [`../../README.md`](../../README.md)

## Purpose / ownership

| Owns | Does **not** own |
|------|------------------|
| Station / global callable registries | Element kernels, cascade maths |
| Opt-in backend hooks (e.g. CCX register) | Production orchestration modes |
| Theory → solver dispatch tables | Capability *flags* (owned by containers) |

Capability flags live in
[`../containers/shell_theory_capabilities.py`](../containers/shell_theory_capabilities.py);
this home stores **callables** once a theory opts in.

---

## Mathematics / contract

Dispatch is identity on the physics: it selects which registered callable
implements the same FSDT / validation path. Production default:

\[
\mathrm{ShellTheory.FSDT\ (MITC4)}
\;\mapsto\;
\texttt{SectionLinearStaticRunner}\ \text{track}.
\]

CalculiX is **not** auto-registered; callers must
`register_calculix_ccx_backend()` ([`../../calculix_ccx/`](../../calculix_ccx/README.md)).

---

## Exact I/O

| Path | Role |
|------|------|
| [`shell_theory_registry.py`](shell_theory_registry.py) | Unified theory registration (`ShellTheoryRegistration`) |
| [`shell_solve_registry.py`](shell_solve_registry.py) | Global solve backend map |
| [`station_theory_dispatch.py`](station_theory_dispatch.py) | Station-level theory dispatch |

| In | `ShellTheory` enum + callables |
| Out | Resolved station / global solve functions |

```python
from blade_structure_utilities.fem_utilities.section_shell_fea.calculix_ccx.register import (
    register_calculix_ccx_backend,
)
register_calculix_ccx_backend()  # opt-in only
```

---

## PNG pointers

| Figure | Meaning |
|--------|---------|
| MITC4 vs CCX GCI (validation) | [`../../visualisation/output/gci_study_mitc4_vs_ccx/`](../../visualisation/output/gci_study_mitc4_vs_ccx/) when present |
| Production GCI (MITC4) | [`../../visualisation/output/gci_study_mitc4/plots/ladder_governing_fi.png`](../../visualisation/output/gci_study_mitc4/plots/ladder_governing_fi.png) |

---

## Provenance / what happens today

| Claim | Today |
|-------|-------|
| Production theory | MITC4 via static runner (not CCX) |
| CCX quarantine | No import at module load; explicit register |
| Channel-A periodic | Removed (Phase 0) |

## See also

- [`../../calculix_ccx/README.md`](../../calculix_ccx/README.md)
- [`../containers/README.md`](../containers/README.md)
- [`../../docs/ELEMENT_AGNOSTIC_ARCHITECTURE.md`](../../docs/ELEMENT_AGNOSTIC_ARCHITECTURE.md)
- [`../../docs/CALCULIX_CCX_ELEMENT_LIBRARY_RESTRUCTURE.md`](../../docs/CALCULIX_CCX_ELEMENT_LIBRARY_RESTRUCTURE.md)
