# Element-agnostic shell formulation architecture

For layer-by-layer fidelity vs 1d-fem see [FEM_ARCHITECTURE_1DFEM_ALIGNMENT.md](FEM_ARCHITECTURE_1DFEM_ALIGNMENT.md).
Product surface: [LINEAR_STATIC_MAIN_TRACK_PHASE0.md](LINEAR_STATIC_MAIN_TRACK_PHASE0.md).

**Last updated:** 2026-07-22 — Phase 0: Channel-A / `api.periodic` / `api.k6` /
`operations/periodic/` **removed**. Two runners only. Full-section buckling is the
default buckling track. HSDT / clamped FE deleted; FSDT-only production; CalculiX
opt-in via `register_calculix_ccx_backend()`. Package-level `api/` dissolved.

## 1d-fem mapping (current)

| 1d-fem | `section_shell_fea` | Fidelity |
|--------|------------------------|----------|
| `Element1DBase` | `ShellElementBase` | Direct |
| `ElementObject` / `ForceObject` | `ShellStiffnessObject` / `ShellForceObject` | Direct |
| `FormulationResultSet` | `ShellFormulationCache` | Direct |
| `ELEMENT_CLASS_MAP` | `resolve_element_factory` | Direct (mechanism differs) |
| `AssembleGlobalSystem` | `assemble_section_system` / stage `AssembleGlobalShellSystem` | Direct |
| `LinearStaticSimulationRunner` | `SectionLinearStaticRunner` | Direct |
| `LinearBucklingSimulationRunner` | `SectionLinearBucklingRunner` (full-section `K`/`Kg` default) | Direct |
| Stage operations | `processing/static/operations/` stage ops (+ assembly orchestra) | Direct |
| Solve-state DTOs | `processing/static/operations/solve_states.py` | Direct |
| Results tiers | P→S→T→**Q** | Quaternary is blade extension |
| Diagnostics | `processing/static/diagnostics/` | Direct |

## Layer rules

1. **`pre_processing/`** — prepares runner inputs (`mesh/`, `loading/`, `element_library/`, `constitutive/`, `containers/`, `dispatch/`). Per-domain `*/api/` where needed. Does **not** orchestrate solves.
2. **`simulation_runner/`** — orchestrates ops + result computers; owns solve state; records diagnostics. APIs: `linear_static`, `linear_buckling`, `station_recovery` only.
3. **`processing/`** — computes stage ops, assembly orchestra, results, diagnostics implementations.
4. **`interface/`** — cross-package adapters only.

**Separation:** pre_processing *prepares*; processing *computes*; simulation_runner *sequences and records*. No package-level `api/` façade.

## Production flows (FSDT)

**Linear-static FI:**

```text
mesh + formulation cache + beam_resultants_to_nodal_load(R)
  → SectionLinearStaticRunner.from_cache(...).solve(f_override=f)
  → P → S → T → Q → governing Hashin
```

**Buckling (default full-section):**

```text
static cascade (same mesh / K / T / secondary prestress)
  → assemble_section_geometric_stiffness
  → solve_section_linear_buckling_eigen  (K_r φ = λ (−Kg_r) φ)
  → P → S → fold_station_buckling_bundle
```

Panel-classical strip eigen remains available via `eigen_track="panel"`.

**Deleted (do not revive):** Channel-A / Saint-Venant periodic product surfaces,
`engine/`, package-level `api/`, HSDT, legacy clamped MITC4 monoliths.

## Element factory registry

`register_element_factory` / `resolve_element_factory` in
`pre_processing/element_library/formulation_factory.py`. FSDT→MITC4 registers at
import; CalculiX remains opt-in and must not register the element factory at
import time.
