# Element-agnostic shell formulation architecture

For layer-by-layer fidelity vs 1d-fem see [FEM_ARCHITECTURE_1DFEM_ALIGNMENT.md](FEM_ARCHITECTURE_1DFEM_ALIGNMENT.md).
Product surface: [LINEAR_STATIC_MAIN_TRACK_PHASE0.md](LINEAR_STATIC_MAIN_TRACK_PHASE0.md).

**Last updated:** 2026-09-20 — formulation registry, canonical projection, honest Hashin.
`operations/periodic/` **removed**. Two runners only. Full-section buckling is the
default buckling track. HSDT / clamped FE deleted; FSDT-only production; CalculiX
opt-in via `register_calculix_ccx_backend()`. Worker `api/` barrel is retained as
the `API_STANDARDS` seam; the unused `types/` barrel was removed because it
shadowed stdlib `types` when tools ran from the package directory.

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

**Separation:** pre_processing *prepares*; processing *computes*; simulation_runner *sequences and records*. Worker `api/` re-exports the package `__all__` and is not a second sequencer.

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
`engine/`, unused `types/` barrel, HSDT, legacy clamped MITC4 monoliths.
Worker `api/` remains the public import seam.

## Element factory registry

Discrete formulations register in
`pre_processing/element_library/registry.py` (keyed by formulation ID).
`FSDT → MITC4-FSDT` is the default mapping. MITC4 self-registers lazily from
`mitc4/registration.py`; generic factory modules do not import MITC4 at module
load. CalculiX remains an opt-in backend and is not a native formulation.

Canonical result projection is owned by each registration's
`NativeToCanonicalShellProjector`. Frame mathematics stays in
`geometry_utilities.coordinate_frames`.

See [FORMULATION_CONTRACT.md](FORMULATION_CONTRACT.md).
