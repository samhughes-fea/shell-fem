# Element-agnostic shell formulation architecture

See [`section_optimisation/docs/ELEMENT_AGNOSTIC_REFACTOR.md`](../../section_optimisation/docs/ELEMENT_AGNOSTIC_REFACTOR.md) for the optimiser-side analogue.

For the physical `pre_processing/` package layout (containers, element_library, mesh, loading), `processing/diagnostics`, the FSDT registry (Phase 7), and the CalculiX quarantine (Phase 6), see [PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md](PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md). For the full layer-by-layer fidelity evaluation against 1d-fem's computational philosophy, see [FEM_ARCHITECTURE_1DFEM_ALIGNMENT.md](FEM_ARCHITECTURE_1DFEM_ALIGNMENT.md).

**Last updated:** 2026-07-14 — HSDT package deleted (not an opt-in backend); production theories are FSDT-only (`SUPPORTED_SHELL_THEORY_BY_REGIME[LINEAR] = (FSDT,)`); CalculiX remains opt-in validation via `register_calculix_ccx_backend()`. Clamped FE (`api/legacy_clamped.py`) deleted. Phase 7 registry and Phase 5 stage relocation are landed.

## 1d-fem mapping (current)

| 1d-fem | `section_shell_model` | Fidelity |
|--------|------------------------|----------|
| `Element1DBase` | `pre_processing/element_library/shell_element_base.py` → `ShellElementBase` | Direct |
| `ElementObject` / `ForceObject` | `pre_processing/containers/formulation_objects.py` → `ShellStiffnessObject` / `ShellForceObject` | Direct |
| `FormulationResultSet` | `pre_processing/containers/` → `ShellFormulationCache` | Direct |
| `ELEMENT_CLASS_MAP` (registry) | `pre_processing/element_library/formulation_factory.py` → registry-based `resolve_element_factory` | Direct (Phase 7 landed) |
| `AssembleGlobalSystem` | `processing/operations/assemble_global.py` → `assemble_section_system` (re-exported from `engine/section_assembly.py`) | Direct |
| `LinearStaticSimulationRunner` | `simulation_runner/section_linear_static.py` → `SectionLinearStaticRunner` | Direct |
| Stage operations | `processing/operations/` (one file per stage; `engine/section_solve/` shim deleted) | Direct |
| Solve-state DTOs (`GlobalResults`, `IndexMapSet`, …) | `processing/operations/solve_states.py` | Direct (relocated) |
| Results tiers | `processing/results/` (primary / secondary / tertiary / quaternary × resolution) | Direct — quaternary is a blade_precompute extension of 1d-fem's three tiers, scoped strictly to Hashin + Von Mises FI ([PROCESSING_ARCHITECTURE_PROPOSAL.md](PROCESSING_ARCHITECTURE_PROPOSAL.md)) |
| `processing/static/diagnostics/` | `processing/diagnostics/` | **Direct** |
| `pre_processing/containers/` (1d-fem's own, legacy) | — | **No longer claimed as the source of the direct analogy.** 1d-fem's active formulation cache lives in `element_library/gauss_point_data.py`; see [FEM_ARCHITECTURE_1DFEM_ALIGNMENT.md](FEM_ARCHITECTURE_1DFEM_ALIGNMENT.md) |

## Layer rules

1. **`pre_processing/containers/` + `pre_processing/element_library/`** — agnostic DTOs, formulation physics, the element factory registry. Solve-state DTOs (`GlobalShellSystem`, `ReducedShellSystem`, …) are **not** here — see rule 3. Load-channel request DTOs live in `pre_processing/containers/load_channel_types.py`.
2. **`engine/`** — mesh maps, node numbering, constraints, homogenisation, assembly orchestration, and the shared periodic BVP reduction (`engine/periodic_reduced_bvp.py`) used by both K6 modes and face traction (mixed pre/post-solve spine; no direct 1d-fem counterpart — long-term direction in [FEM_ARCHITECTURE_1DFEM_ALIGNMENT.md](FEM_ARCHITECTURE_1DFEM_ALIGNMENT.md) is assembly → `processing/operations`, mesh maps → `pre_processing/mesh`, homogenisation → recovery/processing).
3. **`processing/`** — solve-state DTOs (`operations/solve_states.py`), solve operations (`operations/`), diagnostics (`diagnostics/`), results taxonomy (`results/`), and optional formulation-physics protocols (`operations/physics_contract.py`; production uses the linear contract only). Diagnostics may call kernels only through `resolve_kernel` (import-boundary guard).
4. **`simulation_runner/`** — `SectionLinearStaticRunner` **orchestrates** the linear stage pipeline and owns threaded state (cache, K, f, T, fixed, factor, index maps). Production regime is linear-only (`ShellAnalysisRegime.NONLINEAR` empty / out of scope).
5. **`api/`** — thin public wrappers delegating to runner + recovery. Production dispatch (`api/periodic.py`, `engine/shell_solve_registry.py`, `engine/station_theory_dispatch.py`) registers only FSDT by default; CalculiX registers via the explicit opt-in hook `register_calculix_ccx_backend()` — see Phase 6 of [PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md](PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md).

**Generalised section BVP:** beam resultants and face traction are **load channels** of one
`PeriodicReducedBVP`, not two independent solves. Swapping element formulations must not
introduce a second free set — see [LOAD_APPLICATION_DOUBLE_COUNTING.md](LOAD_APPLICATION_DOUBLE_COUNTING.md).

## Production flow (FSDT — the only default-registered theory)

```text
panel mesh maps
  → build_formulation_cache(..., shell_theory=ShellTheory.FSDT)   # resolves Mitc4ElementFactory via the registry
  → SectionLinearStaticRunner.from_cache(...) or assemble_section_system + from_global
  → runner.solve()  [prepare → assemble → constrain → reduce → solve → reconstruct → compute_primary]
  → secondary/tertiary/quaternary results (opt-in via processing/results/)
```

Production callers use `SectionLinearStaticRunner.from_global(...).solve()` (or `from_cache`)
directly. The former `reduce_propagate_solve` thin adapter was removed (see
[DEPRECATED_ALIASES.md](DEPRECATED_ALIASES.md)).

**Deleted legacy paths (do not revive):** `panel_mitc4_model.solve_panel_mitc4`, thin-wall MITC4
orchestrator wrappers, and `api/legacy_clamped.py` / `run_section_with_mitc4_shell`. Production is
periodic-only via `run_section_periodic_bc`.

## Element factory registry (Phase 7 — landed)

`pre_processing/element_library/formulation_factory.py` exposes a registration map keyed by `ShellTheory`, the direct analogue of 1d-fem's `ElementFactory.ELEMENT_CLASS_MAP`:

```python
_ELEMENT_FACTORY_REGISTRY: dict[ShellTheory, type[ElementFactory]] = {}

def register_element_factory(theory: ShellTheory, factory_cls: type[ElementFactory]) -> None:
    _ELEMENT_FACTORY_REGISTRY[theory] = factory_cls

def resolve_element_factory(theory: ShellTheory) -> ElementFactory:
    try:
        factory_cls = _ELEMENT_FACTORY_REGISTRY[theory]
    except KeyError:
        raise ValueError(f"No element factory registered for {theory!r}")
    return factory_cls()

# FSDT registers unconditionally — the only production default theory
register_element_factory(ShellTheory.FSDT, Mitc4ElementFactory)
```

CalculiX does **not** call `register_element_factory` (it is an external subprocess, not a native
kernel). It registers dispatch callables only when `calculix_ccx/register.py::register_calculix_ccx_backend()`
runs. The former HSDT package and `register_hsdt_backend()` have been **deleted**.

### Adding a kernel (registry-based)

1. Implement `XxxQuad4Element(ShellElementBase)` returning stiffness/force objects, under a new package in `pre_processing/element_library/`.
2. Implement `XxxElementFactory` and call `register_element_factory(ShellTheory.XXX, XxxElementFactory)` — either unconditionally at import time if the theory is a new production default, or inside an opt-in `register_xxx_backend()` hook if it is validation-only (the CCX pattern for *dispatch*, not the formulation factory).
3. Add a `ShellTheory` capability row in `pre_processing/containers/shell_theory_capabilities.py` (`SUPPORTED_SHELL_THEORY_BY_REGIME` for production defaults, `EXPERIMENTAL_SHELL_THEORIES_BY_REGIME` for opt-in theories).
4. Extend the import-boundary test allowlist only if a registry shim is required.
5. **No edits to `formulation_factory.py` itself are required** — this is the point of the registry: `resolve_element_factory` is a pure lookup, not a per-theory branch.

## Staged solve (production)

`processing/operations/` decomposes the solve into educational stages (`PrepareLocalShellSystem` … `DisassembleGlobalShellField`), operating on solve-state DTOs from `processing/operations/solve_states.py`. `SectionLinearStaticRunner` threads state and returns a typed result plus `legacy_dict()`. The former `engine/section_solve/` re-export shim is **deleted**.

`PrimaryShellResults.r_full` is always **`K @ d_global − f_global`** (legacy contract). Masked support reactions and free-DOF residual are separate provenance fields — they must **not** be summed to form `r_full`.

Opt-in: `SHELL_DOF_PROVENANCE_DUMP=1`, `SHELL_FORMULATION_CACHE_STRICT_SHAPE=1`. See `processing/results/RESULTS_DESIGN.md`.

## Theory support (current contract)

| Theory | Status | How selected |
|--------|--------|--------------|
| **FSDT (MITC4)** | Production default | Registered at import; sole member of `SUPPORTED_SHELL_THEORY_BY_REGIME[LINEAR]` |
| **CalculiX CCX** | Validation-only, opt-in | Enum + capability row retained; call `register_calculix_ccx_backend()` before theory-dispatched routing. Hybrid MITC4 energy `K6` + CCX S4R stresses — see [PERIODIC_CCX_HOMOGENIZATION.md](PERIODIC_CCX_HOMOGENIZATION.md) |
| **HSDT** | **Removed** | Package, `ShellTheory.HSDT`, and dispatch hooks deleted. `ProvenanceKind.HSDT` retained only as vocabulary for old artefacts |

Promoting CalculiX (or any new theory) to a production default requires an explicit scoped milestone — track in `docs/architecture/REFACTOR_BACKLOG_blade_precompute.md`; do not promote without one.
