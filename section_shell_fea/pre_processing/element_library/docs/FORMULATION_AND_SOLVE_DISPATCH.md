# Formulation registration vs external solve dispatch

Native discrete-element formulations and external global-solve backends are
**separate** registration graphs.

## Native formulations (`ShellFormulationRegistration`)

| Concern | Owner |
|---------|-------|
| Identity | `formulation_id` on `ShellFormulationDescriptor` |
| Graph | `registry._FORMULATIONS` via `register_formulation` |
| Theory default | `default_formulation_id(ShellTheory)` → one native id |
| Factory resolution | `resolve_element_factory(formulation_id=...)` or theory default |
| Validation hooks | optional `validate_element` / `validate_cache` |

CalculiX is **not** a native `ShellFormulationRegistration` and must **not**
appear in `_FORMULATIONS`.

## External solve dispatch (`shell_solve_registry`)

| Concern | Owner |
|---------|-------|
| Opt-in CCX / global-coupled backends | `pre_processing/dispatch/shell_solve_registry.py` |
| Registration | `register_calculix_ccx_backend()` / `register_global_backend` |
| Production periodic | does **not** go through this registry |

## Deprecation note

`compat-fe-element-factory-registry-facade` (`register_element_factory` /
`_ELEMENT_FACTORY_REGISTRY`) is **removed**. Removal does **not** require
CalculiX in `_FORMULATIONS`; CalculiX remains on the solve-dispatch path only.

## See also

- [`FORMULATION_VALIDATION_LIFECYCLE.md`](FORMULATION_VALIDATION_LIFECYCLE.md)
- [`../dispatch/shell_solve_registry.py`](../../dispatch/shell_solve_registry.py)
- [`../../docs/FEM_ARCHITECTURE_1DFEM_ALIGNMENT.md`](../../docs/FEM_ARCHITECTURE_1DFEM_ALIGNMENT.md)
