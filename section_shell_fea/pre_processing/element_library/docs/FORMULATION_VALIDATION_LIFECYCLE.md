# Formulation validation lifecycle

Optional hooks on `ShellFormulationRegistration` augment the always-on
cache-shape guard. Generic modules invoke them automatically when non-`None`.

## Timing

| Hook | When | Inputs | Failure |
|------|------|--------|---------|
| `validate_element` | After `factory.iter_elements(...)` materialises each live element, **before** stiffness/force assembly | One live element instance | Raise `ValueError` (or subclass); build aborts |
| `validate_formulation_cache_shapes` | Always, after `fill_global_dof_indices` | `ShellFormulationCache` | Raise `ValueError` |
| `validate_cache` | Immediately after shape validation, if non-`None` | `ShellFormulationCache` | Raise `ValueError`; build aborts |

`validate_cache` **augments** — it does not replace —
`validate_formulation_cache_shapes`. Hooks that are `None` are skipped (MITC4
may register concrete hooks; other formulations may leave them unset).

## Exception semantics

Hooks must fail closed with `ValueError` (or a subclass). Soft warnings are not
a substitute for invariant failure. Callers of `build_formulation_cache` rely
on exceptions to refuse a bad cache.

## Invocation points (implementation)

In `formulation_factory.build_formulation_cache`:

1. Materialise `elements = list(factory.iter_elements(...))`.
2. Resolve the registration for the cache's `formulation_id`.
3. If `registration.validate_element is not None`, call it once per element.
4. Build stiffness/force objects and construct `ShellFormulationCache`.
5. `fill_global_dof_indices` → `validate_formulation_cache_shapes(cache)`.
6. If `registration.validate_cache is not None`, call `validate_cache(cache)`.

## MITC4 invariants (when registered)

**Live element (`validate_element`):**

- `formulation_id` / element type identity is MITC4-FSDT.
- Geometry lengths `L_s`, `L_x` are finite and positive.
- Material thickness is finite and positive; ABD (or typed FSDT blocks) finite.
- Local node count is 4; `n_dof_node == 6`.

**Cache (`validate_cache`):**

- `cache.formulation_id == "MITC4-FSDT"`.
- `cache.n_dof_node == 6`.
- Every stiffness/force pair has 24 element DOFs (4×6) when global indices are set.

## See also

- [`../README.md`](../README.md) — extension recipe
- [`formulation_validation.py`](../formulation_validation.py) — always-on shapes
- [`registry.py`](../registry.py) — `ShellFormulationRegistration` hooks
