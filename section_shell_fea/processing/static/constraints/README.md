# processing/static/constraints — junction MPCs, cap ties, supports

Cap ties are **rigid-offset MPCs**, not elements. The element library stays
limited to element-local `Ke`/`Fe`.

## Sequence

```text
frozen mesh + bound zones + frozen SFV
    → assemble K and element RHS once
    → compose one immutable constraint set
    → f_total = f_element + f_station
    → constrain → reduce → solve
    → P → S → T → Q
```

Composer order in `compose.build_section_constraint_set`:

1. junction equations from topology/metadata
2. essential supports (`fixed_dofs`)
3. cap-tie equations, expanding any junction-slave masters
4. validate / coalesce
5. freeze sorted equations, fixed DOFs, and `SectionConstraintReport`

## Ownership

| Owns | Does not own |
|------|----------------|
| Junction MPCs, cap-tie geometry + rigid-offset equations, supports, composition, provenance | Element `Ke`/`Fe`, global scatter-add, SFV shaping, core P→S→T→Q filtering |
| Typed `ConstraintEquation` / `SectionConstraintSet` | Solver dictionary (adapter only, at constrain/reduce) |

Constraint metadata may tag tie slaves and junction ownership. Diagnostics /
governing policy own any exclusion. Core recovery computes all supported
element results regardless of those tags.

`transformed_basis` is diagnostic-only (non-periodic). Periodic transformed-basis
K6 is ruled out.

## CCX

Share only backend-neutral projection/discovery with `calculix_ccx/cap_skin_ties.py`.
Node/surface DTOs, `*TIE` emission, and indexing stay CCX-owned.
