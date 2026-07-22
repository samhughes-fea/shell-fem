# Section shell load application — exclusive beam vs face traction

**Status:** Production path is classic linear-static FE under `contour_support`.  
**Scope:** MITC4 strip solves via `SectionLinearStaticRunner` / `run_section_linear_static_from_resultants`.

Related: [SHELL_MODEL.md](SHELL_MODEL.md),
[`pre_processing/loading/`](../pre_processing/loading/),
[`interface/traction_load_adapter.py`](../interface/traction_load_adapter.py),
[`simulation_runner/api/linear_static.py`](../simulation_runner/api/linear_static.py).

---

## Two load channels (exclusive ownership)

| Channel | Meaning | Mechanism |
|---------|---------|-----------|
| **A — beam resultants** | Station forces/moments \(R=(N,V_y,V_z,M_y,M_z,T)\) | `beam_resultants_to_nodal_load` → nodal \(f\) |
| **B — surface tractions** | Skin pressure/shear on the section mesh | `traction_face_loads` added to the same \(f\) |

Linear elasticity allows superposition only when the channels do not both claim the same
physical load. When face traction is applied, beam shears \(V_y\)/\(V_z\) must be zeroed
(adapter: `section_resultants_from_load_field` / `run_section_linear_static_from_load_field`).

```text
BeamResultantVector R  ──► beam_resultants_to_nodal_load ──┐
                                                           ├─► f ─► SectionLinearStaticRunner.solve()
TractionField (optional) ─► traction_face_loads ───────────┘         (always P→S→T→Q)
```

BC policy: `build_clamp_supports(..., bc_mode="contour_support")` (panel contour endpoints + RBM pin).

---

## Hard gate

Reconstruct \(R'\) from the beam portion of \(f\); components must match \(R\) within
relative/absolute \(1\mathrm{e}{-6}\). Failures indicate a load-mapper bug.

---

## Soft rules

- Do not apply tip-integrated station resultants and a simultaneous load-field overlay
  that carries the same shears (`load_field_for_dispatch` suppresses the overlay when
  explicit tip resultants win).
- Governing FI comes from quaternary results after `solve()`, not from a parallel
  panel-row FI path.
