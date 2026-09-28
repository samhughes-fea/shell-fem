# Section shell load application — exclusive beam vs face traction

**Status:** Production path is classic linear-static FE under `contour_support`.  
**Scope:** MITC4 strip solves via `SectionLinearStaticRunner` / `run_section_linear_static_from_resultants`.

Related: [SHELL_MODEL.md](SHELL_MODEL.md),
[`pre_processing/loading/`](../pre_processing/loading/),
[`interface/traction_load_adapter.py`](../interface/traction_load_adapter.py),
[`simulation_runner/api/linear_static.py`](../simulation_runner/api/linear_static.py).

---

## Load channels (exclusive beam ownership)

| Channel | Meaning | Mechanism |
|---------|---------|-----------|
| **C — thin-wall shaped** | Same \(R\), distributed via thin-wall \(\sigma/q\) on midlines | `section_loads.distribute_loads(..., channel="thinwall")` → `SectionForceVector` → `section_force_vector_to_nodal_load` (**default**) |
| **A — equipollent beam resultants** | Station forces/moments \(R=(N,V_y,V_z,M_y,M_z,T)\) | `equipollent_shape_on_midlines` → `SectionForceVector` → `section_force_vector_to_nodal_load`. `beam_resultants_to_nodal_load` is regression/reference only. |
| **B — surface tractions** | Skin pressure/shear on the section mesh | `section_loads.distribute_loads(..., channel="surface_traction")` → `SectionForceVector` → `section_force_vector_to_nodal_load` |

Default: `SectionLinearStaticRunner(..., load_channel="thinwall", section_force_vector=sfv)`.
Opt-in `"equipollent"` or `"surface_traction"` still apply a **frozen SFV**; FE does
not select a load-shaping theory. Channel C is a **load-shape** path only (not shell
FI / ULS). Kernels live under
`blade_structure_utilities.load_utilities.section_loads.{thinwall,equipollent,surface_traction}`.

Linear elasticity allows superposition only when the channels do not both claim the same
physical load. When face traction is applied as Channel B, do **not** also apply tip-integrated
beam shears \(V_y\)/\(V_z\) that already include those forces.
Channel A and Channel C are alternative beam mappers — never both.

```text
BeamResultantVector R  ──► distribute_loads(thinwall) → SFV ──┐
                     └──► equipollent_shape_on_midlines → SFV ├─► section_force_vector_to_nodal_load → f_station
TractionField (B only) ──► surface_traction → SFV ────────────┘
f_total = f_element + f_station  ─► constrain → reduce → solve ─► P→S→T→Q
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
