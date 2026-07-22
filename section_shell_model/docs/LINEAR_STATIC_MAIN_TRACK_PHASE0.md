# Phase 0 decision memo — linear-static main track (completed)

**Status:** Channel-A / Saint-Venant periodic / optimiser product surfaces **removed**.
`section_shell_model` is a pure FE package with two runners:
`SectionLinearStaticRunner` and `SectionLinearBucklingRunner`.

## Production path

```text
BeamResultantVector R → beam_resultants_to_nodal_load (± traction_face_loads)
→ SectionLinearStaticRunner.solve()  # always P→S→T→Q
→ governing Hashin from quaternary
```

Entry: `simulation_runner.api.linear_static.run_section_linear_static_from_resultants`.

BC policy: `contour_support` via `build_clamp_supports`.

## Removed

- `simulation_runner.api.periodic` / `api.k6`
- `processing/static/operations/periodic/`
- `section_optimisation` package and `optim_geometry_builder`
- Dual-run / Channel-A FI parity oracles

## Consumers

Fatigue and orchestration may call pure FE APIs from outside this package.
This package must not import or productise fatigue/optimisation.
