# pre_processing.loading

Per-station surface traction load fields for `section_shell_model`. Turns extreme
distributed line loads `(q_y, q_z, m_x)` plus XFOIL-weighted pressure/shear
coefficients into a B-frame traction field on the section skin.

Formerly a top-level `section_loads` package; now owned here under
`section_shell_model.pre_processing.loading`.

---

## Pipeline position

```
section_geometry  (ShellMeshInputs: closed B-frame skin ring)
        │
        ▼
pre_processing.loading   ←── this module
        │  SectionStationLoadField (TractionField + provenance)
        │  → PNGs / serialization (audit; not default production FI RHS)
        ▼
section_shell_model simulation_runner
        │  Production FI: tip-integrated R(z) → beam_resultants_to_nodal_load
        │  Optional: face traction via run_section_linear_static_from_load_field
        ▼
```

**Production shell policy (beam-only by default):** the orchestration shell stage
hands off tip-integrated cantilever resultants
\(R(z)=(N,V_y,V_z,M_y,M_z,T)\) and does **not** pass `station_load_fields` into
the solve. This module still builds TractionFields for load-field PNGs and audit
artifacts.

**Standalone API** (`run_section_linear_static_from_load_field`): when the load
field has a usable TractionField, beam shears are zeroed (`Vy=Vz=0`) and traction
owns strip force (`T=m_x` remains on the beam resultant). Both are solved on the
contour-supported linear-static RHS. Adapter helpers live in
`section_shell_model.interface.traction_load_adapter`.

> Fatigue \(L_\mathrm{rec} + L_\mathrm{rec,tr}\) composition can still double-count if
> callers do not use the residual partition. See
> [LOAD_APPLICATION_DOUBLE_COUNTING.md](../../docs/LOAD_APPLICATION_DOUBLE_COUNTING.md).

---

## Layout

```
loading/
├── api/build_station_load_field.py
├── builders/          # incident_aero, outline_frames, station_load_builder
├── thin_wall/         # Vlasov / sectorial warping (CCX / diagnostic prep)
├── beam_resultants_to_nodal_load.py
├── traction_face_load.py
├── shear_centre_loading_axis.py
├── vis.py
└── tests/
```

---

## Methodology

### Workflow (per station)

1. Receive `DistributedStationLoadSpec` (orchestration builds this from `PrecomputeInputs` columns).
2. Read the closed B-frame skin ring from `ShellMeshInputs` (same concat rule as meshing).
3. Derive `Re`, `Ma`, `α` from `SectionLoadsJobConfig` + `StationAeroContext`.
4. Run XFOIL (or synthetic uniform Cp/Cf in test mode) → panel pressure/shear coefficients.
5. Scale pressure/shear panel forces to match `(q_y, q_z)` via LoadMatcher (`m_x` is stored in
   `distributed_source` for shell handoff but is **not** reconstructed aerodynamically).

Loads come from resampled `PrecomputeInputs` columns. For standalone DAT loading use
`blade_precompute.beam_resultants` / orchestration load helpers (the former
`BladeDesignProblem.load_extreme_loads_dat` optimiser API is removed).

### Sign convention

Effective angle of attack (default derivation):

```
alpha_eff = pitch_deg + inflow_phi_deg - twist_deg
```

Relative speed uses a minimal TSR model: `U_rel = sqrt(U_inf² + (Ω · r)²)` with
`Ω = U_inf · TSR / R_tip`. The B-frame traction target is `[q_y, q_z]` N/m on a unit strip
(flapwise, edgewise).

### Test / CI without XFOIL

Set `SectionLoadsJobConfig.use_direct_xfoil_scalars=True` or env
`ADVANCED_BLADE_SECTION_LOADS_SYNTHETIC_AERO=1` to use analytic Cp/Cf shapes instead of XFOIL (no
executable required). The chosen path is recorded in the output `LoadFieldProvenance`
(`derivation` is `"direct_xfoil_scalars"` or `"tsr_relative_wind_v1"`).

---

## Key data structures

All DTOs are frozen dataclasses under this package / shared containers.
`SectionLoadsJobConfig` is owned by orchestration (`orchestration.precompute.containers`).

### `SectionStationLoadField` (handoff output)

| Field | Type | Description |
|-------|------|-------------|
| `station_id` | str | Station identifier |
| `z_station` | float | Spanwise position [m] |
| `chord_m` / `twist_rad` | float | Section chord [m] / twist [rad] |
| `outline_yz` | `(N, 2)` float | Closed B-frame skin ring |
| `traction_field` | `TractionField` | Per-panel surface traction (from `blade_utilities`) |
| `distributed_source` | `DistributedStationLoadSpec` | The line loads this field was matched to |
| `incident` | `IncidentAeroSpec` | Resolved operating point |
| `xfoil_artifacts` | Mapping | Optional XFOIL diagnostics |
| `provenance` | `LoadFieldProvenance` | Audit trail |

See also [LOAD_APPLICATION_DOUBLE_COUNTING.md](../../docs/LOAD_APPLICATION_DOUBLE_COUNTING.md)
and [LINEAR_STATIC_MAIN_TRACK_PHASE0.md](../../docs/LINEAR_STATIC_MAIN_TRACK_PHASE0.md).
