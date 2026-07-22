# section_loads

Production owner of **per-station surface traction load fields** between `section_geometry` and
`section_shell_model`. Turns extreme distributed line loads `(q_y, q_z, m_x)` plus XFOIL-weighted
pressure/shear coefficients into a B-frame traction field on the section skin.

---

## Pipeline position

```
section_geometry  (ShellMeshInputs: closed B-frame skin ring)
        │
        ▼
section_loads   ←── this package
        │  SectionStationLoadField (TractionField + provenance)
        │  → PNGs / serialization (not production Channel B)
        ▼
section_shell_model
        │  Channel A: tip-integrated R(z) (DistributedLoadIntegrator)
        │  Channel B: off in production orchestration
        ▼
```

**Production shell policy (beam-only):** the orchestration shell stage hands off tip-integrated
cantilever resultants \(R(z)=(N,V_y,V_z,M_y,M_z,T)\) and does **not** pass
`station_load_fields` into the solve. `section_loads` still builds TractionFields for
load-field PNGs and audit artifacts.

**Standalone API** (`run_section_periodic_bc_from_load_field`): when the load field has a
usable TractionField, Channel A shears are zeroed (`Vy=Vz=0`) and traction owns strip force
(`T=m_x` remains on Channel A). Adapter helpers live in
`section_shell_model.interface.traction_load_adapter`.

> **Open:** traction overlay still uses contour clamps unlike the periodic beam field
> (Problem 2). Fatigue \(L_\mathrm{rec} + L_\mathrm{rec,tr}\) composition can still
> double-count. See
> [LOAD_APPLICATION_DOUBLE_COUNTING.md](../../docs/LOAD_APPLICATION_DOUBLE_COUNTING.md).

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
`BladeDesignProblem.load_extreme_loads_dat` in `section_optimisation.api`.

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

All DTOs are frozen dataclasses in [core/types.py](core/types.py). `SectionLoadsJobConfig` is owned
by [orchestration](../orchestration/README.md) (`orchestration.precompute.containers`).

### `SectionStationLoadField` (handoff output)
The contract from `section_loads` to `section_shell_model`.

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
| `provenance` | `LoadFieldProvenance` | Audit trail (see below) |

### `StationAeroContext`
Per-station geometry context for incident-aero derivation: `station_index`, `z_station`,
`radial_pos_m`, `chord_m`, `twist_deg`, `r_tip_m`.

### `IncidentAeroSpec`
Resolved operating point passed to XFOIL: `u_rel_m_s`, `reynolds_number`, `mach_number`,
`alpha_deg`, `omega_rad_s`, `u_tan_m_s`.

### `DistributedStationLoadSpec`
Authoritative distributed line loads at one station: `z_station`, `q_y_Npm`, `q_z_Npm`, `m_x_Nmpm`.

### `LoadFieldProvenance`
Audit trail: `derivation`, `xfoil_used`, `synthetic_aero`, `direct_scalar_override`,
`load_matcher_rank`, `notes`.

---

## Package layout

```
section_loads/
├── api/
│   └── build_station_load_field.py   # Public entry: build_station_load_field()
├── core/
│   └── types.py                      # All DTOs (SectionStationLoadField, …)
├── engine/
│   ├── incident_aero.py              # Re/Ma/α + relative-wind derivation
│   ├── outline_frames.py             # B-frame → S-frame chord transforms
│   └── station_load_builder.py       # XFOIL run + load matching
├── io/                    # Reserved for future load-field I/O helpers
├── vis.py                            # Per-station traction diagnostic PNGs
└── tests/
    ├── test_station_load_builder.py
    └── test_vis.py
```

When orchestration runs with ``persist_pngs=True`` (default), each plotted station
writes under ``section_loads/station_i{idx}_z{rz}/``:

- ``load_field.json``
- ``force_vs_chord.png`` — 2×2 running-load decomposition (S-frame + B-frame)
- ``airfoil_area_diagrams.png`` — signed pressure / shear / total traction ribbons

Set ``persist_pngs=False`` to emit JSON only (used during design optimisation loops).

---

## Quick start

```python
from blade_precompute.section_loads import build_station_load_field
from blade_precompute.section_loads.core.types import (
    DistributedStationLoadSpec, StationAeroContext, SectionLoadsJobConfig,
)

field = build_station_load_field(
    shell_inputs,                       # ShellMeshInputs from section_geometry
    distributed=DistributedStationLoadSpec(z_station=5.0, q_y_Npm=1.2e3, q_z_Npm=4.0e2, m_x_Nmpm=80.0),
    job_config=SectionLoadsJobConfig(use_direct_xfoil_scalars=True),   # synthetic aero (no XFOIL)
    station_context=StationAeroContext(
        station_index=10, z_station=5.0, radial_pos_m=12.0,
        chord_m=2.1, twist_deg=6.0, r_tip_m=60.0,
    ),
)
print(field.provenance.derivation)      # "direct_xfoil_scalars"
```

---

## References

- [section_geometry/README.md](../section_geometry/README.md) — produces the B-frame skin ring (`ShellMeshInputs`)
- [section_shell_model/README.md](../section_shell_model/README.md) — consumes the station load field
- [contract/README.md](../contract/README.md) — geometry handoff conventions (B-frame)
- [orchestration/README.md](../orchestration/README.md) — owns `SectionLoadsJobConfig` and drives the stage via `SectionLoadsStage`; builds only stations matching `section_plot_station_spec`
