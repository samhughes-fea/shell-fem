# `reference_thin_wall/`

FE **reference / Vlasov adapters** live here. These modules wrap the canonical
thin-wall theory for the section-shell FE worker's pre-solve stage:

- `sectorial_warping.py` — open/closed sectorial coordinates and shear-centre
  from thin-wall sectorial static moments.
- `section_vlasov.py` — E-weighted closed-cell (Batho) sectorial warping,
  warping constant `I_ω`, and warping normal-stress / shear-flow fields.

## Where the theory lives

The canonical thin-wall **theory** (ω increments, wall-graph sectorial warping,
coincidence tolerance) remains in
`geometry_utilities.section_thin_wall_model`. The adapters here delegate to it
(TW-002) and must not reimplement it.

## Migration note

These modules previously lived under
`pre_processing/loading/thin_wall/`. That location is now a **time-bounded
compatibility shim** that re-exports from here and emits a `DeprecationWarning`
(deprecation register id `compat-fe-loading-thin-wall-shim`). Import from
`reference_thin_wall` directly.
