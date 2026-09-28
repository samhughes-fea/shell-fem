# reference_thin_wall — FE Vlasov adaptation over BSU thin-wall theory

Internal FE home for adapting BSU thin-wall / sectorial theory into FE
pre-solve reference products and Vlasov result packaging.

↑ Parent: [`../README.md`](../README.md) · Package: [`../../README.md`](../../README.md)

## Ownership

| Package | Owns |
|---------|------|
| [`geometry_utilities.section_thin_wall_model`](../../../../geometry_utilities/section_thin_wall_model/README.md) | Sectorial / thin-wall theory kernel (`omega_increment`, `sectorial_warping`, Batho, `I_ω`); **no runner lifecycle** — call pure functions directly |
| [`geometry_utilities.section_properties`](../../../../geometry_utilities/section_properties/README.md) reference-axes | Shear-centre kernels from midlines / loops |
| **`pre_processing.reference_thin_wall` (this home)** | FE pre-solve adaptation and `SectionVlasovResult` packaging |
| Public `section_shell_fea` barrel | Canonical worker export surface |

Do **not** introduce a `SectionThinWallModelRunner`. FE adapters call BSU
deterministic functions. This package must not import FE `processing` or solve
modules.

`pre_processing/loading/thin_wall/` remains a warning-alias shim until
**2026-11-30** (`compat-fe-loading-thin-wall-shim`).

## Exact I/O

| Path | Role |
|------|------|
| [`section_vlasov.py`](section_vlasov.py) | `compute_section_vlasov` → `SectionVlasovResult` (midlines path → BSU `sectorial_warping`) |
| [`sectorial_warping.py`](sectorial_warping.py) | Open-polyline FE helpers over BSU `omega_increment` (outer-skin SC estimate) |

## See also

- [`../../../../geometry_utilities/section_thin_wall_model/README.md`](../../../../geometry_utilities/section_thin_wall_model/README.md)
- [`../../docs/DEPRECATED_ALIASES.md`](../../docs/DEPRECATED_ALIASES.md)
