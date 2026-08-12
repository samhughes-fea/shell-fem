# CalculiX ccx backend — quarantined, external validation-only backend

↑ Parent: [`../README.md`](../README.md)

This package drives the external `ccx` (CalculiX) solver as an S4R shell
element oracle for cross-validating the native FSDT MITC4 element. It is
**not** a production analysis theory: `ShellTheory.CALCULIX_CCX` is removed
from `SUPPORTED_SHELL_THEORY_BY_REGIME` (`pre_processing/containers/types.py`
→ `shell_theory_capabilities.py`) — a caller cannot select it through the
general regime/theory config validator (`validate_shell_configuration`). The
enum value, `ProvenanceKind`, and its `SHELL_THEORY_CAPABILITIES` row
(dispatch-feature flags consulted directly by the three dispatch seams) are
unchanged; only production-dispatch *registration* and general-config
selectability are quarantined.

## Opt-in registration

Per the CalculiX quarantine, production dispatch
(`pre_processing/dispatch/station_theory_dispatch.py`,
`pre_processing/dispatch/shell_solve_registry.py`) does **not** import or register
this package at module load time. Callers who need CCX through the theory-dispatched
router must explicitly call:

```python
from blade_structure_utilities.fem_utilities.section_shell_fea.calculix_ccx.register import (
    register_calculix_ccx_backend,
)

register_calculix_ccx_backend()
```

before invoking any theory-dispatched CCX path (`dispatch_station_theory`,
`resolve_backend(ShellTheory.CALCULIX_CCX)` / `ShellTheory.CALCULIX_CCX in BACKENDS`).
The hook is idempotent — safe to call multiple times (e.g. once per worker
process or test module). Orchestration
(`orchestration/precompute/shell_runtime.py`) calls it automatically when
`shell_theory="calculix_ccx"` is selected.

**Channel-A periodic product APIs are removed** (Phase 0). CCX may still keep an
internal `periodic.py` helper for validation oracles; that is **not** a production
FI entry. Station dispatch via `station_dispatch_adapter` raises
`NotImplementedError` for the former thin-wall periodic CCX product path.

CCX has no local element-factory entry: it is an external subprocess, not a
native element formulation, so there is nothing to register with the
`formulation_factory` registry. The former HSDT package (and its element-factory
registration) was deleted 2026-07-14.

## Layout

- `mesh.py` — `build_ccx_section_mesh` (panel-strip → ccx shell mesh; optional
  `cap_tie_mode` builds cap–skin tie surfaces).
- `cap_skin_ties.py` — project cap nodes onto skin midlines; emit surface sets +
  measured `POSITION TOLERANCE` for bonded joints.
- `section_export.py` — `.inp` deck writers (clamped and periodic BC); clamped
  decks can emit `*TIE` (default) or closest-node `*EQUATION` fallback for
  `cap_tie_mode="skin_offset"`.
- `joint_validation.py` — MITC4↔CCX joint oracle (offline deck checks + optional
  live `ccx`); CLI via `benchmarks/run_ccx_joint_validation.py`.
- `strip_export.py`, `orientation.py` — element/orientation card helpers.
- `solve.py` — `solve_global_coupled_ccx` (subprocess invocation + `.dat`/`.frd`
  parsing orchestration for diagnostic/global-coupled decks).
- `results.py` — `.dat`/`.frd` result parsers (FRD bending returns coverage stats);
  also root-segment multi-row mappers (`map_root_segment_stresses_to_resultants`,
  `parse_ccx_dat_node_reactions`).
- `periodic.py` — `run_section_periodic_bc_ccx`, `assemble_k6_from_steps`
  (periodic unit-cell BC path; also used directly by
  `validation/periodic_k6_oracle.py`). **Hybrid:** amplitudes from MITC4 energy
  `K6`; CCX supplies S4R unit-mode stresses. Reaction-based `K6` is **no-promote**
  — see `docs/PERIODIC_CCX_HOMOGENIZATION.md`.
- `root_segment_mesh.py` / `root_segment_export.py` — **diagnostic-only** clamped
  multi-row lofted root segment (statically equivalent outboard `*CLOAD` + root
  clamp; ccx forbids rigid/distributing coupling on shell nodes). See
  `docs/ROOT_SEGMENT_CCX_DIAGNOSTIC.md`. Not in production dispatch. The
  **production root path** is the contour-supported linear-static cascade
  (station 0 uses the same `run_section_linear_static_from_resultants` path as
  all stations). Channel-A periodic RVEs are removed (Phase 0).
- `thin_wall_adapter.py` — `ccx_global_coupled_solve_from_thin_wall` helper retained
  for diagnostic/CCX-only callers; the former `api/legacy_clamped.py` production
  route that used it has been **deleted**.
- `station_dispatch_adapter.py` — per-station dispatch adapter
  (`ccx_station_solve_from_thin_wall`, `dispatch_ccx_station_solve`); raises
  `NotImplementedError` for the former Channel-A thin-wall periodic CCX product
  path. Shell panel topology geometry remains available for validation helpers.
- `register.py` — `register_calculix_ccx_backend()`, the sole opt-in entry
  point for the three production dispatch maps.
- `tests/` — CCX-marked suite (`test_calculix_ccx_*.py` / `test_ccx_*.py` /
  `test_frd_bending_apply.py` / `test_root_segment_ccx.py` /
  `test_joint_validation.py`), collected via
  `pyproject.toml` `testpaths` and excluded from the default gate by
  `@pytest.mark.ccx` where a live `ccx` binary is required (offline deck/mesh
  tests in `test_root_segment_ccx.py` / joint deck tests run without the marker).

## Joint modeling (validation oracle)

| MITC4 production | CalculiX export |
|------------------|-----------------|
| `interface_constraint_mode="shared"` (skin–web / cap–web) | Shared junction nodes in `build_ccx_section_mesh` |
| `cap_tie_mode="skin_offset"` (skin–cap) | `*TIE` with `POSITION TOLERANCE` ≈ measured wall offset (`cap_skin_ties.py`) |
| `cap_tie_mode="none"` | Caps left disconnected (BL-003 control) |
| `transformed_basis` | Not exported (diagnostic-only MITC4 path) |

**Caveat:** the CalculiX manual restricts `*TIE` to continuum 3-D elements; shells
are expanded to bricks internally and edge/face ties often work in practice.
If a CCX build rejects shell `*TIE`, pass `cap_tie_export="equation"` for
closest-node DOF locks (approximate offset kinematics).

Compare **section integrals, interior peaks, and cap N12 load share** — not
elemental Hashin maxima on tie-slave bands or Defect-K corners. Defect K under
shared nodes is a **shared-node modeling** issue (MITC4 and CCX), not MITC4-only.

```powershell
python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.run_ccx_joint_validation --no-mitc4
python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.run_ccx_joint_validation --live-ccx --layouts 3D-F
```

## Membrane GCI study (40 keys)

Live CCX recovers membrane resultants only (`N11`, `N12`). The CCX grid-independence
campaign therefore gates **skin/web `peak_abs_n12`** drift + ASME GCI (Hashin /
mid_sig / `governing_fi` are NaN and excluded from gates).

| Piece | Path |
|-------|------|
| SFV → nx/nxy | `sfv_panel_loads.py` |
| Cascade rung | `ccx_cascade_mesh_convergence.run_ccx_cascade_at_frac` |
| Study orchestration | `grid_independence_study` with `backend="ccx"` / `gates="ccx"` |
| CLI | `benchmarks/run_ccx_grid_independence_study.py` |

```powershell
python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.run_ccx_grid_independence_study --keys fast
python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.run_ccx_grid_independence_study --keys all --gates ccx
```

Outputs land in `visualisation/output/gci_study_ccx/` (`gci_ladder.csv`,
`gci_summary.csv`, `plots/`, `ccx_work/`). Cap `peak_abs_n12` remains
informational under `skin_offset` (same policy as MITC4).

## Consumers

Direct-import consumers bypass the dispatch registry entirely and are
**unaffected** by this quarantine (no registration call needed):

- `examples/benchmark_shell_elements.py` — imports `run_section_periodic_bc_ccx`
  directly from `calculix_ccx.periodic`; the benchmark's whole purpose is
  FSDT (MITC4) vs CCX S4R comparison.
- `validation/periodic_k6_oracle.py` — imports `calculix_ccx.periodic` /
  `.results` / `.section_export` / `.mesh` directly for its `ccx_periodic_k6`
  oracle.
- `benchmarks/run_ccx_reaction_k6_diag.py` — research-only RF vs MITC4 K6 dump
  (reaction K6 is not production; see `docs/PERIODIC_CCX_HOMOGENIZATION.md`).
- `benchmarks/run_ccx_root_segment_diag.py` — research-only clamped root-segment
  vs native linear-static baseline at station 0 (see
  `docs/ROOT_SEGMENT_CCX_DIAGNOSTIC.md`).

Registry-dispatched consumers (need `register_calculix_ccx_backend()` first):

- `orchestration/precompute/shell_runtime.py`'s `shell_theory="calculix_ccx"`
  station option (calls the registration hook automatically).
- The dedicated `calculix_ccx/tests/` suite; tests that exercise the
  theory-dispatch maps register explicitly, while pure direct-import tests
  do not need to.

## Deferred scope

Both deferred items from the original Phase 6 pass are now done:

- ~~`ccx_extra` / `ccx_work_dir` / `ccx_exe` / `ccx_work_subdir` genericisation~~ —
  **done** (Bucket 1): production DTOs and public routers now use
  `external_backend_options` (`work_dir` / `work_subdir` / `exe`) and
  `StationTheorySolveResult.backend_extra`; package-local CCX entry points still
  accept `ccx_work_dir` / `ccx_exe`.
- ~~Physical relocation of `test_calculix_ccx_*.py` / `test_ccx_*.py`~~ —
  **done**: tests live under `calculix_ccx/tests/` with a matching
  `pyproject.toml` `testpaths` entry; still gated by `@pytest.mark.ccx`.

Still deferred (research only, not production FI):

- Reaction-based native `K6` (MPC redesign required — no-promote).
- Raising FRD remap coverage toward 100% only if promoting CCX Hashin trust.
