# `element_library` restructure — CalculiX backend is the external `ccx` binary

## Summary

`element_library/` now cleanly holds exactly the three shell-element backends the benchmark
compares:

| Package | Kind | Driver |
| --- | --- | --- |
| `element_library/mitc4` | native Python FSDT (MITC4 quad) | `run_section_periodic_bc_mitc4` |
| `element_library/hsdt` | native Python HSDT (8-DOF) | `run_section_periodic_bc_hsdt` |
| `element_library/calculix_ccx` | **external CalculiX `ccx` binary** (S4R) | `run_section_periodic_bc_ccx` |

The previous `element_library/calculix_s4r` package was a **native pure-Python S4R
reimplementation** that nothing in the benchmark used, while the *real* external-binary backend
lived scattered across `engine/` and `io/`. That native kernel has been **removed**, and the
external CalculiX backend has been **re-homed** under `element_library/calculix_ccx`.

## What was done

### 1. Re-homed the external `ccx` backend (`git mv`, history preserved)

| From | To |
| --- | --- |
| `engine/calculix_ccx_solve.py` | `element_library/calculix_ccx/solve.py` |
| `engine/ccx_mesh_assembly.py` | `element_library/calculix_ccx/mesh.py` |
| `engine/periodic_ccx_homogenisation.py` | `element_library/calculix_ccx/periodic.py` |
| `io/calculix_section_export.py` | `element_library/calculix_ccx/section_export.py` |
| `io/calculix_results.py` | `element_library/calculix_ccx/results.py` |
| `io/calculix_export.py` | `element_library/calculix_ccx/strip_export.py` |
| `element_library/calculix_s4r/orientation.py` | `element_library/calculix_ccx/orientation.py` |

Moving one directory deeper turned every `..x` relative import into `...x`; cross-references
between the moved modules became same-package `.x` imports. A new
`element_library/calculix_ccx/__init__.py` re-exports the public API
(`run_section_periodic_bc_ccx`, `solve_global_coupled_ccx`, `build_ccx_section_mesh`, plus the
orientation helpers).

### 2. Deleted the native pure-Python S4R kernel

- `element_library/calculix_s4r/` (element + package).
- `engine/global_s4r_assembly.py` (`solve_global_coupled_s4r`).
- `S4rStripKernel` / `S4R_KERNEL` and the `_KERNELS[CALCULIX_S4R]` registration in
  `element_library/kernel.py` (+ `element_library/__init__.py` re-exports).
- `_solve_native_s4r` backend in `engine/shell_solve_registry.py`.
- `_s4r_global_coupled_solve_from_thin_wall` in `engine/mitc4_solve_orchestrator.py` and the
  `ShellTheory.CALCULIX_S4R` branch in `engine/shell_solve_registry.py` (formerly `api/thin_wall_bridge.py`).
- `CalculixS4RConfig` / `CALCULIX_S4R_DEFAULTS` in `core/fem_config.py`.

### 3. Collapsed the enum / provenance to a single external CalculiX

`core/types.py` lost `ShellTheory.CALCULIX_S4R` and `ProvenanceKind.CALCULIX_S4R`;
`CALCULIX_CCX` is the single external-CalculiX identity. `EXPERIMENTAL_SHELL_THEORIES_BY_REGIME`
`LINEAR` is now `(HSDT,)` (CCX already sits in the *supported* LINEAR tuple).

### 4. Repointed importers + tests

Registry, thin-wall bridge, orchestrator, validation oracle, the benchmark, and the
external-ccx tests now import from `element_library.calculix_ccx.*`. The three native-S4R test
files were deleted; their strip-export / `quad_coords_from_strip` coverage was preserved in the
new `tests/test_calculix_ccx_strip.py`.

## Findings

- **The benchmark already ran the three-way comparison.** `ccx` (CalculiX 2.23) is on this
  machine's PATH, and `benchmark_summary.json` already carried `"ccx_available": true`. The
  request was therefore structural, not a missing feature — the `calculix_s4r` native kernel was
  a confusing, unused duplicate of the external backend.
- **The native `S4rStripKernel` was latently broken** (now removed): its `.stiffness` called the
  *global-coords* `s4r_stiffness(coords, abd, thickness)` with `L_s, L_x` in the coordinate
  slots, and `.recover_resultants` raised `NotImplementedError`. Nothing exercised it via the
  periodic path, so the breakage was dormant.
- **Two generic mesh helpers had to be rescued.** `_assign_shared_nodes` / `_assign_unique_nodes`
  lived in the deleted `global_s4r_assembly.py` but were imported by the ccx mesh builder; they
  are not S4R-specific and were relocated into `element_library/calculix_ccx/mesh.py`.
- **`orientation.py` is shared geometry.** Its `quad_coords_from_strip` etc. are used by the ccx
  strip export, so it survived the native-kernel deletion (moved into `calculix_ccx`).

## Verification

- Static: repo-wide grep for `CALCULIX_S4R`, `calculix_s4r`, `global_s4r_assembly`,
  `S4R_KERNEL`, `S4rStripKernel`, and the old `engine.*`/`io.calculix_*` module paths → **no
  matches**.
- Import smoke: `element_library.calculix_ccx`, `shell_solve_registry`
  (`BACKENDS = {FSDT, HSDT, CALCULIX_CCX}`), `api/periodic`, `api/shell_panel_topology`, `periodic_k6_oracle`, and the
  orchestrator all import; `ShellTheory`/`ProvenanceKind` = `{fsdt, hsdt, calculix_ccx}`.
  (`api/legacy_clamped` was deleted 2026-07-14 — do not reintroduce.)
- Unit tests: `pytest blade_precompute/section_shell_fea/tests` → **223 passed, 1 skipped,
  6 xfailed, 1 xpassed**.
- End-to-end benchmark (`benchmark_shell_elements.py naca2412_twinweb_carbon`, `ccx` on PATH):
  the external-ccx column populates. Regenerated `benchmark_summary.json` has
  `"ccx_available": true` with `rel_error_fsdt_vs_ccx`, `section_integrals.ccx`, and
  `max_fi.ccx`. Sample (load `M22`): section-integral `|FSDT-ccx|/scale ≈ 1.9e-2`,
  max Hashin FI FSDT `3.33e-2` vs ccx `6.26e-2`. Per-element rel errors are large by design
  (different element formulation; self-equilibrated redistribution).

## Addendum (2026-07) — lift `calculix_ccx` out of `element_library`

**Status: complete.** The external CalculiX backend is a full end-to-end process (mesh
export → `ccx` subprocess → results parsing → periodic homogenisation), not element tensor
assemblies plus co-located utilities. Following the 1d-fem
`pre_processing/element_library` convention, `element_library/` now holds only
native Python element kernels (`mitc4`, `hsdt`) and their co-located
`utilities/` helpers (solve orchestrators, mesh builders, recovery).

| From | To |
| --- | --- |
| `element_library/calculix_ccx/` | `section_shell_fea/calculix_ccx/` |
| `engine/mitc4_solve_orchestrator.py` | `element_library/mitc4/utilities/mitc4_solve_orchestrator.py` |
| `engine/panel_mitc4_model.py` | `element_library/mitc4/utilities/panel_mitc4_model.py` |
| `core/mitc4_mesh.py` | `element_library/mitc4/utilities/mitc4_mesh.py` |
| `engine/hsdt_solve_orchestrator.py` | `element_library/hsdt/utilities/hsdt_solve_orchestrator.py` |
| `engine/hsdt_recovery.py` | `element_library/hsdt/utilities/hsdt_recovery.py` |
| `engine/hsdt_homogenisation.py` | `element_library/hsdt/utilities/hsdt_homogenisation.py` |

The moves above are applied on disk: `section_shell_fea/calculix_ccx/` is the canonical
home (with `element_library/calculix_ccx/__init__.py` a re-export shim), and the MITC4/HSDT
utilities live under `element_library/*/utilities/`. The old `engine/` and `core/` re-export
shims for the moved MITC4/HSDT utilities (`engine/mitc4_solve_orchestrator.py`,
`engine/panel_mitc4_model.py`, `engine/hsdt_solve_orchestrator.py`, `engine/hsdt_recovery.py`,
`engine/hsdt_homogenisation.py`, `core/mitc4_mesh.py`) have since been **removed**; callers now
import directly from `element_library/*/utilities/` (or, for mesh types, `core/section_strip_mesh.py`).
`solve_global_coupled_shell` is an alias of `solve_global_coupled_shell` on
`engine/global_shell_assembly.py`; the former `engine/global_mitc4_assembly.py` alias shim has been removed.
`ShellTheory.CALCULIX_CCX`, provenance identities, and `resolve_kernel` dispatch are unchanged;
importers now use `section_shell_fea.calculix_ccx`.
