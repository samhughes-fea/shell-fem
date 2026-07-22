# Periodic-BC CalculiX (ccx) S4R homogenization — design

> **Non-product / historical.** Channel-A periodic product APIs are removed (Phase 0).
> This note documents the CCX validation oracle only (`calculix_ccx/periodic.py`).
> Production FI is the linear-static cascade — see
> [LINEAR_STATIC_MAIN_TRACK_PHASE0.md](LINEAR_STATIC_MAIN_TRACK_PHASE0.md).

**Status:** *validated for axial, bending, and torsion at the section-integral level (real ccx 2.23).*
Per-wall bending ``Mx/My/Mxy`` are recovered from ``.frd`` through-thickness IP stresses when FRD
output is enabled (default in ``write_section_inp_periodic``; disable via ``emit_frd=False`` on
``run_section_periodic_bc_ccx``).

The full external-`ccx` periodic pipeline is implemented and exercised against **CalculiX 2.23**:
- deck export — `write_section_inp_periodic`
  ([`calculix_ccx/section_export.py`](../calculix_ccx/section_export.py)): unit-cell S4R deck with
  periodic `*EQUATION` cut-face coupling (deduped so each dependent DOF appears once), 6 reference
  nodes (beam unit modes), **no cut-face mesh-node clamp** (every physical node lies on a periodic
  cut face), optional ``*EL FILE`` ``S`` for ``.frd`` bending recovery (``*NODE FILE`` omitted —
  node stresses do not map to strip elements), and 6 mode `*STEP`s printing per-element **`S`** +
  reference-node `RF`;
- 3D-stress recovery — `parse_ccx_dat_stresses` / `parse_ccx_dat_periodic` /
  `map_ccx_stresses_to_panel_resultants` ([`calculix_ccx/results.py`](../calculix_ccx/results.py)):
  membrane `Nx = sxx·t`, shear flow `Nxy = (sxy·ty + sxz·tz)·t`, contour `Ny`, transverse `Qx`;
- FRD bending — `parse_ccx_frd_stress_blocks`, `superpose_frd_stress_blocks`,
  `apply_frd_bending_to_panel_resultants`: ``Mx/My/Mxy`` from **min/max IP stress spread**
  (ccx 2.23 FRD has no SNEG/SPOS labels). Returns coverage stats
  (`n_applied` / `n_missing_ip` / `n_rejected_bound`); elements without usable FRD data
  keep ``RESERVED`` provenance with an explicit skip reason rather than a silent zero.
  Packed ``-1`` rows use Fortran ``i10`` internal element id + integration-point digit
  fused with the first ``e12.5`` field; remapped to mesh ``eid`` via
  ``ccx_frd_element_base(mesh)`` (``max(node id) + 6`` ref nodes) with a secondary
  ``internal // 10`` fallback for high-numbered expanded elements;
- **Mode amplitudes** — `run_section_periodic_bc_ccx` uses the MITC4 energy homogenizer
  (`mitc4_periodic_k6` in [`validation/periodic_k6_oracle.py`](../validation/periodic_k6_oracle.py))
  for `K6 q = f_beam` because ccx reference-node reactions do not yield a reliable 6×6 `K6` for
  bending/torsion with the current `*EQUATION` MPC layout; ccx supplies **S4R unit-mode stress**
  fields superposed by the trusted amplitudes;
- `run_section_periodic_bc_ccx` ([`calculix_ccx/periodic.py`](../calculix_ccx/periodic.py)).

**Validated (real ccx 2.23):**
- Pure-N **axial equilibrium** `∮Nx ds = N` to **~0.2–5 %** (`test_periodic_ccx_axial_equilibrium`).
- Pure-`My` / `Mz` **section integrals** within **~10 %** of MITC4
  (`test_periodic_ccx_section_integrals_match_mitc4`).
- Applied **`T` drives substantial `Nxy`** (same order of magnitude as MITC4;
  `test_periodic_ccx_torsion_drives_nxy`).

**Known limitation — reaction K6 (Spike 2026-07-14, diagnose-only):**
`assemble_k6_from_steps` from reference-node `RF` matches MITC4 on the **axial** diagonal
only. Measured diagonal ratios ``K6_ccx_RF[i,i] / mitc4_periodic_k6_global[i,i]``
(NACA 2412 + spar, ``NE=4``, real ccx 2.23, ``L_x = 1``):

| mode | ε₀ | κ_y | κ_z | T | V_z | V_y |
|------|----|-----|-----|---|-----|-----|
| ratio | **0.97** | **313** | **2071** | 5.5 | 21 | 35 |

Same deck on a trivial L-strip (two isotropic panels) still over-stiffens bending/torsion
(κ_y ~16×, κ_z ~31×, T ~26×) — **not** an airfoil-mesh artefact. Production does **not**
use reaction K6 (hybrid MITC4 energy `K6` stays).

**Root-cause verdict (confirmed):** ref-node `RF` are **MPC Lagrange reactions** conjugate
to the shared DOF-1 amplitude of the multi-term `*EQUATION` layout, not beam section
resultants conjugate to `(ε₀, κ_y, κ_z, T, V_z, V_y)`. Evidence:

- Every ref-node `RF` component is pure `fx`; `fy`/`fz` are identically zero (macro
  amplitude lives on DOF 1 only — indexing is correct, not a component mix-up).
- Error is **not** a uniform scale factor (κ_y ≠ κ_z ≠ T ratios), so it is not a missing
  `L_x` or unit conversion alone (`L_x = 1` on both fixtures).
- Large spurious κ_y–κ_z reaction coupling appears on the NACA fixture while MITC4 energy
  `K6` stays nearly diagonal in that block.
- MITC4 `dof_frame="local"` vs `"global"`: ε₀/κ_y/κ_z diagonals identical; torsion/shear
  frame deltas exist but are ≪ the reaction error on κ_y/κ_z — frame choice is **not**
  the 300× bug. Hybrid production currently calls `mitc4_periodic_k6` (local default);
  the reaction lock test compares against `mitc4_periodic_k6_global` (deck-aligned).

**No-promote:** do not wire `assemble_k6_from_steps` / `ccx_periodic_k6` into production
amplitudes. A future **MPC redesign** spike (separate control DOFs with energy-/work-
conjugate extraction, or virtual-work `K6` from ccx stress fields) is required before
fully ccx-native homogenisation; until then hybrid MITC4 `K6` remains.

Repro diagnostic (not production):

```powershell
python -m blade_structure_utilities.section_shell_model.benchmarks.run_ccx_reaction_k6_diag
```

Lock: `test_periodic_ccx_k6_matches_mitc4_diagonal` asserts axial ~20% and κ_y still `> 100×`.

**Acceptance policy (2026-07-14 Spike 4A/4B):** `test_ccx_periodic_acceptance.py` tracks the
same load-aligned ~10% integral / torsion Nxy `rel=0.85` bands as the engineering suite;
the aspirational all-channel 0.5× xfail band was withdrawn (parasitic off-axis channels +
S4R redistribution — not FRD). See [DEFERRED_RESEARCH.md](DEFERRED_RESEARCH.md).

**Still deferred:** exact ccx-vs-MITC4 per-element `Nxy` match under torsion (S4R local
redistribution); reaction-based `K6` after MPC redesign (see no-promote above).

> Related fixes already landed: a **frame-aware** periodic transform
> (`compute_k6_periodic_bc(..., dof_frame="global")`, kept for reuse).
>
> The **external `ccx` solve is the only S4R path** and is the intended production route
> (this document). An in-house native pure-Python S4R *element*
> (`element_library/calculix_s4r`, `ShellTheory.CALCULIX_S4R`) previously existed as a
> no-dependency cross-check, but it and its native periodic entry have since been **removed**
> (see `docs/CALCULIX_CCX_ELEMENT_LIBRARY_RESTRUCTURE.md`): comparing its section `K6` against
> MITC4 showed the torsion / transverse-shear modes were structurally wrong (`K6[3,3]` ≈
> **3000×** too stiff, **mesh-independent** for `NE = 8..64`, so not shear locking) with ~12 %
> axial over-stiffness (hourglass), a defect in the element that would have needed a full
> reformulation. The periodic machinery / frame mapping is correct (the prescribed global
> `u_y/u_z` jump is provably equivalent to MITC4's `u_s` jump). The external `ccx` route does
> not share that element, so it is the right way to get real S4R resultants.

## Entry point

```python
from blade_structure_utilities.section_shell_model.calculix_ccx import (
    run_section_periodic_bc_ccx,
)

bundle = run_section_periodic_bc_ccx(
    pb,
    My=1.0e5,
    n_elements_per_panel=8,
    ccx_work_dir="/tmp/ccx_section",
    emit_frd=True,  # set False to skip .frd wall-bending recovery (faster)
)
```

Requires `ccx` on `PATH` and a writable `ccx_work_dir`.

> **Note (Phase 0):** the former `simulation_runner.api.periodic` re-export was
> **removed**. Import CCX helpers from `calculix_ccx` only. Production FI is the
> linear-static cascade, not Channel-A periodic.

## Tests

- [`calculix_ccx/tests/test_calculix_ccx_periodic.py`](../calculix_ccx/tests/test_calculix_ccx_periodic.py)
  — deck, parser, hybrid solve, reaction-K6 lock, ccx-gated E2E (``@pytest.mark.ccx``).
- [`calculix_ccx/tests/test_ccx_periodic_acceptance.py`](../calculix_ccx/tests/test_ccx_periodic_acceptance.py)
  — Spike 4B acceptance bands.
- Diagnostic: [`benchmarks/run_ccx_reaction_k6_diag.py`](../benchmarks/run_ccx_reaction_k6_diag.py)
  — RF dump vs MITC4 columns (research only).

## Risks / open questions

- Shell-normal / element orientation consistency between CalculiX S4R and the strip frame.
- FRD IP layout across ccx versions (parser handles spaced and packed ``-1`` rows; ccx 2.23 uses
  fused ``i10`` + ``e12.5`` fields with ``E+010`` exponent overlap).
- Fully ccx-native `K6` requires an MPC / conjugate redesign (reaction path diagnosed; no-promote).
- FRD expanded-element numbering is dual-layout (``internal - base`` and ``internal // 10``); not
  all strip elements receive multi-IP through-thickness data on every mesh.

## Out of scope / non-issue

HSDT backends were removed from `section_shell_model` (2026-07-14); this periodic ccx path
is FSDT/S4R only. Reaction-K6 redesign remains deferred (see no-promote above).
