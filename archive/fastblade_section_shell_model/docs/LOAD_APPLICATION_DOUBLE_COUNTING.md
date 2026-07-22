# Section shell load application — double-counting and BC split

**Status:** Problem 1 (load provenance / double-counting) **implemented** for orchestration +
adapter (2026-07-14). Fatigue \(L_\mathrm{rec,tr}\) residual partition **implemented** (2026-07-14).
Problem 2 (single periodic free set for face traction) **implemented** (2026-07-14).  
**Scope:** Native MITC4 periodic path and orchestration / fatigue handoffs.  
**Architecture:** generalised section BVP — one free set
([`PeriodicReducedBVP`](../engine/periodic_reduced_bvp.py)) with named load channels
(beam resultants / face traction). Production is **linear static**.  
**Out of scope:** CalculiX reaction `K6` / MPC redesign — ccx is a **benchmark backend** only;
native energy `K6` remains authoritative for homogenisation. Geometric Newton is
**parked permanently** (not scheduled).

Related: [SHELL_MODEL.md](SHELL_MODEL.md) (multi-scale BCs),  
[pre_processing/loading/README.md](../pre_processing/loading/README.md) (`section_loads`),  
[`interface/traction_load_adapter.py`](../interface/traction_load_adapter.py),  
[`engine/periodic_traction_superposition.py`](../engine/periodic_traction_superposition.py),  
[`engine/periodic_reduced_bvp.py`](../engine/periodic_reduced_bvp.py),  
[`engine/homogenisation.py`](../engine/homogenisation.py),  
[`api/periodic.py`](../api/periodic.py).

---

## Intent of the two-channel model

The section shell package solves an **RVE** at a blade station. Loads arrive at two scales:

| Channel | Physical meaning | Native mechanism |
|---------|------------------|------------------|
| **A — beam resultants** | Cantilever station forces/moments \(R=(N,V_y,V_z,M_y,M_z,T)\) | Periodic Saint-Venant homogenisation: energy `K6`, \(K6\,q = f_\mathrm{beam}\), \(d_\mathrm{beam}=U q\) |
| **B — surface tractions** | Skin pressure/shear on the section mesh (hydro / aero patch) | Face loads → nodal \(f_\mathrm{tr}\) reduced with the same \(T_\mathrm{pbc}\) as K6 → \(d_\mathrm{tr}\), then \(d_\mathrm{total}=d_\mathrm{beam}+d_\mathrm{tr}\) on one free set |

Conceptually this is correct:

- Channel A carries the **global** beam constitutive response (and is what optimiser FI uses).
- Channel B adds **local** face loading that a six-component resultant vector cannot resolve by itself.

Linear elasticity allows superposition **only if** the two channels do not both claim the same
physical load content, and **only if** they are solutions of compatible boundary-value problems
(or one consistent reduced system).

There is **no** `include_traction` flag. Channel B activates when **both** `traction_field` and
`traction_points_yz` are non-`None` (see
[`superpose_traction_on_beam_displacement`](../engine/periodic_traction_superposition.py)).
That same condition also switches recovery from
[`recover_resultants_from_modes`](../engine/periodic_secondary_recovery.py) (provenance
`"periodic BC"`) to
[`recover_resultants_from_displacement`](../engine/periodic_secondary_recovery.py)
(provenance `"periodic BC + face traction (single BVP)"`).

---

## Current FE wiring

```text
                    distributed hydro / aero q_y, q_z, m_x
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   DistributedLoadIntegrator   section_loads         (fatigue may keep raw q)
   → ExtremeLoads / R(z)       TractionField
          │                   (PNGs / serialization)
          │
   OPTIMISER FI  +  ORCHESTRATION shell
   Channel A only (tip-integrated R(z))
   Channel B off
```

### Orchestration call chain

Production precompute integrates distributed loads via
[`tip_integrated_station_resultants`](../../orchestration/precompute/shell_station_resultants.py)
and hands the shell stage:

- `station_resultants=<tip-integrated R(z) dict>`
- `station_load_fields=None` (Channel B off; `section_loads` fields stay for PNGs only)

Per station, [`shell_runtime`](../../orchestration/precompute/shell_runtime.py) then:

1. Calls [`resolve_shell_station_resultants`](../interface/traction_load_adapter.py) —
   **`station_resultants` take precedence** over `station_load_fields` (warns if both present).
2. Builds a `StationTheorySolveRequest` with the resolved scalars
   `(N, Vy, Vz, My, Mz, T)` and `load_field=None` in production.
3. [`fsdt_periodic_station_hashin`](../engine/station_theory_dispatch.py) only activates
   traction when a `load_field` with aligned panel centres is supplied.

Standalone entry:
[`run_section_periodic_bc_from_load_field`](../interface/traction_load_adapter.py)
zeros Channel A shears when traction is usable (local-hydro-primary policy).

### Channel A scalars from a load field

[`section_resultants_from_load_field`](../interface/traction_load_adapter.py)
(and the alias `beam_resultants_from_load_field`):

| Component | Usable TractionField | No usable traction |
|-----------|----------------------|--------------------|
| \(V_y\), \(V_z\) | **0** (traction owns strip force) | `q_y`, `q_z` from `distributed_source` |
| \(T\) | `distributed_source.m_x_Nmpm` | same |
| \(N\), \(M_y\), \(M_z\) | **0** | **0** |

When there is no usable traction and `prefer_traction_integration=True`, only \(V_y\) and
\(V_z\) are replaced by `traction_field.get_total_force()`; \(T\) still comes from `m_x_Nmpm`.

### Implementation pointers

| Piece | Location |
|-------|----------|
| Beam modes + optional overlay | [`api/periodic.py`](../api/periodic.py) `run_section_periodic_bc_mitc4` |
| Load-field → `(N,…,T)` + traction | [`interface/traction_load_adapter.py`](../interface/traction_load_adapter.py) |
| `d_total = d_beam + d_tr` | [`engine/periodic_traction_superposition.py`](../engine/periodic_traction_superposition.py) |
| Theory dispatch + `used_traction` | [`engine/station_theory_dispatch.py`](../engine/station_theory_dispatch.py) |
| Optimiser (beam-only) | [`section_optimisation/engine/shell_fi_eval.py`](../../section_optimisation/engine/shell_fi_eval.py) `_run_section_shell_solve` |

---

## Channel A mechanics — periodic energy `K6`

### Assembly and cache

[`_cached_periodic_k6`](../engine/periodic_cache.py) assembles load-independent
`K_global` (zero membrane loads) and extracts the six macro modes. On this path
`bc_mode` is a **cache key only** — contour clamps are **not** applied to the
periodic homogeniser (`bc_mode` is unused for K/f during assembly).

Unit cell: \(L_x = 1\,\mathrm{m}\). **Neither** spanwise cut face is clamped.
Periodic pairing + six macro-mode jumps define the Saint-Venant RVE
([SHELL_MODEL.md](SHELL_MODEL.md) § Multi-scale boundary conditions).

### Macro-mode jumps

[`_macro_jump`](../engine/homogenisation.py) prescribes the inboard→outboard
displacement jump for each mode \(j \in \{0..5\}\):

| Mode \(j\) | Beam sense | \(\Delta u_x\) | \(\Delta u_s\) (contour tangential) |
|------------|------------|----------------|--------------------------------------|
| 0 | \(\varepsilon_0\) (extension) | \(L_x\) | 0 |
| 1 | \(\kappa_y\) | \(-(z-z_e)L_x\) | 0 |
| 2 | \(\kappa_z\) | \((y-y_e)L_x\) | 0 |
| 3 | \(\gamma_t\) (torsion) | 0 | \(L_x\) |
| 4 | \(\gamma_z\) (Vz shear) | 0 | \(t_z\,L_x\) |
| 5 | \(\gamma_y\) (Vy shear) | 0 | \(-t_y\,L_x\) |

Local MITC4 frame maps these onto DOFs \(\{u_x, u_s\}\) via `_macro_jump_dofs`.

### Energy bilinear form and load vector

[`compute_k6_periodic_bc`](../engine/homogenisation.py):

1. Build periodic transform \(T_\mathrm{pbc}\) and particular jump field \(d_0\).
2. For each mode \(j\): solve the reduced system;
   \(u_{\mathrm{modes}}[:,j] = T_\mathrm{pbc}\,q_j + d_0[:,j]\).
3. Form
   \[
   K6_{ij} = u_i^\top K_\mathrm{global}\, u_j.
   \]

In [`run_section_periodic_bc_mitc4`](../api/periodic.py), beam loads must match the
**mode-conjugate order** (not the API keyword order `(N, Vy, Vz, My, Mz, T)`):

\[
T_\mathrm{ec} = T + V_z(y_L - y_e) - V_y(z_L - z_e),
\]

\[
f_\mathrm{beam} = [N,\; M_y,\; M_z,\; T_\mathrm{ec},\; V_z,\; V_y],
\]

then \(K6\,q = f_\mathrm{beam}\) and \(d_\mathrm{beam} = U\,q\).
\((y_L, z_L)\) come from `loading_axis_yz` when supplied, else the elastic centroid
\((y_e, z_e)\).

---

## Channel B mechanics — face traction on the periodic free set

### Face-load assembly

[`traction_face_loads`](../interface/traction_face_load.py) maps each panel force
\((F_y, F_z)\) (from `TractionField.total_forces`, units N/m on the unit strip) to a
nodal RHS:

1. Find the nearest skin node-pair (bottom + top through-thickness) to the panel centre.
2. Resolve into local tangential \(u_s\) and outward-normal \(w\); scale by \(L_x\)
   (default `1.0` so N/m → N on the unit cell).
3. Split 50/50 onto the bottom and top nodes of that pair.
4. **`u_x` receives nothing** — face pressure/shear lives in the section plane.

Net applied force is conserved: \(\sum F_\mathrm{nodal}(y,z) = \sum\) `total_forces`.

### Shared periodic solve

[`PeriodicReducedBVP`](../engine/periodic_reduced_bvp.py) owns the same
\(T_\mathrm{pbc}\) / \(K_{rr}\) factor used by K6 homogenisation (memoised beside
`_cached_periodic_k6`). Face traction:

- RHS = \(f_\mathrm{tr}\) from the face-load map above.
- Solve \(K_{rr}\,u = T_\mathrm{pbc}^\top f_\mathrm{tr}\) (zero particular macro jump).
- Reconstruct \(d_\mathrm{tr} = T_\mathrm{pbc}\,u\).
- **No** `build_clamp_supports` / contour LE–TE clamps / RBM pins.

[`superpose_traction_on_beam_displacement`](../engine/periodic_traction_superposition.py)
returns \(d_\mathrm{beam} + d_\mathrm{tr}\) on that shared free set when both traction
args are present, else \(d_\mathrm{beam}\). Linear superposition is valid because both
legs satisfy the same periodic constraints.

### LoadMatcher — why the double-count is exact

Production [`station_load_builder`](../pre_processing/loading/) builds a
`TractionField` via `LoadMatcher.match` in
[`blade_utilities/point_to_surftraction/matching.py`](../../../blade_utilities/point_to_surftraction/matching.py):

\[
[F_p \mid F_s]\,\begin{bmatrix}\lambda_p\\ \lambda_s\end{bmatrix}
= \begin{bmatrix}q_y\\ q_z\end{bmatrix}
\qquad\text{(2×2 least squares)},
\]

then `total_forces = λ_p · pressure_forces + λ_s · shear_forces`.
By construction \(\sum\) `total_forces` \(\approx (q_y, q_z)\).

`match_with_moment` (3×2 system that also targets torque) exists but is **not** used in
production `station_load_builder`. Distributed \(m_x\) is stored only in
`distributed_source` and handed to Channel A as \(T\); it is **not** reconstructed
aerodynamically into the traction field
([loading/README.md](../pre_processing/loading/README.md)).

---

## Problem 1 — Double-counting

### Definition

Double-counting occurs when the **same hydro/aero intensity** \(q=(q_y,q_z)\) (or an
integrated resultant built only from that \(q\)) drives **both**:

1. Channel A via \(f_\mathrm{beam}\) (or fatigue \(L_\mathrm{rec}@R(q)\)), and  
2. Channel B via a `TractionField` matched so that \(\sum F \approx (q_y,q_z)\).

Then the section FE feels that load content twice (under two different kinematic maps).
Because LoadMatcher enforces \(\sum F \approx q\) by construction, the force double-count
on the load-field path is **exact** (same net \((q_y,q_z)\)), not an accidental near-overlap.

### What each path does today

| Caller | Channel A | Channel B | Double-count? |
|--------|-----------|-----------|---------------|
| Optimiser `shell_fi_batch` | Tip-integrated `ExtremeLoads` \(R(z)\) from `DistributedLoadIntegrator` | **Off** (`_run_section_shell_solve` never passes traction args) | **No** |
| Orchestration shell stage | Tip-integrated \(R(z)\) via `tip_integrated_station_resultants` | **Off** (`station_load_fields=None`) | **No** |
| `run_section_periodic_bc_from_load_field` | \(V_y=V_z=0\), \(T=m_x\) when traction usable | Matched `TractionField` | **No** |
| Fatigue field recovery with traction ops | \(L_\mathrm{rec}@R\) (tip-integrated) | Residual \(L_\mathrm{rec,tr}^\mathrm{res}@q\) (net force removed) | **No** (residual partition) |
| Spanwise recovery cache (`main_analysis`) | Unit beam modes only | No `L_rec_tr` | **No** |

### Secondary load-provenance bug on the load-field handoff (fixed for orchestration)

**Historical bug:** when `station_load_fields` took precedence over `station_resultants`
([`resolve_shell_station_resultants`](../interface/traction_load_adapter.py)):

- Channel A received **distributed intensities** \(q_y, q_z\) [N/m] as if they were
  section shears \(V_y, V_z\) on a **1 m** unit cell — i.e. the identity
  \(V = q \cdot L_x\) with \(L_x=1\), **not** tip-integrated cantilever shears
  from root-to-tip integration of \(q(z)\).
- \(M_y\) and \(M_z\) were hard-zeroed, so **global bending moments** from the beam
  integration were missing entirely.
- \(N\) was also zeroed.

**Current policy:**

- Production orchestration passes tip-integrated `station_resultants` and
  `station_load_fields=None` (Channel B off). `section_loads` still builds TractionFields
  for PNGs / serialization only.
- Explicit `station_resultants` take precedence over `station_load_fields` (warns if both).
- When a load field has a usable `TractionField`, Channel A shears are zeroed so the same
  \(q\) cannot enter both channels (`section_resultants_from_load_field`).

### Fatigue operator composition

[`element_recovery.py`](../../../blade_analysis/fatigue_damage/section_field/engine/element_recovery.py)
builds:

- \(L_\mathrm{rec}\) — six unit beam solves
  `{N, My, Mz, T, Vy, Vz}` via `run_section_periodic_bc` (**beam-only**, no traction).
  Shape `(n_elem, n_ply_max, 3, 6)`.
- \(L_\mathrm{rec,tr}\) (when `surface_traction=True`, the default) — two unit traction
  solves for unit \(q_y\) and \(q_z\) with **zero beam loads**, via
  `build_unit_traction_field`, then **residual-partitioned** against the matching
  beam shear columns. Shape `(n_elem, n_ply_max, 3, 2)`.
  Distributed \(m_x\) is **not** a traction mode.

**Residual partition** (`apply_residual_vy_vz_partition`, policy **(a)** residual local
content): a unit \(q_y\) (resp. \(q_z\)) traction field has net force \(1\,\mathrm{N}\) on
the unit cell — the same resultant content as unit \(V_y\) (resp. \(V_z\)). After each
successful traction solve:

\[
L_\mathrm{rec,tr}^\mathrm{res}[\ldots,0] = L_\mathrm{rec,tr}[\ldots,0] - L_\mathrm{rec}[\ldots,4]\quad (V_y),
\]
\[
L_\mathrm{rec,tr}^\mathrm{res}[\ldots,1] = L_\mathrm{rec,tr}[\ldots,1] - L_\mathrm{rec}[\ldots,5]\quad (V_z).
\]

Failed traction modes stay all-zero (beam-only fallback) and are **not** replaced by
\(-L_\mathrm{rec}[V]\). `ElementRecoveryField.traction_partition` records
`"residual_vy_vz"` when traction is on, else `"none"`.

Runtime composition in
[`element_fatigue.py`](../../../blade_analysis/fatigue_damage/section_field/engine/element_fatigue.py)
is unchanged:

\[
\sigma = L_\mathrm{rec}\,r_6 + L_\mathrm{rec,tr}^\mathrm{res}\,q_{yz},
\]

where \(r_6\) is interpolated tip-integrated `ResultantHistory` → `[N, My, Mz, T, Vy, Vz]`
and \(q_{yz}\) is interpolated `DistributedLoadHistory` → `[q_y, q_z]` (raw DAT
intensities). Channel A carries the full cantilever envelope; Channel B adds only
zero-net local redistribution. Both operators are now built on the **same** periodic
free set (Problem 2 closed); residual partition remains a linear-regime convenience.

### Clean policies (mutually exclusive)

**(a) Residual local content**  
- A = full cantilever \(R(z)\) (hydro already folded into \(V,M,T\)).  
- B = only content **not** in \(R\): zero-net redistribution, higher multipoles, or a true
  residual after projecting face loads out of the resultant subspace.  
- Rare with today’s `LoadMatcher` (which forces \(\sum F = q\)).

**(b) Channels mutually exclusive**  
- **FI / sizing:** A = \(R(z)\), B = off (current optimiser — keep).  
- **Local hydro primary:** A carries only resultants that traction does *not* represent
  (e.g. \(N,M_y,M_z,T\) from the beam; **\(V_y=V_z=0\)** if traction owns strip force),  
  B = `TractionField` matched to \(q\). Never also put \(q\) into \(V_y,V_z\).

Current load-field wiring follows **(b)** (shears zeroed when traction usable).
Fatigue follows **(a)** via residual \(L_\mathrm{rec,tr}^\mathrm{res}\).

---

## Problem 2 — BC inconsistency (**resolved**)

**Historical bug:** face traction used `build_clamp_supports` (contour \(w,\beta_s,\beta_x\)
+ RBM pins) while beam modes used periodic \(T_\mathrm{pbc}\). Their displacement sum
mixed two free sets.

**Current policy:** one [`PeriodicReducedBVP`](../engine/periodic_reduced_bvp.py)

| | Beam (Channel A) | Face traction (Channel B) |
|--|------------------|---------------------------|
| Stiffness | `K_global` | same `K_global` |
| Constraints | Periodic \(T_\mathrm{pbc}\) (+ cluster pins if needed) | **same** \(T_\mathrm{pbc}\) |
| Contour LE/TE clamps | Not applied | **Not** applied |
| RHS | \(K6\,q = f_\mathrm{beam}\), \(d_\mathrm{beam}=U q\) | \(K_{rr}\,u = T^\top f_\mathrm{tr}\) |
| Reconstruction | mode superposition | \(d_\mathrm{tr}=T\,u\) (zero particular jump) |

\[
d_\mathrm{total}=d_\mathrm{beam}+d_\mathrm{tr}
\quad\text{on one periodic free set (linear-consistent)}.
\]

Tests: `tests/test_periodic_traction_superposition.py` (single-BVP identity, no
`build_clamp_supports`, factor reuse). Contour clamps remain only for CCX thin-wall /
diagnostic helpers that are not the production traction path.

---

## What is already correct

| Practice | Why |
|----------|-----|
| Optimiser Hashin FI = Channel A only with tip-integrated \(R(z)\) | No double-count; matches multi-scale BC story; `_run_section_shell_solve` omits traction kwargs |
| Native energy `K6` (MITC4), not ccx reaction `K6` | Package owns homogenisation; ccx benchmarks native formulations |
| Optional traction API for demos / local hydro | Valid engineering need — needs clean provenance + BC policy |
| Fatigue \(L_\mathrm{rec}\) built from six beam-only unit solves | Operators stay on the periodic free set |
| Spanwise recovery cache without `L_rec_tr` | Beam-mode cache path does not silently overlay traction |

---

## Recommended load matrix

| Use case | Channel A | Channel B | BC | Status |
|----------|-----------|-----------|-----|--------|
| Production FI / sizing | Full tip-integrated \(R(z)\) | Off | Periodic only | **Done** (optimiser) |
| Orchestration shell recovery | Tip-integrated \(R(z)\) | Off | Periodic only | **Done** (beam-only policy) |
| Standalone load-field API | \(T=m_x\), \(V_y=V_z=0\) when traction usable | Traction owns \(q\) | Shared periodic free set | **Done** (provenance + BC) |
| Fatigue σ ops | Full tip-integrated \(R\) via \(L_\mathrm{rec}\) | Residual \(L_\mathrm{rec,tr}^\mathrm{res}\) (net \(V_y/V_z\) removed) | Shared periodic free set | **Done** (residual + BC) |
| Avoid | \(V_y=q_y\) **and** `TractionField(q)` | — | Dual free-set add | Guarded in adapter |

### Verified path × channel × BC

| Path | Channel A | Channel B | Traction BC |
|------|-----------|-----------|-------------|
| `run_section_periodic_bc_from_load_field` | \(V_y=V_z=0\), \(T=m_x\), \(N=M_y=M_z=0\) when traction usable | Matched `TractionField` if points valid | Periodic \(T_\mathrm{pbc}\) (single BVP) |
| Orchestration shell stage | Tip-integrated \(R(z)\) (`tip_integrated_station_resultants`) | **Off** | N/A |
| Optimisation `shell_fi_eval` | Full integrated \(R(z)\) from `ExtremeLoads` | **Disabled** | N/A |
| Fatigue \(L_\mathrm{rec}\) / \(L_\mathrm{rec,tr}^\mathrm{res}\) | 6 unit beam solves | 2 unit \(q_y\)/\(q_z\) traction solves minus \(L_\mathrm{rec}[V_y,V_z]\) | Periodic \(T_\mathrm{pbc}\) |
| Spanwise recovery cache | Unit \(N=1\) beam modes | No `L_\mathrm{rec,tr}` | N/A |

---

## Non-goals

- Making CalculiX drive production homogenisation or “fixing” ref-node `RF` as beam conjugates
  (see [PERIODIC_CCX_HOMOGENIZATION.md](PERIODIC_CCX_HOMOGENIZATION.md)).  
- Soft traction penalties / Defect K as optimiser gates (accepted limitation on `shared`
  junctions — [SHELL_MODEL.md](SHELL_MODEL.md)).  
- Reviving the deleted clamped Path B FE as a production load path.

---

## Acceptance criteria

1. Documented, enforced policy: for every caller, Channel A and B are (a) residual / exclusive,
   with a unit test that equal \(q\) cannot appear in both. **Done**
   (`tests/test_traction_load_adapter.py`; fatigue residual partition in
   `blade_analysis/fatigue_damage/section_field/tests/test_surface_traction.py`).  
2. Orchestration handoff either uses tip-integrated \(R(z)\) without traction, or traction
   without stuffing the same \(q\) into \(V_y,V_z\). **Done** (tip-integrated \(R(z)\),
   Channel B off).  
3. Traction solve shares the periodic free set (or an explicitly named approximation with
   quantified error vs a one-shot periodic+traction solve). **Done**
   (`PeriodicReducedBVP` + `tests/test_periodic_traction_superposition.py`).  
4. Optimiser path remains beam-only; no behavioural change required there for this fix.
   **Done** (untouched).

---

## Summary

The multi-scale story is a **generalised section BVP**: beam-resultant periodic
homogenisation plus optional face traction as load channels on one free set.
**Problem 1** (load provenance) and **Problem 2** (shared periodic free set) are closed.
Fatigue residual partition keeps Channel B content zero-net under tip-integrated \(R\).
Keep optimiser FI on beam-only \(R(z)\). Nonlinear Newton remains out of scope.
