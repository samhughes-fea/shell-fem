# 30 · Mesh convergence

How much of a full-blade result is discretisation error? This example puts the Example 29
blade and loads on a ladder of five systematically refined kernel meshes and estimates
the error of each result with the ASME grid convergence index (GCI) of Celik et al.
(2008), *J. Fluids Eng.* 130(7), 078001.

![Tip flap deflection, tip twist and ULS buckling load factor against element size on the five meshes, with the Richardson extrapolation and the GCI of the finest mesh where the convergence is reliable; below, the distance of each mesh from the extrapolated value on log axes with the observed order as a slope, and for the tip twist the change from each mesh to the next](figures/mesh_convergence.png)

## The mesh ladder

The GCI assumes every element shrinks by the same ratio from one mesh to the next. The
kernel's edge-length option `shell_mesh(geometry, h=...)` doesn't refine that way here:
span bays may fall anywhere in [h/2, h], and chordwise counts are rounded per region. On
this blade, h = 0.1 gives 156 span stations and h = 0.07 gives 144. The ladder therefore
uses counts. Level k has 10k span bays and 2k elements across every chordwise region
(`shell_mesh(geometry, n_span=10k + 1, n_chord=2k + 1)`), so the element count grows as k²:

| k | h (m) | DOFs | run (s) |
|---|---|---|---|
| 2 | 0.151 | 4,332 | 10 |
| 3 | 0.099 | 10,224 | 21 |
| 4 | 0.076 | 17,496 | 38 |
| 6 | 0.051 | 39,486 | 96 |
| 8 | 0.038 | 70,296 | 171 |

h is the representative size (A/N)^½ from `processing.diagnostics.mesh_size`. The
refinement ratios are 1.53, 1.30, 1.50 and 1.33, all above the 1.3 that Celik et al.
recommend.

```python
from shell_fem.simulation_runner.convergence import blade_convergence_study

study = blade_convergence_study(lambda k: blade(n_span=10 * k + 1, n_chord=2 * k + 1), load_cases, "ULS",
                                [2, 3, 4, 6, 8], combinations=[ULS], buckling_modes=1)
results = study.execute().get_results()
results.gci("buckling").extrapolated, results.gci("buckling").gci, results.reliable("buckling")
```

## Results (ULS, GCI over the three finest meshes)

| Result | Finest mesh | Extrapolated | Observed order p | GCI fine |
|---|---|---|---|---|
| tip flap deflection | 339.95 mm | 339.84 mm | 2.25 | 0.04 % |
| tip edge deflection | 6.985 mm | 6.969 mm | 2.26 | 0.28 % |
| tip twist | −0.3749° | — | — | not reliable |
| root Mx | 103.71 kN m | 103.71 kN m | 1.28 | 0.009 % |
| ULS buckling load factor | 1.633 | 1.568 | 2.04 | 5.0 % |

- **Deflection converges at second order** (p ≈ 2.25 on both tip deflections), as
  expected of these elements. The finest mesh is within 0.04 % of the limit, and even the
  coarsest (4k DOFs) is within 0.7 %.
- **Root loads hardly depend on the mesh.** The root reactions balance the applied
  loads exactly, and those only change through how each mesh's span bays sample the
  spanwise distributions. The differences are 0.03 % or less.
- **Tip twist hasn't settled.** It moves by about 0.6 % over the ladder but not
  monotonically: its magnitude shrinks from k = 2 to 4 and grows again at k = 6 and 8,
  so the finer pair differs by more than the coarser pair and no positive order fits.
  `results.reliable("tip_twist")` is False, the table marks it with `?`, and the figure
  shows the change from mesh to mesh instead of an extrapolation. The last three meshes
  agree to 0.0004° (0.1 %), which is the honest error bar until a finer mesh shows a
  trend.
- **The buckling factor is the result to watch.** It converges cleanly at second order,
  but from far away. The Example 29 mesh (`h = 0.1`, 43k DOFs) gives 1.76, and the
  extrapolated limit is 1.57, so that mesh overstates the buckling margin by about 12 %.
  The local panel mode needs several elements across each half-wave, which a mesh sized
  for global deflection doesn't give it. Take buckling factors from a converged ladder,
  or from the extrapolated value with its GCI.

## Is the procedure right?

`grid_convergence_index` recovers the order and the limit of φ = φ₀ + C hᵖ exactly, for
uniform and non-uniform refinement ratios (`tests/test_mesh_convergence.py`). On a tube
under a uniform torque per span (Quad4, four meshes from 64 to 676 elements), the
extrapolated tip twist is within 0.07 % of beam theory, mL²/(2GJ), and the GCI band of
the finest mesh contains it:

```
fine 2.404426e-04  extrapolated 2.366326e-04  p = 2.13  GCI 1.98 %
band [2.356801e-04, 2.452050e-04]   beam theory 2.364588e-04
```

## Reading a GCI

- `gci` is relative to the finest value, with a safety factor of 1.25 for three meshes
  and an observed order (use `order=`, `safety_factor=3` for a fixed formal order).
- `oscillatory` means the differences change sign between meshes. The estimate is
  weaker, and the table marks it with `*`.
- `reliable` asks that a positive order fit the finest triple and, with four or more
  meshes, that the finest two triples agree on p within 30 %.
- Results fixed by equilibrium or zero by symmetry report `converged` with a GCI of
  zero. `atol` sets what counts as no change for a result that is zero.

Run with `python examples/30_mesh_convergence/mesh_convergence.py` (about six minutes;
needs implicit-geometry-kernel v0.9.5).
