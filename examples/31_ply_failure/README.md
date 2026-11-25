# 31 · Ply failure

The Tsai-Wu failure index of every ply of a full blade, under every load case, mapped
over the blade surface. The blade and loads are those of
[Example 29](../29_full_blade_loading/): a 10 m NACA blade from the kernel's shell mesh
(h = 0.1 m, 7534 elements) under aero, gravity and rotation, and ULS = 1.35 aero +
1.1 gravity + 1.0 rotation.

![Governing Tsai-Wu index over the suction and pressure sides as span-chord maps; the section through the worst point away from the root, with skins, caps and webs coloured by index; the worst index of each region in every span bay; and how often each ply governs](figures/ply_failure.png)

## How it is computed

```python
from shell_fem import TsaiWuCriterion
from shell_fem.simulation_runner.blade import BladeStaticRunner

results = BladeStaticRunner(blade, load_cases(blade), [ULS],
                            failure_criterion=TsaiWuCriterion()).execute().get_results()
results["ULS"].failure.worst()        # element, ply, index, strength ratio, mode, location
env = results.failure_envelope()      # worst case per element and ply
env.field.max_index, env.governing_case
```

`ply_failure_field` (`processing.static.results.ply_failure`) recovers the ply
stresses at every recovery point of every shell element. It samples each ply at its
bottom, middle and top, applies the criterion from laminate-constitutive-model, and
keeps the largest index for each element and ply, together with its strength ratio, mode
and location. It uses the same stress recovery as `ResultsExtractor`, and the tests
check that the two agree exactly. Each point's strains are computed once for all its
plies, so it runs about four times faster than `ResultsExtractor` (measured on a
1802-element version of this blade). Any criterion works:
`HashinCriterion`, `MaximumStressCriterion` and so on.

The maps use `post_processing.plots`:
- `surface_fringe` draws each element as a polygon on the span-chord plane, coloured
  by one value.
- `section_fringe` draws the elements crossing one span position as segments in the
  section plane.

Both need matplotlib (`pip install shell-fem[plots]`).

## Results

| Case | Max index | Strength ratio | Ply | Where |
|---|---|---|---|---|
| aero | 0.091 | 7.8 | 3 | pressure skin at the root |
| gravity | 0.025 | 27.7 | 3 | pressure skin at the root |
| rotation | 0.0006 | 1051 | 3 | skin at the root |
| ULS | 0.132 | 5.5 | 3 | pressure skin at the root |

Worst index by region over all cases (every governing case is ULS):

| Region | Max index | Strength ratio | Ply | z (m) |
|---|---|---|---|---|
| pressure skin | 0.132 | 5.5 | 3 | 0.01 |
| pressure cap | 0.123 | 5.8 | 19 | 0.01 |
| suction skin | 0.120 | 6.5 | 0 | 4.41 |
| webs | 0.106 | 7.0 | 3 | 5.58 |
| suction cap | 0.054 | 12.5 | 21 | 0.81 |

- **The largest indices sit at the clamp.** The rigid root stops the section warping
  and contracting, which raises the local stresses in the first bay (the spike at
  z = 0 in the span plot). A real root has a flange or an insert that this model doesn't
  have, so the example also reports the worst point beyond 0.5 m: the suction skin at
  mid-span, index 0.120,
  where the forward web meets it (the cross in the section panel).
- **Strength isn't what limits this blade.** The smallest strength ratio is 5.5, but
  the ULS buckling factor is about 1.57 ([Example 30](../30_mesh_convergence/)). The
  thin glass skin between the caps buckles long before any ply reaches its strength.
- **Away from the root, the glass governs.** Beyond the first bay, the suction skin
  and the webs reach indices of 0.11 to 0.12 and the suction cap only 0.054. Glass plies
  are weak transverse to their fibres, while the carbon caps carry the bending along
  theirs. Only the pressure (tension) cap reaches a comparable index, and only at the
  clamp. All the governing modes are Tsai-Wu's interactive mode.

## Limits

- First-ply failure on linear static stresses: no progressive damage, no buckling
  interaction, no stress concentrations at joints that the mid-surface model doesn't
  resolve.
- Through-thickness stresses come from FSDT, with transverse shear but no σz. Use
  `HashinCriterion` or `DelaminationOnsetCriterion` with an equilibrium recoverer for
  out-of-plane checks.
- The index at the root clamp depends on the mesh and the support model. Read the
  margins away from it.

Run with `python examples/31_ply_failure/ply_failure.py` (about two minutes; needs
implicit-geometry-kernel v0.9.5 and matplotlib).
