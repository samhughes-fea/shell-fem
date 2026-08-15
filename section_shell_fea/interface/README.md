# interface — FE handoff adapters (not a stage home)

↑ Parent: [`../README.md`](../README.md) · Computable set: [`../README.md` §2](../README.md#2-computable-set) ·
Glossary: [`../../../docs/CASCADE_GLOSSARY.md`](../../../docs/CASCADE_GLOSSARY.md)

Auxiliary adapters between station/BSP callers and the FE worker. **Not** a
`pre_processing/` / `processing/` stage — do not put formulation maths here.

## Purpose / ownership

| Owns | Does **not** own |
|------|------------------|
| Station analysis builders / smoke builders | MITC4 `Ke` / assemble / solve |
| Constitutive / traction load adapters into FE inputs | SFV shaping (`section_loads`) |
| Panel identity / full-element result helpers | Geometry / mesh / bind policy |

## Exact I/O (summary)

Callers build station analysis packages and feed
`SectionLinearStaticRunner` / dispatch. Prefer package barrel +
`simulation_runner/` over deep-importing this tree from BSP.

## See also

- [`../simulation_runner/README.md`](../simulation_runner/README.md)
- [`../pre_processing/loading/README.md`](../pre_processing/loading/README.md)
- [`../pre_processing/constitutive/README.md`](../pre_processing/constitutive/README.md)
