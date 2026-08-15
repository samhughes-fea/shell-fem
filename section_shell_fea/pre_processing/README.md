# pre_processing — FE prep homes

Constitutive · loading · element_library · mesh · containers · dispatch.

↑ Parent: [`../README.md`](../README.md) · Docs index: [`../docs/README.md`](../docs/README.md)

**Layer contract:** prepare inputs only. Does **not** assemble, constrain, solve,
or fill cascade tiers (those live under [`../processing/`](../processing/README.md)).

## Homes

| Home | Owns | README |
|------|------|--------|
| [`loading/`](loading/README.md) | Apply SFV → nodal `f` / `Fe` | equations + I/O |
| [`element_library/`](element_library/README.md) | MITC4 / FSDT `Ke` | equations + I/O |
| [`constitutive/`](constitutive/README.md) | Panel ABD stacks from bound laminates | equations + I/O |
| [`mesh/`](mesh/README.md) | FE mesh map / `ShellPanelTopology` | equations + I/O |
| [`containers/`](containers/README.md) | FE-local DTOs / theory capabilities | I/O |
| [`dispatch/`](dispatch/README.md) | Theory / solve registries | I/O |

## Pipeline pointer

\[
(\mathrm{zones},\,\mathrm{SFV},\,\mathrm{mesh})
\;\xrightarrow{\mathrm{prep}}\;
\{\mathbf{f},\,\mathbf{K}_e,\,\mathrm{ABD},\,\mathrm{topology}\}
\;\xrightarrow{\mathrm{processing}}\;
\mathbf{KU}=\mathbf{F}.
\]

```mermaid
flowchart LR
  SFV[SectionForceVector] --> loading
  zones[MaterialZone] --> constitutive
  strip[SectionStripMesh] --> mesh
  constitutive --> element_library
  loading --> runner[simulation_runner]
  element_library --> runner
  mesh --> runner
  containers --> runner
  dispatch --> runner
```

## PNG pointers

| Figure | Stage |
|--------|-------|
| [`../visualisation/output/section_centerline_mesh_v2/section_mesh_2C-F_i000.png`](../visualisation/output/section_centerline_mesh_v2/section_mesh_2C-F_i000.png) | Mesh / topology consumed after prep |
| Loads shaping (upstream) | [`section_loads` viz](../../../load_utilities/section_loads/visualisation/README.md) — not owned here |

## See also

- [`../docs/SECTION_SOLVE_STAGES.md`](../docs/SECTION_SOLVE_STAGES.md)
- [`../docs/SHELL_MODEL.md`](../docs/SHELL_MODEL.md)
- [`../processing/README.md`](../processing/README.md)
