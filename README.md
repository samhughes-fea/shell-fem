# section_shell_fea (reference fork from blade-structure)

This branch holds blade-structure's section shell finite-element package,
`blade_structure_utilities.fem_utilities.section_shell_fea`, with its git history
(90 commits from 2026-07-22). It was extracted on 2026-11-05 from blade-structure
`main` at 51ba12f.

It is a reference to read, diff and port from into `shell_fem`. It is never merged
into `main`, and it does not run on its own: it imports blade-structure's
`section_mesh`, `section_geometry`, `section_loads`, `section_contract` and
`material_utilities`.

Earlier locations of the same code are kept under their own names in the history:
`section_shell_model/` (blade_structure_utilities/section_shell_model and
fem_utilities/section_shell_model) and `archive/fastblade_section_shell_model/`.
