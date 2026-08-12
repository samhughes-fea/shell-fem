"""Pre-solve package: agnostic DTOs, element kernels, meshing, and loading.

Mirrors the 1d-fem ``pre_processing/`` role for the pre-solve half of
``section_shell_fea``. See ``docs/PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md``
for the full rationale and migration plan.

Subpackages:

- ``containers`` — agnostic pre-solve DTOs and formulation state objects
  (direct 1d-fem analogue).
- ``element_library`` — native element kernels (mitc4) and factories
  (direct 1d-fem analogue).
- ``mesh`` — consolidated meshing logic (convention-only home borrowing
  1d-fem placement/naming).
- ``loading`` — absorbed ``blade_precompute.section_loads`` package
  (convention-only home borrowing 1d-fem placement/naming).
"""
