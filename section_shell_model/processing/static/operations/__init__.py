"""Lean stage classes for the section static solve (mirror 1d-fem operations).

**Stage ops** (1d-fem analogues): prepare / assemble / constrain / reduce / solve /
reconstruct / disassemble — invoked by ``SectionLinearStaticRunner``.

**Assembly orchestra** (blade geometry, also in this package): ``section_assembly``,
``global_shell_assembly``, ``connected_*``, ``physics_contract`` — build
``SectionAssemblyResult`` / formulation cache for the runner; not a second solve spine.

See ``docs/SECTION_SOLVE_STAGES.md``.
"""

from __future__ import annotations

from .assemble_global import AssembleGlobalShellSystem, assemble_section_system
from .constrain_global import ConstrainGlobalShellSystem
from .disassemble_field import DisassembleGlobalShellField
from .global_shell_assembly import assemble_global_shell
from .prepare_local import PrepareLocalShellSystem
from .reconstruct_field import ReconstructGlobalShellField
from .reduce_constrained import ReduceConstrainedShellSystem
from .section_assembly import SectionAssemblyResult, assemble_section_from_panels
from .solve_reduced import SolveReducedShellSystem

__all__ = [
    "AssembleGlobalShellSystem",
    "assemble_global_shell",
    "assemble_section_from_panels",
    "assemble_section_system",
    "ConstrainGlobalShellSystem",
    "DisassembleGlobalShellField",
    "PrepareLocalShellSystem",
    "ReconstructGlobalShellField",
    "ReduceConstrainedShellSystem",
    "SectionAssemblyResult",
    "SolveReducedShellSystem",
]
