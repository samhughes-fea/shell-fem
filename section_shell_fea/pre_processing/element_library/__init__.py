"""Shell element formulations plug-in library (native MITC4 / FSDT; external CalculiX ccx opt-in).

Meshing, assembly spine, and periodic homogenisation are element-agnostic; they
select a kernel from :mod:`.kernel` at the stiffness/resultant-recovery seam.

Theory registries and station dispatch live in
:mod:`blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.dispatch`.
"""

from .contracts import (
    AnalysisDofLayout,
    ShellFormulationDescriptor,
    mitc4_fsdt_descriptor,
)
from .kernel import (
    MITC4_KERNEL,
    Mitc4StripKernel,
    StripElementKernel,
    kernel_provenance_kind,
    resolve_kernel,
)
from .registry import (
    default_formulation_id,
    ensure_default_formulations,
    register_formulation,
    resolve_formulation,
)

__all__ = [
    "AnalysisDofLayout",
    "MITC4_KERNEL",
    "Mitc4StripKernel",
    "ShellFormulationDescriptor",
    "StripElementKernel",
    "default_formulation_id",
    "ensure_default_formulations",
    "kernel_provenance_kind",
    "mitc4_fsdt_descriptor",
    "register_formulation",
    "resolve_formulation",
    "resolve_kernel",
]
