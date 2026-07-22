"""Shell element formulations plug-in library (native MITC4 / FSDT; external CalculiX ccx opt-in).

Meshing, assembly spine, and periodic homogenisation are element-agnostic; they
select a kernel from :mod:`.kernel` at the stiffness/resultant-recovery seam.
"""

from .kernel import (
    MITC4_KERNEL,
    Mitc4StripKernel,
    StripElementKernel,
    kernel_provenance_kind,
    resolve_kernel,
)

__all__ = [
    "MITC4_KERNEL",
    "Mitc4StripKernel",
    "StripElementKernel",
    "kernel_provenance_kind",
    "resolve_kernel",
]
