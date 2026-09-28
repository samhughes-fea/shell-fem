"""Compatibility shim — moved to ``reference_thin_wall.section_vlasov``.

Canonical home:
``blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.reference_thin_wall.section_vlasov``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Import from blade_structure_utilities.fem_utilities.section_shell_fea."
    "pre_processing.reference_thin_wall.section_vlasov. "
    "loading.thin_wall.section_vlasov is a temporary re-export.",
    DeprecationWarning,
    stacklevel=2,
)

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.reference_thin_wall.section_vlasov import (  # noqa: E402,F401
    SectionVlasovResult,
    compute_section_vlasov,
)

__all__ = [
    "SectionVlasovResult",
    "compute_section_vlasov",
]
