"""Compat alias — import FE mesh adaptation from ``pre_processing.mesh``.

Logic-free warning aliases for one migration cycle. Canonical home is
``pre_processing.mesh`` (FE maps, numbering, ``BoundSectionShellInputs``).
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Import FE mesh adaptation from "
    "blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh "
    "(topology is a reverse-compatibility alias until 2026-11-30).",
    DeprecationWarning,
    stacklevel=2,
)

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh import (  # noqa: E402,F401
    BoundSectionShellInputs,
    TaperedBoundSectionShellInputs,
    bind_section_shell_inputs,
)

__all__ = [
    "BoundSectionShellInputs",
    "TaperedBoundSectionShellInputs",
    "bind_section_shell_inputs",
]
