"""Compat shim — prefer ``fem_utilities.section_shell_fea``.

Deep submodule imports under this package name are not provided; update
callers to ``blade_structure_utilities.fem_utilities.section_shell_fea``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "blade_structure_utilities.fem_utilities.section_shell_model is deprecated; "
    "import from blade_structure_utilities.fem_utilities.section_shell_fea instead.",
    DeprecationWarning,
    stacklevel=2,
)

from blade_structure_utilities.fem_utilities.section_shell_fea import *  # noqa: F403
from blade_structure_utilities.fem_utilities import section_shell_fea as _fea

__all__ = list(getattr(_fea, "__all__", []))
