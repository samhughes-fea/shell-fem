"""Compatibility shim — prefer ``blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.equilibrium.equilibrium_checks``.

See ``docs/DEPRECATED_ALIASES.md``.

This module is an alias of the canonical implementation so private and
public symbols (including test-only ``_*`` helpers) remain importable.
"""
from __future__ import annotations

from importlib import import_module
import sys as _sys

_canonical = import_module("blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.equilibrium.equilibrium_checks")
_sys.modules[__name__] = _canonical
