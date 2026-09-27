"""Compatibility shim — prefer ``blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.solve.solve_phase_timing``.

See ``docs/DEPRECATED_ALIASES.md``.

This module is an alias of the canonical implementation so private and
public symbols (including test-only ``_*`` helpers) remain importable.
"""
from __future__ import annotations

from importlib import import_module
import sys as _sys

_canonical = import_module("blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.solve.solve_phase_timing")
_sys.modules[__name__] = _canonical
