"""Strict-shape validation for ShellFormulationCache (re-export).

Canonical implementation: ``pre_processing.element_library.formulation_validation``.
Kept here so historical ``processing.diagnostics`` imports continue to work.
"""

from __future__ import annotations

from ...pre_processing.element_library.formulation_validation import validate_formulation_cache_shapes

__all__ = ["validate_formulation_cache_shapes"]
