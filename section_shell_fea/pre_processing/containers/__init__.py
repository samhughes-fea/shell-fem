"""Agnostic pre-solve DTOs and formulation state objects.

Direct 1d-fem analogue of ``pre_processing/containers`` — the mirror of
``processing/static/results/containers`` on the input side.

Handoff DTOs live in leaf modules (``station_types``, ``recovery_types``,
``buckling_types``, ``station_analysis_types``, ``shell_theory_capabilities``);
:mod:`.types` re-exports them for compatibility.
"""
