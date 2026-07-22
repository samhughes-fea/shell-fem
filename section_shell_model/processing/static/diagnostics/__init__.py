"""Diagnostic helpers and provenance tooling for the solve pipeline.

A ``blade_precompute``-specific extension (no direct 1d-fem counterpart at
this path) grouping diagnostic-flavoured code that supports the solve
pipeline but is not a stage operation or a results container. Cross-solver
oracles remain in top-level ``validation/``.

See ``docs/PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md`` for the full rationale.
"""
