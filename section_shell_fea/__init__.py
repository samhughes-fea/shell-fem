"""
FE kernel worker: element ``Ke``/``Fe`` → assemble / solve → cascade bin.

Use-case agnostic. Prefer::

    bin = SectionLinearStaticRunner(...).execute().get_results()  # CascadeBin / P→S→T→Q

Public re-exports are loaded lazily so mid-refactor package moves do not break
submodule imports (diagnostics / runner / cascade).
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = [
    "SectionLinearStaticRunner",
    "SectionLinearStaticResult",
    "CascadeBin",
    "DiagnosticsBin",
]


def __getattr__(name: str) -> Any:
    if name in ("SectionLinearStaticRunner", "SectionLinearStaticResult", "CascadeBin"):
        from .simulation_runner.section_linear_static_runner import (
            CascadeBin,
            SectionLinearStaticResult,
            SectionLinearStaticRunner,
        )

        return {
            "SectionLinearStaticRunner": SectionLinearStaticRunner,
            "SectionLinearStaticResult": SectionLinearStaticResult,
            "CascadeBin": CascadeBin,
        }[name]
    # Riders read the diagnostics off the cascade bin, so the container belongs
    # in the barrel beside its parent rather than behind a five-deep import path.
    if name == "DiagnosticsBin":
        from .processing.static.results.containers.diagnostics_bin import DiagnosticsBin

        return DiagnosticsBin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
