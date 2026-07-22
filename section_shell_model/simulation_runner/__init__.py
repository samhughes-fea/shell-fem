"""Simulation runners for the section shell model.

Only two runners: linear-static (main FI) and linear-buckling (P→S).
"""

from __future__ import annotations

from .section_linear_buckling_runner import (
    SectionLinearBucklingResult,
    SectionLinearBucklingRunner,
)
from .section_linear_static_runner import (
    SectionLinearStaticResult,
    SectionLinearStaticRunner,
    section_solve_legacy_dict,
)

__all__ = [
    "SectionLinearBucklingResult",
    "SectionLinearBucklingRunner",
    "SectionLinearStaticResult",
    "SectionLinearStaticRunner",
    "section_solve_legacy_dict",
]
