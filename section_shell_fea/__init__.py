'''FE kernel worker public barrel.'''

from __future__ import annotations

from typing import Any

__version__ = '0.1.0'

__all__ = [
    'SectionLinearStaticRunner',
    'SectionLinearStaticResult',
    'CascadeBin',
    'DiagnosticsBin',
    'LinearBucklingCascadeResult',
    'run_section_linear_buckling_from_static',
]


def __getattr__(name: str) -> Any:
    if name in ('SectionLinearStaticRunner', 'SectionLinearStaticResult', 'CascadeBin'):
        from .simulation_runner.section_linear_static_runner import (
            CascadeBin,
            SectionLinearStaticResult,
            SectionLinearStaticRunner,
        )

        return {
            'SectionLinearStaticRunner': SectionLinearStaticRunner,
            'SectionLinearStaticResult': SectionLinearStaticResult,
            'CascadeBin': CascadeBin,
        }[name]
    if name == 'DiagnosticsBin':
        from .processing.static.results.containers.diagnostics_bin import DiagnosticsBin

        return DiagnosticsBin
    if name in ('LinearBucklingCascadeResult', 'run_section_linear_buckling_from_static'):
        from .simulation_runner.api.linear_buckling import (
            LinearBucklingCascadeResult,
            run_section_linear_buckling_from_static,
        )

        return {
            'LinearBucklingCascadeResult': LinearBucklingCascadeResult,
            'run_section_linear_buckling_from_static': run_section_linear_buckling_from_static,
        }[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
