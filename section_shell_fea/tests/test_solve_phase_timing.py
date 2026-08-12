"""Unit tests for FE solve-phase collector → DiagnosticsBin."""

from __future__ import annotations

import time

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.solve_phase_timing import (
    PHASE_ASSEMBLE_GLOBAL_SYSTEM,
    PHASE_ELEMENT_FORCE_VECTORS,
    PHASE_ELEMENT_STIFFNESS_MATRICES,
    PHASE_PREPARE_LOCAL_SYSTEM,
    PHASE_PRIMARY_RESULTS,
    PHASE_QUATERNARY_RESULTS,
    PHASE_RECONSTRUCT_GLOBAL_FIELD,
    PHASE_REDUCE_CONSTRAINED_SYSTEM,
    PHASE_SECONDARY_RESULTS,
    PHASE_SOLVE_REDUCED_SYSTEM,
    PHASE_TERTIARY_RESULTS,
    PHASE_CONSTRAIN_GLOBAL_SYSTEM,
    SolvePhaseCollector,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.containers.diagnostics_bin import (
    DiagnosticsBin,
)


_FE_PHASES = (
    PHASE_ELEMENT_STIFFNESS_MATRICES,
    PHASE_ELEMENT_FORCE_VECTORS,
    PHASE_PREPARE_LOCAL_SYSTEM,
    PHASE_ASSEMBLE_GLOBAL_SYSTEM,
    PHASE_CONSTRAIN_GLOBAL_SYSTEM,
    PHASE_REDUCE_CONSTRAINED_SYSTEM,
    PHASE_SOLVE_REDUCED_SYSTEM,
    PHASE_RECONSTRUCT_GLOBAL_FIELD,
    PHASE_PRIMARY_RESULTS,
    PHASE_SECONDARY_RESULTS,
    PHASE_TERTIARY_RESULTS,
    PHASE_QUATERNARY_RESULTS,
)


def test_fe_phase_name_constants():
    assert _FE_PHASES == (
        "element_stiffness_matrices",
        "element_force_vectors",
        "prepare_local_system",
        "assemble_global_system",
        "constrain_global_system",
        "reduce_constrained_system",
        "solve_reduced_system",
        "reconstruct_global_field",
        "primary_results",
        "secondary_results",
        "tertiary_results",
        "quaternary_results",
    )


def test_collector_accumulates_phase_totals_and_freezes_diagnostics_bin():
    collector = SolvePhaseCollector()
    for name in _FE_PHASES:
        with collector.phase(name):
            time.sleep(0.0005)
    bin_ = collector.freeze()
    assert isinstance(bin_, DiagnosticsBin)
    assert bin_.wall_s > 0.0
    for name in _FE_PHASES:
        assert name in bin_.phase_totals_s
        assert bin_.phase_totals_s[name] > 0.0
        assert bin_.phase_counts[name] == 1


def test_collector_reset_clears_prior_phases():
    collector = SolvePhaseCollector()
    with collector.phase(PHASE_PRIMARY_RESULTS):
        pass
    collector.reset()
    with collector.phase(PHASE_QUATERNARY_RESULTS):
        pass
    bin_ = collector.freeze()
    assert PHASE_PRIMARY_RESULTS not in bin_.phase_totals_s
    assert PHASE_QUATERNARY_RESULTS in bin_.phase_totals_s


def test_diagnostics_bin_exported_from_package():
    from blade_structure_utilities.fem_utilities.section_shell_fea import (
        CascadeBin,
        DiagnosticsBin as Exported,
        SectionLinearStaticResult,
    )

    assert Exported is DiagnosticsBin
    assert CascadeBin is SectionLinearStaticResult
