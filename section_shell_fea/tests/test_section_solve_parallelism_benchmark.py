"""Benchmark and profiling acceptance for section_shell_fea follow-ups."""

from __future__ import annotations

import warnings

import pytest

from blade_precompute.orchestration.precompute.containers import warn_nested_worker_oversubscription

_bench = pytest.importorskip(
    "blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.benchmark_section_solve_parallelism",
    reason="benchmark_section_solve_parallelism module not present on this tree",
)
assess_deferred_perf_items = _bench.assess_deferred_perf_items
benchmark_shell_fi_parallelism = _bench.benchmark_shell_fi_parallelism
profile_station_topology_cache_churn = _bench.profile_station_topology_cache_churn


def test_warn_nested_worker_oversubscription_emits_when_both_pools_active() -> None:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warn_nested_worker_oversubscription(
            design_n_workers=2,
            section_solve_n_workers=2,
            cpu_count=4,
        )
    assert any("Nested parallelism" in str(w.message) for w in rec)


def test_warn_nested_worker_oversubscription_emits_when_product_exceeds_cpu() -> None:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warn_nested_worker_oversubscription(
            design_n_workers=4,
            section_solve_n_workers=4,
            cpu_count=8,
        )
    assert any("exceeds cpu_count" in str(w.message) for w in rec)


def test_topology_cache_churn_profile_not_warranted_for_slsqp_like_loop() -> None:
    report = profile_station_topology_cache_churn(n_iterations=24)
    assert report.n_iterations == 24
    assert report.partial_invalidation_warranted is False


def test_deferred_perf_items_not_warranted_on_hot_paths() -> None:
    assessment = assess_deferred_perf_items()
    assert assessment["finding_8_cusp_penalty_analytical_derivatives"]["warranted"] is False
    assert assessment["finding_9_legacy_panel_local_bc_lil"]["warranted"] is False
    assert assessment["hsdt_element_gauss_batching"]["warranted"] is False


@pytest.mark.slow
def test_shell_fi_parallel_benchmark_reports_speedup_or_parity() -> None:
    """Real-process benchmark: parallel workers should not regress vs serial."""
    manifest = benchmark_shell_fi_parallelism(
        n_stations=4,
        n_elements_per_panel=3,
        repeats=1,
        parallel_workers=2,
    )
    assert manifest["serial"]["wall_s_mean"] > 0.0
    assert manifest["parallel"]["wall_s_mean"] > 0.0
    assert manifest["speedup"] > 0.5
    assert manifest["topology_cache_churn"]["partial_invalidation_warranted"] is False
