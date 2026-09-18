"""Smoke: fringe auto-workers resolve via allocate_core_budget."""

from __future__ import annotations

from blade_structure_utilities.parallel_utilities.core_budget import allocate_core_budget
from blade_structure_utilities.fem_utilities.section_shell_fea.visualisation.mesh_fringe import (
    run_elemental_fringe_jobs,
)


def test_fringe_auto_matches_budget(monkeypatch) -> None:
    expected = allocate_core_budget(
        n_outer_tasks=10, allow_nested_inner=False
    ).outer_workers
    if expected <= 1:
        return

    captured: dict[str, int] = {}

    class _F:
        def result(self):
            return None

    class _Pool:
        def __init__(self, *, max_workers=1):
            captured["max_workers"] = int(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, job):
            return _F()

    monkeypatch.setattr(
        "blade_structure_utilities.fem_utilities.section_shell_fea.visualisation.mesh_fringe.ProcessPoolExecutor",
        _Pool,
    )
    monkeypatch.setattr(
        "blade_structure_utilities.fem_utilities.section_shell_fea.visualisation.mesh_fringe.as_completed",
        lambda futs: list(futs),
    )
    run_elemental_fringe_jobs([{"_i": i} for i in range(10)], workers=None)
    assert captured["max_workers"] == expected


def test_fringe_workers_one_stays_serial(monkeypatch) -> None:
    called = {"pool": False}

    def _boom(*a, **k):
        called["pool"] = True
        raise AssertionError("ProcessPoolExecutor must not run when workers=1")

    monkeypatch.setattr(
        "blade_structure_utilities.fem_utilities.section_shell_fea.visualisation.mesh_fringe.ProcessPoolExecutor",
        _boom,
    )

    class _Canvas:
        def render_elemental(self, **kwargs):
            return None

        def close(self):
            return None

    monkeypatch.setattr(
        "blade_structure_utilities.fem_utilities.section_shell_fea.visualisation.mesh_fringe.FringeCanvas",
        lambda **k: _Canvas(),
    )
    jobs = [
        {
            "keep_poly": [],
            "keep_seg": [],
            "face_vals": [0.0],
            "seg_vals": [0.0],
            "output_path": "x.png",
            "title": "t",
            "colorbar_label": "c",
        }
        for _ in range(3)
    ]
    run_elemental_fringe_jobs(jobs, workers=1)
    assert called["pool"] is False
