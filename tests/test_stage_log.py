"""Shell stages report on the shared logger."""

from __future__ import annotations

from pathlib import Path

import pytest

from shell_fem.processing.stage_log import (
    LOGGER_NAME,
    StageCollector,
    StageReport,
    collecting,
    fold_reports,
    stage,
)


def test_a_stage_records_seconds_and_an_error_keeps_its_time() -> None:
    with collecting() as sink:
        with stage("shell_fem", "kept"):
            pass
        with pytest.raises(RuntimeError):
            with stage("shell_fem", "failed"):
                raise RuntimeError("stage failed")
    folded = fold_reports(sink.take())
    assert folded["kept"] >= 0.0
    failed = folded["failed"]
    assert isinstance(failed, dict)
    assert failed["seconds"] >= 0.0
    assert failed["error"] == "stage failed"


def test_public_contract() -> None:
    import dataclasses
    import logging

    assert LOGGER_NAME == "blade.stage"
    assert [item.name for item in dataclasses.fields(StageReport)] == [
        "package", "stage", "seconds", "error", "depth", "started",
        "cpu_seconds", "measures"]
    assert callable(stage) and callable(fold_reports)
    assert issubclass(StageCollector, logging.Handler)


def test_stage_log_stays_out_of_the_package_root() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "shell_fem"
    assert sorted(path.name for path in root.glob("*.py")) == ["__init__.py", "_legacy.py"]
    assert (root / "processing" / "stage_log.py").is_file()
