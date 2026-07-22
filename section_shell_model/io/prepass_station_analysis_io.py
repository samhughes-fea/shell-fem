"""Prepass station-analysis persistence for parallel orchestration workers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from blade_structure_utilities.section_shell_model._utils.jsonutil import write_json

PREPASS_MANIFEST_NAME = "prepass_station_analysis.json"
PREPASS_PICKLE_NAME = "prepass_station_analysis.pkl"
PREPASS_SCHEMA = "prepass_station_analysis_v1"


def write_prepass_station_analysis(
    station_dir: Path,
    station_analysis: Any,
    *,
    station_index: int,
    r_z_m: float,
) -> Path:
    """Write station analysis to disk; return JSON manifest path (not the pickle)."""
    station_dir = Path(station_dir).resolve()
    station_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = station_dir / PREPASS_PICKLE_NAME
    with open(pkl_path, "wb") as fh:
        pickle.dump(station_analysis, fh, protocol=pickle.HIGHEST_PROTOCOL)
    manifest_path = station_dir / PREPASS_MANIFEST_NAME
    write_json(
        manifest_path,
        {
            "schema": PREPASS_SCHEMA,
            "station_index": int(station_index),
            "r_z_m": float(r_z_m),
            "pkl": PREPASS_PICKLE_NAME,
        },
    )
    return manifest_path


def load_prepass_station_analysis(manifest_path: Path) -> Any:
    """Reload ``ShellSectionStationAnalysis`` written by :func:`write_prepass_station_analysis`."""
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != PREPASS_SCHEMA:
        raise ValueError(f"unsupported prepass station analysis schema: {payload.get('schema')!r}")
    pkl_path = manifest_path.parent / str(payload["pkl"])
    with open(pkl_path, "rb") as fh:
        return pickle.load(fh)


__all__ = [
    "PREPASS_MANIFEST_NAME",
    "PREPASS_PICKLE_NAME",
    "PREPASS_SCHEMA",
    "load_prepass_station_analysis",
    "write_prepass_station_analysis",
]
