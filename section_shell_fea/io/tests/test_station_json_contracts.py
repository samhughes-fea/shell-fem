"""Contract tests for station JSON v1/v2 and prepass persistence v1."""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.io.job_outputs import (
    build_section_shell_station_json_payload,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.io.prepass_station_analysis_io import (
    PREPASS_SCHEMA,
    load_prepass_station_analysis,
    write_prepass_station_analysis,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.io.station_json import write_station_json


def _clpt():
    return SimpleNamespace(
        fi=np.array([0.1, 0.2]),
        eps0=np.zeros(3),
        kappa=np.zeros(3),
        N_vec=np.zeros(3),
        M_vec=np.zeros(3),
        sig_lam_mid=[np.zeros(3)],
        eps_lam_mid=[np.zeros(3)],
    )


def _zone():
    return SimpleNamespace(
        label="skin",
        s=np.array([0.0, 1.0]),
        nodes=np.array([[0.0, 0.0], [1.0, 0.0]]),
    )


def test_section_shell_station_v1_failure_assessment_and_aliases():
    gov = {
        "fi": 0.25,
        "utilization": 0.5,
        "status": "available",
        "criterion": "hashin",
        "criterion_id": "hashin_1980_plane_stress",
    }
    q = SimpleNamespace(
        governing=gov,
        max_utilization_governing=0.5,
        formulation_id="TEST-CACHE-FSDT",
        needs_review=np.array([False]),
    )
    bundle = SimpleNamespace(
        zones=[_zone()],
        q_tot=[np.array([0.0, 0.0])],
        sig_p=[np.array([0.0, 0.0])],
        sig_b=[np.array([0.0, 0.0])],
        y_e=0.1,
        z_e=0.2,
        y_sc=0.1,
        z_sc=0.2,
        webs_geom=[],
        booms=[],
        quaternary_results=q,
        all_panel_results=[[]],
    )
    payload = build_section_shell_station_json_payload(
        station_tag="s0",
        airfoil=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.1], [0.0, 0.0]]),
        spars=[0.35],
        n_elements_per_panel=4,
        bundle=bundle,
        strengths={"Xt": 1.0},
        clpt_reference_skin=_clpt(),
        all_panel_results=[[]],
        all_panel_fi=[np.array([0.25])],
        panel_labels_m=["skin"],
        sig_omega_reference_panel=None,
    )
    json.dumps(payload)
    assert payload["schema"] == "section_shell_station_v1"
    audit = payload["failure_assessment"]
    alias = payload["fsdt_periodic_hashin_audit"]
    assert audit == alias
    assert audit["formulation_id"] == "TEST-CACHE-FSDT"
    assert audit["governing_criterion"] == "hashin_1980_plane_stress"
    assert audit["governing_status"] == "available"
    assert audit["max_utilization_governing"] == pytest.approx(0.5)
    assert audit["max_fi_hashin"] == pytest.approx(0.25)
    assert "clpt_reference_skin_station" in payload


def test_section_shell_station_v2_json_safe_required_keys(tmp_path: Path):
    mesh = SimpleNamespace(
        panel_meshes=[
            SimpleNamespace(
                panel_label="skin",
                kind="skin",
                arc_length_m=1.0,
                thickness_m=0.002,
                n_elements=4,
            )
        ],
        layout_key="0A",
        chord_m=1.0,
        twist_rad=0.1,
        panels=[object()],
        n_total_nodes=8,
        n_total_elements=4,
        clusters=[],
        summary=lambda: {"n_elements": 4},
    )
    path = write_station_json(
        tmp_path,
        station_tag="root",
        mesh=mesh,
        N=1.0,
        Vy=2.0,
        Vz=3.0,
        My=4.0,
        Mz=5.0,
        T=6.0,
        n_elements_per_panel=4,
        mesh_element_length_frac=0.012,
        failure_assessment={
            "max_fi_hashin": 0.2,
            "max_utilization_governing": 0.45,
            "governing_status": "available",
        },
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "section_shell_station_v2"
    for key in (
        "station_tag",
        "unit_section_resultants",
        "n_elements_per_panel",
        "layout_key",
        "thin_wall",
        "mesh_summary",
    ):
        assert key in payload
    assert payload["failure_assessment"] == payload["fsdt_periodic_hashin"]
    assert payload["unit_section_resultants"]["Vy_N"] == pytest.approx(2.0)


def test_prepass_station_analysis_v1_round_trip(tmp_path: Path):
    obj = SimpleNamespace(station_index=3, label="root")
    manifest = write_prepass_station_analysis(
        tmp_path / "st", obj, station_index=3, r_z_m=0.4
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema"] == PREPASS_SCHEMA
    loaded = load_prepass_station_analysis(manifest)
    assert loaded.station_index == 3
    assert loaded.label == "root"


def test_prepass_station_analysis_v1_rejects_wrong_schema(tmp_path: Path):
    manifest = tmp_path / "prepass_station_analysis.json"
    manifest.write_text(json.dumps({"schema": "not_a_schema", "pkl": "x.pkl"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported prepass"):
        load_prepass_station_analysis(manifest)


def test_prepass_station_analysis_v1_missing_sidecar(tmp_path: Path):
    manifest = write_prepass_station_analysis(
        tmp_path / "st", SimpleNamespace(ok=True), station_index=0, r_z_m=0.0
    )
    (manifest.parent / "prepass_station_analysis.pkl").unlink()
    with pytest.raises(FileNotFoundError):
        load_prepass_station_analysis(manifest)
