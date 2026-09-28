from pathlib import Path


def test_contract_homes_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "api").is_dir()
    assert not (root / "types").exists()
    assert (root / "pre_processing").is_dir()
    assert (root / "processing").is_dir()
    assert (root / "simulation_runner").is_dir()
