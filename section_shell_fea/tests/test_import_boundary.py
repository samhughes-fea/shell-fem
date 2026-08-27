from pathlib import Path


def test_contract_homes_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "api").is_dir()
    assert (root / "types").is_dir()
