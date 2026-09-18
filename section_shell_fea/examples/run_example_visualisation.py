from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_renderer() -> callable:
    package_root = Path(__file__).resolve().parents[1]
    module_path = package_root / "visualisation" / "example_plot.py"
    spec = spec_from_file_location("visualisation.example_plot", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load visualisation module: {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    renderer = getattr(module, "render_demo_visualisation", None)
    if renderer is None:
        raise AttributeError(
            f"visualisation module missing render_demo_visualisation(): {module_path}"
        )
    return renderer


def main() -> None:
    render_demo_visualisation = _load_renderer()
    output = Path(__file__).resolve().parents[1] / "visualisation" / "output" / "example_visualisation.png"
    artefact = render_demo_visualisation(output)
    print(f"[demo] wrote {artefact}")


if __name__ == "__main__":
    main()