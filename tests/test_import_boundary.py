"""The shell FE's boundaries with the packages around it.

    implicit_geometry_kernel, laminate_constitutive_model  <-  shell_fem

shell-fem builds on the geometry kernel and the constitutive model. It never imports
the section model (anisotropic_section_model, which stays independent of shell
models) or the old package names. Every import is checked, including ones inside
functions.
"""

import ast
import subprocess
import sys
from pathlib import Path

PACKAGE = "shell_fem"
ROOT = Path(__file__).resolve().parents[1] / "src" / PACKAGE
FORBIDDEN = {"anisotropic_section_model", "laminate_modelling_package", "composite_laminate_model", "onedfem"}


def _imports(path):
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            yield node.module, node.lineno


def test_never_imports_the_section_model_or_old_names():
    found = [f"{p.relative_to(ROOT.parent)}:{line} imports {target}"
             for p in sorted(ROOT.rglob("*.py")) for target, line in _imports(p)
             if target.split(".")[0] in FORBIDDEN]
    assert found == []


def test_imports_without_the_section_model():
    code = ("import sys, shell_fem; "
            "print(sorted({m.split('.')[0] for m in sys.modules} & {'anisotropic_section_model', 'composite_laminate_model'}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_reexports_the_constitutive_model():
    import laminate_constitutive_model
    import shell_fem

    assert all(getattr(shell_fem, name) is getattr(laminate_constitutive_model, name)
               for name in laminate_constitutive_model.__all__)
