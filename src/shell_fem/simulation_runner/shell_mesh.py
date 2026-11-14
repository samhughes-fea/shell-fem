"""Accept either an FEAMesh or a blade mesh that holds one."""

from __future__ import annotations

from shell_fem.pre_processing.containers.mesh import FEAMesh


def as_shell_mesh(model) -> FEAMesh:
    """The shell mesh a vibration or buckling solve runs on.

    ``model`` is an ``FEAMesh``, or a blade mesh whose ``mesh`` attribute is one.
    Solids are refused: these solves do not add solid mass or solid geometric
    stiffness, and a blade for these results is a shell.
    """
    mesh = model
    if not hasattr(model, "shell_elements") and hasattr(model, "mesh"):
        mesh = model.mesh
    if not isinstance(mesh, FEAMesh):
        raise TypeError("expected an FEAMesh, or a blade mesh with a .mesh attribute")
    if mesh.has_solids:
        raise ValueError("vibration and buckling run on a shell mesh; this mesh also contains solids")
    if not mesh.shell_elements:
        raise ValueError("the mesh has no shell elements")
    return mesh
