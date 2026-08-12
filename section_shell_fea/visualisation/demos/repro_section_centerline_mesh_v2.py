"""Deprecated redirect — centerline taxonomy demos moved out of FE.

Geometry (40-key SDF registry sweep)::

    python blade_structure_utilities/geometry_utilities/section_geometry/visualisation/demos/section_centerline_geometry_taxonomy.py

Mesh (strip / panel / element figures)::

    python blade_structure_utilities/mesh_utilities/section_mesh/visualisation/demos/section_centerline_mesh.py
"""

from __future__ import annotations

import sys


def main() -> None:
    from blade_structure_utilities.mesh_utilities.section_mesh.visualisation.demos.section_centerline_mesh import (
        main as _mesh_main,
    )

    print(
        "[deprecated] repro_section_centerline_mesh_v2 → "
        "section_mesh/visualisation/demos/section_centerline_mesh.py "
        "(geometry registry: section_geometry/.../section_centerline_geometry_taxonomy.py)",
        file=sys.stderr,
    )
    _mesh_main(sys.argv[1:])


if __name__ == "__main__":
    main()
