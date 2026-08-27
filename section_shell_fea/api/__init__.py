"""Public API barrel for `blade_structure_utilities.fem_utilities.section_shell_fea`.

Re-exports symbols listed in the worker package `__all__` so consumers can
depend on a consistent `api` import seam.
"""

from importlib import import_module as _import_module

_root = _import_module('blade_structure_utilities.fem_utilities.section_shell_fea')
__all__ = list(getattr(_root, '__all__', ()))
globals().update({name: getattr(_root, name) for name in __all__ if hasattr(_root, name)})
