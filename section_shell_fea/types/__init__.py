"""Type/DTO barrel for `blade_structure_utilities.fem_utilities.section_shell_fea`.

Re-exports public type-like symbols from the worker root `__all__` when available.
"""

from importlib import import_module as _import_module

_root = _import_module('blade_structure_utilities.fem_utilities.section_shell_fea')
_candidates = list(getattr(_root, '__all__', ()))
_suffixes = ('Type', 'Types', 'Config', 'Result', 'Results', 'Row', 'Rows', 'Bin', 'Set', 'Spec')
__all__ = [name for name in _candidates if name.endswith(_suffixes)]
globals().update({name: getattr(_root, name) for name in __all__ if hasattr(_root, name)})
