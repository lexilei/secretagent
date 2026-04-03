"""Induced ptool lifecycle: create @interface, bind to simulate, lookup."""

import re
from secretagent.core import interface as make_interface, Interface

_INDUCED_INTERFACES: dict[str, Interface] = {}


def create_ptool_interface(name: str, doc: str) -> Interface:
    """Create a real @interface ptool and bind to simulate."""
    def stub(narrative: str, focus: str) -> str: ...
    stub.__name__ = name
    stub.__qualname__ = name
    stub.__doc__ = doc
    stub.__annotations__ = {'narrative': str, 'focus': str, 'return': str}
    stub.__module__ = 'ptools_induced'

    iface = make_interface(stub)
    iface.implement_via('simulate')
    _INDUCED_INTERFACES[name] = iface
    return iface


def find_induced_ptool(display_name: str) -> Interface | None:
    """Find an induced ptool by display name (case-insensitive)."""
    for name, iface in _INDUCED_INTERFACES.items():
        if name.lower() == display_name.lower():
            return iface
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', display_name).lower()
        if name == snake:
            return iface
    return None
