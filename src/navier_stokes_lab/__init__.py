"""
navier_stokes_lab: public package wrapper.

Your solver code lives in `ns/`.
We re-export key modules here to provide a stable import path:
    import navier_stokes_lab as nsl
"""

from ns.solver import *  # noqa
from ns.cavity import *  # noqa

__all__ = []
