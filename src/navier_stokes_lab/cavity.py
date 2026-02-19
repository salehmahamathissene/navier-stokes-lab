from __future__ import annotations

import inspect
from typing import Any, Dict

import numpy as np


def _cavity_params_class():
    """
    Import CavityParams from the *real* solver package: ns.cavity
    """
    from ns.cavity import CavityParams  # type: ignore
    return CavityParams


def _solver_run_fn():
    """
    Import solver function from ns.cavity with signature:
        run_lid_driven_cavity(p: CavityParams) -> dict[str, np.ndarray]
    """
    from ns.cavity import run_lid_driven_cavity  # type: ignore
    return run_lid_driven_cavity


def _build_params(nx: int, ny: int, steps: int, dt: float) -> Any:
    """
    Build a CavityParams object WITHOUT guessing unknown required fields.

    Strategy:
    - Introspect CavityParams signature.
    - Fill only recognized names:
        n, nx, ny, steps/nt/nsteps, dt
    - If CavityParams has other required (no default) fields we cannot map,
      raise with the exact signature so you can set them explicitly.
    """
    CavityParams = _cavity_params_class()
    sig = inspect.signature(CavityParams)

    kwargs: dict[str, Any] = {}

    def is_required(p: inspect.Parameter) -> bool:
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            return False
        return p.default is inspect._empty

    for name, p in sig.parameters.items():
        # Map common names only (safe)
        if name == "n":
            kwargs[name] = int(nx)  # choose nx for single-size grid
            continue
        if name == "nx":
            kwargs[name] = int(nx)
            continue
        if name == "ny":
            kwargs[name] = int(ny)
            continue
        if name in ("steps", "nsteps", "nt"):
            kwargs[name] = int(steps)
            continue
        if name == "dt":
            kwargs[name] = float(dt)
            continue

        # If required and unknown -> STOP (no guessing)
        if is_required(p):
            raise RuntimeError(
                "CavityParams has a required field we cannot map safely.\n"
                f"Missing required parameter: '{name}'\n"
                f"CavityParams signature: {sig}\n"
                "Fix: either rename your solver param to nx/ny/steps/dt/n or "
                "update this wrapper to map that param explicitly."
            )

    return CavityParams(**kwargs)


def run_lid_driven_cavity(
    nx: int = 64,
    ny: int = 64,
    steps: int = 200,
    dt: float = 0.005,
) -> Dict[str, np.ndarray]:
    """
    Wrapper that returns EXACT solver output:
        dict[str, np.ndarray]

    It calls ns.cavity.run_lid_driven_cavity(p) with p = CavityParams(...)
    """
    run_fn = _solver_run_fn()
    p = _build_params(nx=nx, ny=ny, steps=steps, dt=dt)
    out = run_fn(p)

    if not isinstance(out, dict):
        raise RuntimeError(f"Solver returned {type(out)}, expected dict[str, np.ndarray].")

    # Validate arrays
    for k, v in out.items():
        if not isinstance(v, np.ndarray):
            raise RuntimeError(f"Solver output key '{k}' is {type(v)}, expected numpy.ndarray.")
    return out
