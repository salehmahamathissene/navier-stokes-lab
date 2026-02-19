from __future__ import annotations

from typing import Dict
import numpy as np

from ns.cavity import CavityParams, run_lid_driven_cavity  # exact imports


def run_cavity(
    n: int = 64,
    re: float = 1000.0,
    dt: float = 0.005,
    t_end: float = 1.0,
    lid_u: float = 1.0,
    save_every: int = 200,
    poisson_iters: int = 200,
    poisson_tol: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Thin wrapper around ns.cavity.run_lid_driven_cavity(p: CavityParams)

    Uses EXACT solver signature (no guessing).
    Returns dict[str, np.ndarray] as returned by solver.
    """
    p = CavityParams(
        n=int(n),
        re=float(re),
        dt=float(dt),
        t_end=float(t_end),
        lid_u=float(lid_u),
        save_every=int(save_every),
        poisson_iters=int(poisson_iters),
        poisson_tol=float(poisson_tol),
    )
    out = run_lid_driven_cavity(p)

    if not isinstance(out, dict):
        raise RuntimeError(f"Solver returned {type(out)}; expected dict[str, np.ndarray].")

    for k, v in out.items():
        if not isinstance(v, np.ndarray):
            raise RuntimeError(f"Solver output '{k}' is {type(v)}; expected numpy.ndarray.")

    return out
