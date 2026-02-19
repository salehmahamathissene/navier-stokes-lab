from __future__ import annotations

import importlib
import inspect
from dataclasses import is_dataclass, asdict
from typing import Any

import numpy as np


def _try_import(modname: str):
    try:
        return importlib.import_module(modname)
    except ModuleNotFoundError:
        return None


def _call_with_supported_kwargs(fn, nx: int, ny: int, **kwargs):
    """
    Call fn with only supported kwargs, AND auto-fill required parameters like p/u/v if needed.
    This is signature-driven (no hardcoding your solver).
    """
    sig = inspect.signature(fn)

    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}

    # Auto-fill required params that have no defaults
    for name, p in sig.parameters.items():
        if name in supported:
            continue
        if p.default is not inspect._empty:
            continue  # has default, not required
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        # If required array-like state is missing, create it using nx/ny
        if name in ("p", "pressure"):
            supported[name] = np.zeros((ny, nx), dtype=float)
            continue
        if name in ("u", "U"):
            supported[name] = np.zeros((ny, nx), dtype=float)
            continue
        if name in ("v", "V"):
            supported[name] = np.zeros((ny, nx), dtype=float)
            continue
        if name in ("b", "rhs"):
            supported[name] = np.zeros((ny, nx), dtype=float)
            continue

        # If we get here, it's a required argument we cannot safely invent
        raise RuntimeError(
            f"Backend function '{fn.__module__}.{fn.__name__}' requires parameter '{name}' "
            f"with no default. I cannot auto-create it safely."
        )

    return fn(**supported)


def _vorticity_from_uv(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # ω = dv/dx - du/dy (simple central differences)
    # assumes u,v are 2D arrays (ny, nx)
    dvdx = np.zeros_like(v, dtype=float)
    dudy = np.zeros_like(u, dtype=float)

    dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) * 0.5
    dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) * 0.5

    return dvdx - dudy


def _speed_from_uv(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u * u + v * v)


def _normalize_result(res: Any) -> dict[str, np.ndarray]:
    """
    Accepts many possible return styles and normalizes to:
      {"speed": <2D>, "vorticity": <2D>}
    """
    # dataclass -> dict
    if is_dataclass(res):
        res = asdict(res)

    # dict-like
    if isinstance(res, dict):
        # common key names
        u = res.get("u") or res.get("U")
        v = res.get("v") or res.get("V")
        speed = res.get("speed")
        vort = res.get("vorticity") or res.get("vort") or res.get("omega")

        if speed is None and u is not None and v is not None:
            speed = _speed_from_uv(np.asarray(u), np.asarray(v))

        if vort is None and u is not None and v is not None:
            vort = _vorticity_from_uv(np.asarray(u), np.asarray(v))

        if speed is None or vort is None:
            raise RuntimeError(
                "Backend returned a dict, but could not derive both speed and vorticity. "
                "Expected keys like (u,v) or speed/vorticity."
            )

        return {"speed": np.asarray(speed), "vorticity": np.asarray(vort)}

    # tuple/list
    if isinstance(res, (tuple, list)):
        if len(res) == 2:
            speed, vort = res
            return {"speed": np.asarray(speed), "vorticity": np.asarray(vort)}
        if len(res) >= 3:
            # interpret first two as u,v if they look like arrays
            u, v = res[0], res[1]
            u = np.asarray(u)
            v = np.asarray(v)
            speed = _speed_from_uv(u, v)
            vort = _vorticity_from_uv(u, v)
            return {"speed": speed, "vorticity": vort}

    raise RuntimeError(
        "Unsupported backend return type. Return dict with u/v or speed/vorticity, "
        "or return (speed, vorticity) or (u, v, ...)."
    )


def _find_backend():
    """
    We do NOT guess. We try known common module paths that might exist in your repo.
    Add more here ONLY if they exist in your codebase.
    """
    candidates = [
        "ns.cavity",
        "ns.solvers.cavity",
        "ns.demos.cavity",
        "navier_stokes_lab.solvers.cavity",
    ]
    for mod in candidates:
        m = _try_import(mod)
        if m is not None:
            return m
    return None


def run_lid_driven_cavity(
    nx: int = 64,
    ny: int = 64,
    steps: int = 200,
    dt: float = 0.005,
    **kwargs,
) -> dict[str, np.ndarray]:
    """
    Stable entrypoint used by the CLI.
    Finds an existing backend and calls a real function ONLY if present.

    Returns: {"speed": 2D array, "vorticity": 2D array}
    """
    backend = _find_backend()
    if backend is None:
        raise RuntimeError(
            "Could not import a cavity backend. "
            "Expected a module like ns.cavity (or update _find_backend() to your real module)."
        )

    fn_names = [
        "run_lid_driven_cavity",
        "lid_driven_cavity",
        "cavity_demo",
        "run_cavity",
    ]

    fn = None
    for name in fn_names:
        if hasattr(backend, name) and callable(getattr(backend, name)):
            fn = getattr(backend, name)
            break

    if fn is None:
        raise RuntimeError(
            f"Backend module '{backend.__name__}' has no callable cavity function. "
            f"Looked for: {fn_names}"
        )

    res = _call_with_supported_kwargs(
        fn,
        nx=nx,
        ny=ny,
        steps=steps,
        dt=dt,
        **kwargs,
    )

    return _normalize_result(res)
