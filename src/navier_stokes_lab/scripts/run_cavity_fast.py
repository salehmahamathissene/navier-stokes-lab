from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Headless-safe backend for Render/Linux servers
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from navier_stokes_lab.cavity import run_lid_driven_cavity


def _compute_speed_vorticity(result: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    NO guessing about solver internals, but we must pick expected keys.
    For a lid-driven cavity solver, velocity fields are commonly u,v.
    If your solver returns different keys, this will error and show the keys.
    """
    if "u" not in result or "v" not in result:
        raise RuntimeError(
            f"Solver output keys: {sorted(result.keys())}. Expected at least 'u' and 'v'."
        )

    u = result["u"]
    v = result["v"]

    speed = np.sqrt(u * u + v * v)

    # vorticity = dv/dx - du/dy
    dvdx = np.gradient(v, axis=1)
    dudy = np.gradient(u, axis=0)
    vort = dvdx - dudy

    return speed, vort


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fast lid-driven cavity demo (PNG outputs).")
    ap.add_argument("--out", required=True, help="Output directory (e.g. /tmp/out)")
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.005)
    args = ap.parse_args(argv)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Running Lid-Driven Cavity Simulation...")
    print(f"Grid: {args.nx} x {args.ny}, Steps: {args.steps}, dt={args.dt}")
    print(f"Output dir: {out_dir}")

    result = run_lid_driven_cavity(nx=args.nx, ny=args.ny, steps=args.steps, dt=args.dt)
    speed, vort = _compute_speed_vorticity(result)

    speed_path = out_dir / "cavity_fast_speed.png"
    vort_path = out_dir / "cavity_fast_vorticity.png"

    # Speed plot
    plt.figure()
    plt.imshow(speed, origin="lower")
    plt.colorbar()
    plt.title("Cavity Fast: Speed")
    plt.tight_layout()
    plt.savefig(speed_path, dpi=150)
    plt.close()

    # Vorticity plot
    plt.figure()
    plt.imshow(vort, origin="lower")
    plt.colorbar()
    plt.title("Cavity Fast: Vorticity")
    plt.tight_layout()
    plt.savefig(vort_path, dpi=150)
    plt.close()

    print("✅ Wrote:", speed_path)
    print("✅ Wrote:", vort_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
