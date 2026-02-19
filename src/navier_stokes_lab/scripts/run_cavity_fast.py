from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ✅ Correct import: YOUR package
from navier_stokes_lab.cavity import run_lid_driven_cavity


# ------------------------------------------------------------
# FAST + RENDER SAFE CFD DEMO
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast Lid-Driven Cavity Demo (Navier–Stokes Lab)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/tmp/ns_outputs",
        help="Output directory for PNG files (Render-safe)",
    )
    parser.add_argument("--nx", type=int, default=64, help="Grid resolution X")
    parser.add_argument("--ny", type=int, default=64, help="Grid resolution Y")
    parser.add_argument("--steps", type=int, default=200, help="Simulation steps")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step")

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Running Lid-Driven Cavity Simulation...")
    print(f"Grid: {args.nx} x {args.ny}, Steps: {args.steps}, dt={args.dt}")
    print("Output dir:", out_dir)

    # ------------------------------------------------------------
    # ✅ Run solver
    # ------------------------------------------------------------
    result = run_lid_driven_cavity(
        nx=args.nx,
        ny=args.ny,
        steps=args.steps,
        dt=args.dt,
    )

    # ------------------------------------------------------------
    # ✅ Extract fields safely
    # ------------------------------------------------------------
    u = result.get("u")
    v = result.get("v")
    omega = result.get("vorticity")

    if u is None or v is None:
        raise RuntimeError("Solver did not return velocity fields (u,v).")

    speed = np.sqrt(u**2 + v**2)

    if omega is None:
        # Compute vorticity if solver didn't return it
        omega = np.gradient(v, axis=1) - np.gradient(u, axis=0)

    # ------------------------------------------------------------
    # ✅ Save SPEED plot
    # ------------------------------------------------------------
    speed_path = out_dir / "cavity_fast_speed.png"

    plt.figure()
    plt.imshow(speed, origin="lower")
    plt.colorbar()
    plt.title("Cavity Flow Speed")
    plt.tight_layout()
    plt.savefig(speed_path, dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # ✅ Save VORTICITY plot
    # ------------------------------------------------------------
    vort_path = out_dir / "cavity_fast_vorticity.png"

    plt.figure()
    plt.imshow(omega, origin="lower")
    plt.colorbar()
    plt.title("Cavity Flow Vorticity")
    plt.tight_layout()
    plt.savefig(vort_path, dpi=150)
    plt.close()

    print("✅ Done!")
    print("Saved outputs:")
    print(" -", speed_path)
    print(" -", vort_path)


if __name__ == "__main__":
    main()
