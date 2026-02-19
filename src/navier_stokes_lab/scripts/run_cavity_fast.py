from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from navier_stokes_lab.cavity import run_cavity


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fast lid-driven cavity demo (PNG outputs).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64, help="Grid size (n x n)")
    ap.add_argument("--re", type=float, default=1000.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--t-end", type=float, default=1.0, help="Final time (smaller = faster)")
    ap.add_argument("--lid-u", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--poisson-iters", type=int, default=200)
    ap.add_argument("--poisson-tol", type=float, default=0.0)
    args = ap.parse_args(argv)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Running Lid-Driven Cavity Simulation...")
    print(f"n={args.n} re={args.re} dt={args.dt} t_end={args.t_end}")
    print(f"Output dir: {out_dir}")

    out = run_cavity(
        n=args.n,
        re=args.re,
        dt=args.dt,
        t_end=args.t_end,
        lid_u=args.lid_u,
        save_every=args.save_every,
        poisson_iters=args.poisson_iters,
        poisson_tol=args.poisson_tol,
    )

    print("Solver output keys:", sorted(out.keys()))

    # Require u,v keys (if solver uses different keys, you will SEE them above)
    if "u" not in out or "v" not in out:
        raise RuntimeError(f"Missing u/v in output. Keys: {sorted(out.keys())}")

    u = out["u"]
    v = out["v"]
    speed = np.sqrt(u*u + v*v)

    dvdx = np.gradient(v, axis=1)
    dudy = np.gradient(u, axis=0)
    vort = dvdx - dudy

    speed_path = out_dir / "cavity_fast_speed.png"
    vort_path = out_dir / "cavity_fast_vorticity.png"

    plt.figure()
    plt.imshow(speed, origin="lower")
    plt.colorbar()
    plt.title("Cavity Fast: Speed")
    plt.tight_layout()
    plt.savefig(speed_path, dpi=150)
    plt.close()

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
