from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from ns.cavity import CavityParams, run_lid_driven_cavity


def main() -> None:
    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)

    # Good defaults for a “serious” run that still finishes on a laptop
    p = CavityParams(
        n=129,
        re=1000.0,
        dt=2.5e-4,
        t_end=5.0,
        save_every=200,
        poisson_iters=250,
        poisson_tol=0.0,
    )

    sol = run_lid_driven_cavity(p)
    u = sol["u"]
    v = sol["v"]
    omega = sol["omega"]

    speed = np.sqrt(u*u + v*v)

    # Plot vorticity
    plt.figure()
    plt.imshow(omega.T, origin="lower", extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.title(f"Lid-driven cavity vorticity (Re={p.re:g})")
    plt.xlabel("x"); plt.ylabel("y")
    (outdir / "cavity_vorticity.png").write_bytes(b"")  # ensure path exists
    plt.savefig(outdir / "cavity_vorticity.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Plot speed
    plt.figure()
    plt.imshow(speed.T, origin="lower", extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.title(f"Lid-driven cavity speed (Re={p.re:g})")
    plt.xlabel("x"); plt.ylabel("y")
    plt.savefig(outdir / "cavity_speed.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Centerline velocity profile (classic validation plot)
    n = p.n
    x = np.linspace(0, 1, n)
    mid = n // 2
    u_centerline = u[mid, :]          # along y at x=0.5
    v_centerline = v[:, mid]          # along x at y=0.5

    plt.figure()
    plt.plot(x, u_centerline)
    plt.title("Centerline u(y) at x=0.5")
    plt.xlabel("y"); plt.ylabel("u")
    plt.grid(True)
    plt.savefig(outdir / "cavity_centerline_u.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(x, v_centerline)
    plt.title("Centerline v(x) at y=0.5")
    plt.xlabel("x"); plt.ylabel("v")
    plt.grid(True)
    plt.savefig(outdir / "cavity_centerline_v.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:",
          "results/cavity_vorticity.png, results/cavity_speed.png,",
          "results/cavity_centerline_u.png, results/cavity_centerline_v.png")


if __name__ == "__main__":
    main()
