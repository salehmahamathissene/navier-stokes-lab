from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ns.solver import step_projection, periodic_div


def main():
    # Domain and grid
    N = 128
    L = 2.0 * np.pi
    dx = L / N

    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # Taylor–Green vortex initial condition (2D incompressible, periodic)
    u = np.sin(X) * np.cos(Y)
    v = -np.cos(X) * np.sin(Y)

    nu = 0.01
    dt = 0.0025
    steps = 800

    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)

    energies = []
    divs = []

    for n in range(steps):
        u, v, p = step_projection(u, v, nu=nu, dt=dt, dx=dx, L=L)

        if n % 10 == 0:
            div = periodic_div(u, v, dx)
            ke = 0.5 * np.mean(u*u + v*v)
            energies.append(ke)
            divs.append(np.max(np.abs(div)))
            print(f"step={n:04d}  KE={ke:.6f}  max|div|={divs[-1]:.3e}")

    # Plot energy decay
    plt.figure()
    plt.plot(np.arange(len(energies)) * 10 * dt, energies)
    plt.xlabel("time")
    plt.ylabel("mean kinetic energy")
    plt.title("Taylor–Green vortex: energy decay")
    plt.savefig(out / "energy.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot divergence
    plt.figure()
    plt.semilogy(np.arange(len(divs)) * 10 * dt, divs)
    plt.xlabel("time")
    plt.ylabel("max |div u|")
    plt.title("Divergence after projection")
    plt.savefig(out / "divergence.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save a snapshot of vorticity (optional)
    # ω = ∂v/∂x - ∂u/∂y (central diff)
    dvdx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * dx)
    dudy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dx)
    w = dvdx - dudy

    plt.figure()
    plt.imshow(w, origin="lower", extent=[0, L, 0, L], aspect="equal")
    plt.colorbar(label="vorticity")
    plt.title("Vorticity snapshot")
    plt.savefig(out / "vorticity.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved: results/energy.png, results/divergence.png, results/vorticity.png")


if __name__ == "__main__":
    main()
