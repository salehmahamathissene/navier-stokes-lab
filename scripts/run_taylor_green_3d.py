from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from ns.solver3d import NS3DParams, run_taylor_green_3d


def main() -> None:
    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)

    p = NS3DParams(n=32, nu=0.01, dt=5e-3, steps=200, dealias=True)
    stats = run_taylor_green_3d(p)

    # Energy plot
    plt.figure()
    plt.plot(stats["energy"])
    plt.title("3D Taylor–Green: kinetic energy")
    plt.xlabel("step")
    plt.ylabel("KE")
    plt.grid(True)
    plt.savefig(outdir / "tg3d_energy.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Divergence plot
    plt.figure()
    plt.plot(stats["max_div"])
    plt.title("3D Taylor–Green: max divergence")
    plt.xlabel("step")
    plt.ylabel("max|div u|")
    plt.yscale("log")
    plt.grid(True)
    plt.savefig(outdir / "tg3d_div.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved: results/tg3d_energy.png, results/tg3d_div.png")


if __name__ == "__main__":
    main()
