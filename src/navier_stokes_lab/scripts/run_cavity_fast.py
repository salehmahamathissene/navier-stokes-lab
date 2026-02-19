from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# You MUST adjust imports to match your actual ns API:
# If ns.cavity has a function you already use, import and call it.
from ns.cavity import cavity_demo  # <-- if name differs, change it


def main() -> None:
    root = Path(__file__).resolve().parents[3]  # repo root
    results = root / "results"
    results.mkdir(exist_ok=True)

    # Run demo (adjust args to your function)
    # Expected: returns speed, vorticity arrays or produces fields you can plot
    speed, vort = cavity_demo(nx=64, ny=64, steps=200, dt=0.005)

    # Speed plot
    plt.figure()
    plt.imshow(speed, origin="lower")
    plt.colorbar()
    plt.title("Cavity Fast: Speed")
    plt.tight_layout()
    plt.savefig(results / "cavity_fast_speed.png", dpi=150)
    plt.close()

    # Vorticity plot
    plt.figure()
    plt.imshow(vort, origin="lower")
    plt.colorbar()
    plt.title("Cavity Fast: Vorticity")
    plt.tight_layout()
    plt.savefig(results / "cavity_fast_vorticity.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
