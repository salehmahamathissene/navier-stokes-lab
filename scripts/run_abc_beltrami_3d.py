from __future__ import annotations

import argparse
import json

from ns.solver3d import NS3DParams, run_abc_beltrami_3d


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the exact 3D ABC/Beltrami benchmark."
    )
    parser.add_argument("--n", type=int, default=16, help="Grid size per direction")
    parser.add_argument("--nu", type=float, default=0.01, help="Kinematic viscosity")
    parser.add_argument("--dt", type=float, default=0.0025, help="Time step")
    parser.add_argument("--steps", type=int, default=40, help="Number of time steps")
    parser.add_argument("--no-dealias", action="store_false", dest="dealias")
    parser.set_defaults(dealias=True)
    args = parser.parse_args()

    params = NS3DParams(
        n=args.n,
        nu=args.nu,
        dt=args.dt,
        steps=args.steps,
        dealias=args.dealias,
    )
    result = run_abc_beltrami_3d(params)
    report = {
        "n": params.n,
        "nu": params.nu,
        "dt": params.dt,
        "steps": params.steps,
        "final_time": params.dt * params.steps,
        "dealias": params.dealias,
        "relative_l2_error": result["relative_l2_error"],
        "linf_error": result["linf_error"],
        "numerical_energy": result["energy"][-1],
        "exact_energy": result["exact_energy"],
        "energy_absolute_error": result["energy_absolute_error"],
        "energy_relative_error": result["energy_relative_error"],
        "final_max_divergence": result["max_div"][-1],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
