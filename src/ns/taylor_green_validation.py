from __future__ import annotations

import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .solver3d import NS3DParams, run_spectral_3d, taylor_green_initial

SCHEMA_VERSION = "1.0"
SPATIAL_TIME_ERROR_FRACTION = 0.1
DOMAIN_LENGTH = 2.0 * math.pi

DIAGNOSTICS = {
    "kinetic_energy": {
        "definition": "0.5 times the grid mean of u_x^2 + u_y^2 + u_z^2",
        "units": "velocity_squared_in_code_units",
    },
    "kinetic_energy_dissipation": {
        "definition": (
            "Instantaneous semi-discrete dissipation for the solver's raw-k^2 "
            "spectral diffusion operator: nu/N^6 times the sum over Fourier "
            "modes of k^2 times squared velocity-coefficient magnitude, where "
            "N is the grid size per direction. Even-grid Nyquist bins use "
            "their raw FFT wavenumbers here, matching diffusion, although the "
            "corresponding real-field first-derivative component is zero under "
            "the solver's convention. This rate is not finite-step "
            "backward-Euler energy loss divided by the timestep."
        ),
        "units": "velocity_squared_per_time_in_code_units",
    },
    "max_spectral_divergence": {
        "definition": "grid maximum of the absolute spectral divergence",
        "units": "velocity_per_length_in_code_units",
    },
}


@dataclass(frozen=True)
class RunResult:
    configuration: dict[str, Any]
    history: list[dict[str, int | float]]
    final_velocity: tuple[np.ndarray, np.ndarray, np.ndarray]


def validate_run_parameters(n: int, nu: float, dt: float, steps: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n < 6:
        raise ValueError("n must be an integer greater than or equal to 6.")
    if not math.isfinite(nu) or nu < 0.0:
        raise ValueError("nu must be finite and non-negative.")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive.")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer.")


def steps_for_final_time(dt: float, final_time: float) -> int:
    if not math.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("final_time must be finite and positive.")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive.")
    steps = round(final_time / dt)
    if steps < 1 or not math.isclose(
        steps * dt, final_time, rel_tol=1e-12, abs_tol=1e-14
    ):
        raise ValueError("final_time must be an integer multiple of dt.")
    return steps


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def _document(experiment: str, configuration: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "environment": _environment(),
        "diagnostics": DIAGNOSTICS,
        "configuration": configuration,
    }


def run_case(n: int, nu: float, dt: float, steps: int, dealias: bool) -> RunResult:
    validate_run_parameters(n, nu, dt, steps)
    if not isinstance(dealias, bool):
        raise ValueError("dealias must be a boolean.")

    params = NS3DParams(n=n, nu=nu, dt=dt, steps=steps, dealias=dealias)
    result = run_spectral_3d(
        params,
        taylor_green_initial(n),
        record_dissipation=True,
    )
    energy = result["energy"]
    divergence = result["max_div"]
    dissipation = result["dissipation"]
    if not (
        isinstance(energy, list)
        and isinstance(divergence, list)
        and isinstance(dissipation, list)
    ):
        raise RuntimeError("The solver returned an unexpected diagnostic history.")
    if not (
        np.isfinite(energy).all()
        and np.isfinite(divergence).all()
        and np.isfinite(dissipation).all()
    ):
        raise FloatingPointError("The solver produced a non-finite diagnostic.")

    history = [
        {
            "step": step,
            "time": step * dt,
            "kinetic_energy": float(energy[step]),
            "kinetic_energy_dissipation": float(dissipation[step]),
            "max_spectral_divergence": float(divergence[step]),
        }
        for step in range(steps + 1)
    ]
    configuration = {
        "n": n,
        "domain": [0.0, DOMAIN_LENGTH],
        "boundary_conditions": "periodic_in_all_directions",
        "nu": nu,
        "dt": dt,
        "steps": steps,
        "final_time": steps * dt,
        "dealias": dealias,
    }
    return RunResult(
        configuration=configuration,
        history=history,
        final_velocity=(result["u"], result["v"], result["w"]),
    )


def run_single_experiment(
    n: int, nu: float, dt: float, steps: int, dealias: bool = True
) -> dict[str, Any]:
    case = run_case(n, nu, dt, steps, dealias)
    document = _document("single_run", case.configuration)
    document["history"] = case.history
    return document


def _final_value(case: RunResult, key: str) -> float:
    return float(case.history[-1][key])


def _velocity_difference(coarse: RunResult, fine: RunResult) -> tuple[float, float]:
    return _velocity_component_difference(coarse.final_velocity, fine.final_velocity)


def _velocity_component_difference(
    coarse_velocity: tuple[np.ndarray, np.ndarray, np.ndarray],
    fine_velocity: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float]:
    error_squared = np.zeros_like(fine_velocity[0])
    reference_squared = np.zeros_like(fine_velocity[0])
    for coarse_component, fine_component in zip(coarse_velocity, fine_velocity):
        error_squared += (coarse_component - fine_component) ** 2
        reference_squared += fine_component**2
    absolute_rms = float(np.sqrt(np.mean(error_squared)))
    reference_rms = float(np.sqrt(np.mean(reference_squared)))
    relative_rms = absolute_rms / reference_rms if reference_rms > 0.0 else 0.0
    return absolute_rms, relative_rms


def _shared_grid_velocity_difference(
    coarse: RunResult, fine: RunResult
) -> tuple[float, float]:
    coarse_n = int(coarse.configuration["n"])
    fine_n = int(fine.configuration["n"])
    if fine_n % coarse_n != 0:
        raise ValueError(
            "Each refined grid must be an integer multiple of its coarser grid."
        )
    ratio = fine_n // coarse_n
    fine_on_coarse_grid = tuple(
        component[::ratio, ::ratio, ::ratio] for component in fine.final_velocity
    )
    return _velocity_component_difference(coarse.final_velocity, fine_on_coarse_grid)


def _observed_order(
    difference_1: float, difference_2: float, ratio: float
) -> float | None:
    if difference_1 <= 0.0 or difference_2 <= 0.0 or ratio <= 1.0:
        return None
    return math.log(difference_1 / difference_2) / math.log(ratio)


def run_temporal_refinement(
    n: int,
    nu: float,
    dt: float,
    final_time: float,
    dealias: bool = True,
) -> dict[str, Any]:
    steps_for_final_time(dt, final_time)
    cases = []
    runs: list[RunResult] = []
    for level in range(3):
        refined_dt = dt / (2**level)
        steps = steps_for_final_time(refined_dt, final_time)
        case = run_case(n, nu, refined_dt, steps, dealias)
        runs.append(case)
        cases.append({"configuration": case.configuration, "history": case.history})

    differences = [
        _velocity_difference(runs[index], runs[index + 1]) for index in range(2)
    ]
    diagnostic_differences = {}
    for quantity in ("kinetic_energy", "kinetic_energy_dissipation"):
        absolute_differences = [
            abs(
                _final_value(runs[index], quantity)
                - _final_value(runs[index + 1], quantity)
            )
            for index in range(2)
        ]
        diagnostic_differences[quantity] = {
            "difference_dt_vs_dt_over_2": absolute_differences[0],
            "difference_dt_over_2_vs_dt_over_4": absolute_differences[1],
            "observed_order": _observed_order(
                absolute_differences[0], absolute_differences[1], ratio=2.0
            ),
            "interpretation": (
                "Observed self-convergence order for this scalar diagnostic; "
                "not error against an exact solution."
            ),
        }
    comparison = {
        "quantity": "final_velocity_self_difference",
        "difference_dt_vs_dt_over_2": differences[0][0],
        "difference_dt_over_2_vs_dt_over_4": differences[1][0],
        "relative_difference_dt_vs_dt_over_2": differences[0][1],
        "relative_difference_dt_over_2_vs_dt_over_4": differences[1][1],
        "observed_order": _observed_order(
            differences[0][0], differences[1][0], ratio=2.0
        ),
        "diagnostic_differences": diagnostic_differences,
        "interpretation": (
            "Self-convergence estimate, not error against an exact solution."
        ),
    }
    configuration = {
        "n": n,
        "nu": nu,
        "base_dt": dt,
        "final_time": final_time,
        "dealias": dealias,
        "time_steps": [case.configuration for case in runs],
    }
    document = _document("temporal_refinement", configuration)
    document["runs"] = cases
    document["comparison"] = comparison
    return document


def _geometric_refinement_ratio(grids: Sequence[int]) -> float | None:
    ratios = [grids[index + 1] / grids[index] for index in range(len(grids) - 1)]
    if not ratios or any(not math.isclose(ratio, ratios[0]) for ratio in ratios[1:]):
        return None
    return ratios[0]


def _relative_difference(value: float, reference: float) -> float | None:
    return abs(value - reference) / abs(reference) if reference != 0.0 else None


def run_spatial_refinement(
    grids: Sequence[int],
    nu: float,
    dt: float,
    final_time: float,
    dealias: bool = True,
) -> dict[str, Any]:
    if len(grids) < 3:
        raise ValueError("Spatial refinement requires at least three grid sizes.")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 6 for n in grids):
        raise ValueError(
            "Each grid size must be an integer greater than or equal to 6."
        )
    if any(grids[index] >= grids[index + 1] for index in range(len(grids) - 1)):
        raise ValueError("Grid sizes must be strictly increasing.")
    if any(grids[index + 1] % grids[index] != 0 for index in range(len(grids) - 1)):
        raise ValueError(
            "Each refined grid must be an integer multiple of its coarser grid."
        )
    steps_for_final_time(dt, final_time)
    validate_run_parameters(grids[0], nu, dt, 0)
    if not isinstance(dealias, bool):
        raise ValueError("dealias must be a boolean.")

    runs_by_grid: list[dict[str, RunResult]] = []
    output_runs = []
    for n in grids:
        coarse_steps = steps_for_final_time(dt, final_time)
        fine_dt = dt / 2.0
        fine_steps = steps_for_final_time(fine_dt, final_time)
        coarse_case = run_case(n, nu, dt, coarse_steps, dealias)
        fine_case = run_case(n, nu, fine_dt, fine_steps, dealias)
        runs_by_grid.append({"dt": coarse_case, "dt_over_2": fine_case})
        output_runs.append(
            {
                "n": n,
                "coarse_timestep": {
                    "configuration": coarse_case.configuration,
                    "history": coarse_case.history,
                },
                "refined_timestep": {
                    "configuration": fine_case.configuration,
                    "history": fine_case.history,
                },
            }
        )

    spatial_pairs = []
    temporal_controls = []
    control_passes: list[bool] = []
    for index in range(len(grids) - 1):
        n_coarse, n_fine = grids[index], grids[index + 1]
        coarse = runs_by_grid[index]["dt_over_2"]
        fine = runs_by_grid[index + 1]["dt_over_2"]
        velocity_difference, velocity_relative_difference = (
            _shared_grid_velocity_difference(coarse, fine)
        )
        pair = {
            "n_coarse": n_coarse,
            "n_fine": n_fine,
            "velocity_rms_absolute_difference": velocity_difference,
            "velocity_rms_relative_difference": velocity_relative_difference,
            "kinetic_energy_absolute_difference": abs(
                _final_value(coarse, "kinetic_energy")
                - _final_value(fine, "kinetic_energy")
            ),
            "kinetic_energy_relative_difference": _relative_difference(
                _final_value(coarse, "kinetic_energy"),
                _final_value(fine, "kinetic_energy"),
            ),
            "dissipation_absolute_difference": abs(
                _final_value(coarse, "kinetic_energy_dissipation")
                - _final_value(fine, "kinetic_energy_dissipation")
            ),
            "dissipation_relative_difference": _relative_difference(
                _final_value(coarse, "kinetic_energy_dissipation"),
                _final_value(fine, "kinetic_energy_dissipation"),
            ),
        }
        spatial_pairs.append(pair)

        temporal_velocity_deltas = [
            _velocity_difference(
                runs_by_grid[level]["dt"], runs_by_grid[level]["dt_over_2"]
            )[0]
            for level in (index, index + 1)
        ]
        largest_temporal_delta = max(temporal_velocity_deltas)
        spatial_velocity_delta = pair["velocity_rms_absolute_difference"]
        if spatial_velocity_delta > 0.0:
            fraction = largest_temporal_delta / spatial_velocity_delta
            passed = fraction <= SPATIAL_TIME_ERROR_FRACTION
        else:
            fraction = 0.0 if largest_temporal_delta == 0.0 else None
            passed = fraction == 0.0
        temporal_deltas = {
            "final_velocity": {
                "largest_dt_vs_dt_over_2_rms_difference": largest_temporal_delta,
                "spatial_rms_difference": spatial_velocity_delta,
                "temporal_to_spatial_fraction": fraction,
                "passed": passed,
            }
        }
        control_passes.append(passed)
        temporal_controls.append(
            {
                "n_coarse": n_coarse,
                "n_fine": n_fine,
                "quantities": temporal_deltas,
                "passed": passed,
            }
        )

    grid_ratio = _geometric_refinement_ratio(grids)
    velocity_differences = [
        pair["velocity_rms_absolute_difference"] for pair in spatial_pairs
    ]
    observed_orders = {
        "velocity_rms": (
            [
                _observed_order(
                    velocity_differences[index],
                    velocity_differences[index + 1],
                    grid_ratio,
                )
                for index in range(len(velocity_differences) - 1)
            ]
            if grid_ratio is not None
            else []
        )
    }

    configuration = {
        "grids": list(grids),
        "nu": nu,
        "dt": dt,
        "spatial_comparison_dt": dt / 2.0,
        "final_time": final_time,
        "dealias": dealias,
        "time_error_fraction_limit": SPATIAL_TIME_ERROR_FRACTION,
        "observed_order_note": (
            "Orders are based on velocity RMS differences evaluated at "
            "shared coarse-grid nodes; "
            "they are not proof of convergence."
        ),
    }
    document = _document("spatial_refinement", configuration)
    document["runs"] = output_runs
    document["spatial_differences"] = spatial_pairs
    document["observed_orders"] = observed_orders
    document["temporal_error_control"] = {
        "checks": temporal_controls,
        "passed": all(control_passes),
        "definition": (
            "The larger final-velocity RMS change between dt and dt/2 at the "
            "two grids must be no more than 10% of their velocity RMS "
            "difference at shared grid points, comparing spatial runs at dt/2."
        ),
    }
    return document


def write_json_exclusive(path: str | Path, document: dict[str, Any]) -> Path:
    target = Path(path)
    serialized = json.dumps(document, indent=2, sort_keys=True, allow_nan=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as output:
        output.write(serialized)
        output.write("\n")
    return target


def _add_dealiasing_argument(parser: Any) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dealias", dest="dealias", action="store_true", help="Enable dealiasing"
    )
    group.add_argument(
        "--no-dealias", dest="dealias", action="store_false", help="Disable dealiasing"
    )
    parser.set_defaults(dealias=True)


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run reproducible 3D Taylor-Green validation experiments."
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    single = subparsers.add_parser("run", help="Run one Taylor-Green configuration")
    single.add_argument("--n", type=int, default=32, help="Grid points per direction")
    single.add_argument("--nu", type=float, default=0.01, help="Kinematic viscosity")
    single.add_argument("--dt", type=float, default=5e-3, help="Time step")
    single.add_argument("--steps", type=int, default=200, help="Number of time steps")
    _add_dealiasing_argument(single)
    single.add_argument(
        "--output",
        default="results/taylor_green_3d_validation.json",
        help="Output JSON path (must not already exist)",
    )

    temporal = subparsers.add_parser(
        "temporal", help="Compare dt, dt/2, and dt/4 at fixed final time"
    )
    temporal.add_argument("--n", type=int, default=32, help="Fixed grid size")
    temporal.add_argument("--nu", type=float, default=0.01, help="Kinematic viscosity")
    temporal.add_argument("--dt", type=float, default=5e-3, help="Largest time step")
    temporal.add_argument("--final-time", type=float, default=0.1)
    _add_dealiasing_argument(temporal)
    temporal.add_argument(
        "--output",
        default="results/taylor_green_3d_temporal.json",
        help="Output JSON path (must not already exist)",
    )

    spatial = subparsers.add_parser(
        "spatial",
        help="Compare grid sizes with dt/2 runs for temporal-error control",
    )
    spatial.add_argument(
        "--grids",
        type=int,
        nargs="+",
        default=[6, 12, 24],
        help="At least three increasing nested grid sizes",
    )
    spatial.add_argument("--nu", type=float, default=0.01, help="Kinematic viscosity")
    spatial.add_argument("--dt", type=float, default=5e-3, help="Coarse control step")
    spatial.add_argument("--final-time", type=float, default=1.0)
    _add_dealiasing_argument(spatial)
    spatial.add_argument(
        "--output",
        default="results/taylor_green_3d_spatial.json",
        help="Output JSON path (must not already exist)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.experiment == "run":
            document = run_single_experiment(
                args.n, args.nu, args.dt, args.steps, args.dealias
            )
        elif args.experiment == "temporal":
            document = run_temporal_refinement(
                args.n, args.nu, args.dt, args.final_time, args.dealias
            )
        else:
            document = run_spatial_refinement(
                args.grids, args.nu, args.dt, args.final_time, args.dealias
            )
        target = write_json_exclusive(args.output, document)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote {target}")
