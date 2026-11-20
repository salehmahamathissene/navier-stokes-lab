import json
import math

import numpy as np
import pytest

from ns.solver3d import NS3DParams, run_spectral_3d, taylor_green_initial
from ns.taylor_green_validation import (
    run_single_experiment,
    run_spatial_refinement,
    run_temporal_refinement,
    run_case,
    steps_for_final_time,
    write_json_exclusive,
)


def test_energy_dissipation_and_divergence_diagnostics():
    case = run_case(n=8, nu=0.2, dt=0.001, steps=0, dealias=True)
    sample = case.history[0]

    assert sample["kinetic_energy"] == pytest.approx(0.125, abs=1e-14)
    assert sample["kinetic_energy_dissipation"] == pytest.approx(0.75 * 0.2, abs=1e-13)
    assert sample["max_spectral_divergence"] < 1e-12


@pytest.mark.parametrize(
    ("axis", "component"),
    [(axis, component) for axis in range(3) for component in range(3)],
)
def test_pure_nyquist_dissipation_matches_diffusion_operator_quadratic_form(
    axis, component
):
    n = 8
    nu = 0.1
    mode_shape = tuple(n if coordinate == axis else 1 for coordinate in range(3))
    mode = np.broadcast_to(
        ((-1.0) ** np.arange(n)).reshape(mode_shape), (n, n, n)
    ).copy()
    velocity = [np.zeros_like(mode) for _ in range(3)]
    velocity[component] = mode
    velocity_hat = [np.fft.fftn(field) for field in velocity]
    k = np.fft.fftfreq(n) * n
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    k_components = (kx, ky, kz)
    k2 = kx**2 + ky**2 + kz**2

    result = run_spectral_3d(
        NS3DParams(n=n, nu=nu, dt=0.001, steps=0),
        tuple(velocity),
        record_dissipation=True,
    )
    diffusion_quadratic_form = (
        nu
        * np.sum(
            k2 * sum(np.abs(component_hat) ** 2 for component_hat in velocity_hat)
        )
        / velocity_hat[0].size**2
    )
    nyquist_derivative = np.fft.ifftn(
        1j * k_components[axis] * velocity_hat[component]
    ).real

    assert np.max(np.abs(nyquist_derivative)) == 0.0
    assert result["dissipation"][0] == pytest.approx(diffusion_quadratic_form)
    assert result["dissipation"][0] == pytest.approx(16.0 * nu)


@pytest.mark.parametrize("n", [7, 8])
def test_dissipation_and_energy_use_parseval_normalization_for_odd_and_even_grids(n):
    nu = 0.13
    rng = np.random.default_rng(20261120 + n)
    velocity = tuple(rng.standard_normal((n, n, n)) for _ in range(3))
    velocity_hat = tuple(np.fft.fftn(component) for component in velocity)
    k = np.fft.fftfreq(n) * n
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2
    result = run_spectral_3d(
        NS3DParams(n=n, nu=nu, dt=0.001, steps=0),
        velocity,
        record_dissipation=True,
    )

    physical_mean_square = sum(np.mean(component**2) for component in velocity)
    spectral_mean_square = sum(
        np.sum(np.abs(component_hat) ** 2) / component_hat.size**2
        for component_hat in velocity_hat
    )
    operator_inner_product = -nu * sum(
        np.mean(
            component
            * np.fft.ifftn(-k2 * component_hat).real
        )
        for component, component_hat in zip(velocity, velocity_hat)
    )

    assert result["energy"][0] == pytest.approx(0.5 * physical_mean_square)
    assert spectral_mean_square == pytest.approx(physical_mean_square)
    assert result["dissipation"][0] == pytest.approx(operator_inner_product)


def test_non_nyquist_mode_has_analytic_diffusion_dissipation():
    n = 8
    nu = 0.2
    x = 2.0 * np.pi * np.arange(n) / n
    u = np.broadcast_to(np.sin(2.0 * x)[:, None, None], (n, n, n)).copy()
    zero = np.zeros_like(u)

    result = run_spectral_3d(
        NS3DParams(n=n, nu=nu, dt=0.001, steps=0),
        (u, zero, zero),
        record_dissipation=True,
    )

    assert result["dissipation"][0] == pytest.approx(nu * 2.0)


def test_zero_and_uniform_velocity_have_zero_diffusion_dissipation():
    n = 8
    zero = np.zeros((n, n, n))
    uniform = np.full_like(zero, 3.0)
    params = NS3DParams(n=n, nu=0.2, dt=0.001, steps=0)

    zero_result = run_spectral_3d(
        params, (zero, zero, zero), record_dissipation=True
    )
    uniform_result = run_spectral_3d(
        params, (uniform, -uniform, uniform), record_dissipation=True
    )

    assert zero_result["dissipation"] == [0.0]
    assert uniform_result["dissipation"] == [0.0]


def test_mixed_ordinary_and_nyquist_modes_add_quadratically():
    n = 8
    nu = 0.1
    x = 2.0 * np.pi * np.arange(n) / n
    u = np.broadcast_to(
        (np.sin(x) + 0.25 * (-1.0) ** np.arange(n))[:, None, None],
        (n, n, n),
    ).copy()
    zero = np.zeros_like(u)

    results = [
        run_spectral_3d(
            NS3DParams(n=n, nu=nu, dt=0.001, steps=0, dealias=dealias),
            (u, zero, zero),
            record_dissipation=True,
        )
        for dealias in (True, False)
    ]

    for result in results:
        assert result["dissipation"][0] == pytest.approx(1.5 * nu)
    assert results[0]["dissipation"] == results[1]["dissipation"]


def test_nyquist_diffusion_is_consistent_with_projection_and_dealiasing():
    n = 8
    nu = 0.2
    dt = 0.01
    mode = np.broadcast_to(
        ((-1.0) ** np.arange(n))[:, None, None], (n, n, n)
    ).copy()
    zero = np.zeros_like(mode)
    results = [
        run_spectral_3d(
            NS3DParams(n=n, nu=nu, dt=dt, steps=1, dealias=dealias),
            (mode, zero, zero),
            record_dissipation=True,
        )
        for dealias in (True, False)
    ]

    damping = 1.0 / (1.0 + dt * nu * (n // 2) ** 2)
    for result in results:
        np.testing.assert_allclose(result["u"], damping * mode, atol=1e-14)
        np.testing.assert_array_equal(result["v"], zero)
        np.testing.assert_array_equal(result["w"], zero)
        assert result["energy"][1] == pytest.approx(0.5 * damping**2)
        assert result["dissipation"] == pytest.approx(
            [nu * (n // 2) ** 2, nu * (n // 2) ** 2 * damping**2]
        )

    for key in ("energy", "max_div", "u", "v", "w", "dissipation"):
        np.testing.assert_array_equal(results[0][key], results[1][key])


def test_dissipation_recording_is_opt_in_and_preserves_solver_results():
    params = NS3DParams(n=8, nu=0.01, dt=0.001, steps=2)
    initial = taylor_green_initial(params.n)
    legacy = run_spectral_3d(params, initial)
    measured = run_spectral_3d(params, initial, record_dissipation=True)

    assert set(legacy) == {"energy", "max_div", "u", "v", "w"}
    assert "dissipation" in measured
    for key in ("energy", "max_div", "u", "v", "w"):
        np.testing.assert_array_equal(legacy[key], measured[key])


def test_single_run_schema_and_json_are_reproducible(tmp_path):
    first = run_single_experiment(8, 0.01, 0.001, 2, dealias=True)
    second = run_single_experiment(8, 0.01, 0.001, 2, dealias=True)

    assert first["schema_version"] == "1.0"
    assert first["experiment"] == "single_run"
    assert first["configuration"]["final_time"] == pytest.approx(0.002)
    assert len(first["history"]) == 3
    assert set(first["history"][0]) == {
        "step",
        "time",
        "kinetic_energy",
        "kinetic_energy_dissipation",
        "max_spectral_divergence",
    }

    first_path = write_json_exclusive(tmp_path / "first.json", first)
    second_path = write_json_exclusive(tmp_path / "second.json", second)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert json.loads(first_path.read_text()) == first


def test_temporal_refinement_keeps_final_time_fixed():
    final_time = 0.01
    result = run_temporal_refinement(
        n=8, nu=0.01, dt=0.005, final_time=final_time, dealias=True
    )

    configurations = [run["configuration"] for run in result["runs"]]
    assert [configuration["dt"] for configuration in configurations] == [
        0.005,
        0.0025,
        0.00125,
    ]
    assert all(
        math.isclose(configuration["steps"] * configuration["dt"], final_time)
        for configuration in configurations
    )
    assert result["comparison"]["observed_order"] is not None


def test_spatial_refinement_holds_time_and_reports_time_control():
    final_time = 0.01
    result = run_spatial_refinement(
        grids=[8, 16, 32],
        nu=0.01,
        dt=0.01,
        final_time=final_time,
        dealias=True,
    )

    assert [run["n"] for run in result["runs"]] == [8, 16, 32]
    for run in result["runs"]:
        for key in ("coarse_timestep", "refined_timestep"):
            configuration = run[key]["configuration"]
            assert math.isclose(
                configuration["steps"] * configuration["dt"], final_time
            )
    assert len(result["spatial_differences"]) == 2
    assert len(result["temporal_error_control"]["checks"]) == 2
    assert set(result["observed_orders"]) == {"velocity_rms"}


def test_output_refuses_to_overwrite_existing_file(tmp_path):
    target = tmp_path / "existing.json"
    target.write_text("keep this content")

    with pytest.raises(FileExistsError):
        write_json_exclusive(target, {"new": "data"})

    assert target.read_text() == "keep this content"


@pytest.mark.parametrize(
    ("n", "nu", "dt", "steps"),
    [
        (5, 0.01, 0.001, 1),
        (8, -0.01, 0.001, 1),
        (8, 0.01, 0.0, 1),
        (8, 0.01, 0.001, -1),
        (8, float("nan"), 0.001, 1),
    ],
)
def test_invalid_run_parameters_are_rejected(n, nu, dt, steps):
    with pytest.raises(ValueError):
        run_case(n, nu, dt, steps, dealias=True)


def test_invalid_refinement_configurations_are_rejected():
    with pytest.raises(ValueError, match="integer multiple"):
        steps_for_final_time(0.003, 0.01)
    with pytest.raises(ValueError, match="at least three"):
        run_spatial_refinement([8, 16], 0.01, 0.005, 0.01)
    with pytest.raises(ValueError, match="strictly increasing"):
        run_spatial_refinement([8, 16, 16], 0.01, 0.005, 0.01)
    with pytest.raises(ValueError, match="integer multiple"):
        run_spatial_refinement([8, 12, 24], 0.01, 0.005, 0.01)
