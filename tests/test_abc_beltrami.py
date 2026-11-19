import contextlib
import io

import numpy as np

from ns.solver3d import (
    NS3DParams,
    _dealias_mask,
    _fftfreqs,
    abc_beltrami_energy,
    abc_beltrami_initial,
    run_abc_beltrami_3d,
    run_spectral_3d,
    run_taylor_green_3d,
    taylor_green_initial,
)


def _spectral_divergence(u, v, w):
    n = u.shape[0]
    k = _fftfreqs(n)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    div_hat = 1j * (kx * np.fft.fftn(u) + ky * np.fft.fftn(v) + kz * np.fft.fftn(w))
    return np.fft.ifftn(div_hat).real


def _run_silently(params):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_abc_beltrami_3d(params)


def test_abc_initial_condition_values_shape_and_divergence():
    fields = abc_beltrami_initial(8)
    assert all(field.shape == (8, 8, 8) for field in fields)
    np.testing.assert_allclose([field[0, 0, 0] for field in fields], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(
        [field[1, 2, 3] for field in fields],
        [np.sqrt(2.0) / 2.0, 0.0, 1.0 + np.sqrt(2.0) / 2.0],
        atol=1e-15,
    )
    assert np.max(np.abs(_spectral_divergence(*fields))) < 1e-12


def test_dealiasing_retains_abc_nonlinear_interaction_modes():
    n = 8
    mask = _dealias_mask(n)
    k = _fftfreqs(n)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    abc_interaction_modes = (np.abs(kx) <= 2) & (np.abs(ky) <= 2) & (np.abs(kz) <= 2)
    assert np.all(mask[abc_interaction_modes] == 1.0)


def test_short_abc_run_matches_exact_velocity_and_energy():
    params = NS3DParams(n=8, nu=0.1, dt=0.001, steps=10, dealias=True)
    result = _run_silently(params)
    final_time = params.steps * params.dt

    assert all(np.all(np.isfinite(result[key])) for key in ("u", "v", "w"))
    assert np.all(np.isfinite(result["energy"]))
    assert np.all(np.isfinite(result["max_div"]))
    assert result["relative_l2_error"] < 1e-6
    assert result["linf_error"] < 1e-6
    assert result["energy_absolute_error"] < 1e-6
    assert result["exact_energy"] == abc_beltrami_energy(params.nu, final_time)
    assert max(result["max_div"]) < 1e-12


def test_abc_temporal_refinement_is_first_order():
    errors = []
    for dt in (0.01, 0.005, 0.0025):
        params = NS3DParams(n=8, nu=0.1, dt=dt, steps=round(0.1 / dt), dealias=True)
        errors.append(_run_silently(params)["relative_l2_error"])

    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert errors[0] > errors[1] > errors[2]
    assert all(1.8 < ratio < 2.2 for ratio in ratios)


def test_abc_spatial_refinement_is_grid_independent_for_resolved_modes():
    errors = []
    for n in (8, 12, 16):
        params = NS3DParams(n=n, nu=0.1, dt=0.0005, steps=20, dealias=True)
        errors.append(_run_silently(params)["relative_l2_error"])

    # This low-mode exact field is resolved at every grid. This checks that
    # spatial discretization does not dominate; it is not a spectral-order claim.
    assert max(errors) < 1e-5
    assert max(errors) - min(errors) < 1e-10


def test_taylor_green_default_runner_behavior_is_preserved():
    params = NS3DParams(n=8, nu=0.01, dt=0.001, steps=2, dealias=True)
    with contextlib.redirect_stdout(io.StringIO()):
        default_result = run_taylor_green_3d(params)
        explicit_result = run_spectral_3d(params, taylor_green_initial(params.n))

    assert set(default_result) == {"energy", "max_div"}
    assert len(default_result["energy"]) == params.steps + 1
    assert len(default_result["max_div"]) == params.steps + 1
    for key in ("energy", "max_div"):
        np.testing.assert_allclose(default_result[key], explicit_result[key])
    assert all(key in explicit_result for key in ("u", "v", "w"))
