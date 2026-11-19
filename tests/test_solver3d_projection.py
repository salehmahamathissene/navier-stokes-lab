import contextlib
import io

import numpy as np

from ns.solver3d import (
    NS3DParams,
    _project_incompressible,
    _fftfreqs,
    run_taylor_green_3d,
    taylor_green_initial,
)


def _spectral_divergence(u, v, w, kx, ky, kz):
    uhat = np.fft.fftn(u)
    vhat = np.fft.fftn(v)
    what = np.fft.fftn(w)
    div_hat = 1j * (kx * uhat + ky * vhat + kz * what)
    return float(np.max(np.abs(np.fft.ifftn(div_hat).real)))


def _wave_numbers(n):
    k = _fftfreqs(n)
    return np.meshgrid(k, k, k, indexing="ij")


def test_random_field_projection_reduces_divergence():
    n = 9
    kx, ky, kz = _wave_numbers(n)
    rng = np.random.default_rng(12345)
    u = rng.standard_normal((n, n, n))
    v = rng.standard_normal((n, n, n))
    w = rng.standard_normal((n, n, n))

    before = _spectral_divergence(u, v, w, kx, ky, kz)
    uhat, vhat, what = (np.fft.fftn(f) for f in (u, v, w))
    uhat_p, vhat_p, what_p = _project_incompressible(
        uhat, vhat, what, kx, ky, kz
    )
    u_p = np.fft.ifftn(uhat_p).real
    v_p = np.fft.ifftn(vhat_p).real
    w_p = np.fft.ifftn(what_p).real
    after = _spectral_divergence(u_p, v_p, w_p, kx, ky, kz)

    assert before > 1e-3
    assert after < 1e-12
    assert after < before


def test_even_grid_random_field_projection_preserves_hermitian_structure():
    n = 8
    kx, ky, kz = _wave_numbers(n)
    rng = np.random.default_rng(24680)
    u = rng.standard_normal((n, n, n))
    v = rng.standard_normal((n, n, n))
    w = rng.standard_normal((n, n, n))

    uhat = np.fft.fftn(u)
    vhat = np.fft.fftn(v)
    what = np.fft.fftn(w)
    uhat_before = uhat.copy()
    vhat_before = vhat.copy()
    what_before = what.copy()

    before = _spectral_divergence(u, v, w, kx, ky, kz)
    uhat_p, vhat_p, what_p = _project_incompressible(
        uhat, vhat, what, kx, ky, kz
    )

    u_p = np.fft.ifftn(uhat_p)
    v_p = np.fft.ifftn(vhat_p)
    w_p = np.fft.ifftn(what_p)

    after = _spectral_divergence(u_p.real, v_p.real, w_p.real, kx, ky, kz)

    np.testing.assert_array_equal(uhat, uhat_before)
    np.testing.assert_array_equal(vhat, vhat_before)
    np.testing.assert_array_equal(what, what_before)
    assert uhat_p[0, 0, 0] == uhat_before[0, 0, 0]
    assert vhat_p[0, 0, 0] == vhat_before[0, 0, 0]
    assert what_p[0, 0, 0] == what_before[0, 0, 0]

    assert after < before
    assert after < 1e-12
    assert np.max(np.abs(u_p.imag)) < 1e-12
    assert np.max(np.abs(v_p.imag)) < 1e-12
    assert np.max(np.abs(w_p.imag)) < 1e-12

    for hat_p in (uhat_p, vhat_p, what_p):
        hermitian = np.empty_like(hat_p)
        for index in np.ndindex(hat_p.shape):
            conjugate_index = tuple((-i) % n for i in index)
            hermitian[index] = np.conjugate(hat_p[conjugate_index])
        np.testing.assert_allclose(hat_p, hermitian, rtol=1e-12, atol=1e-12)


def test_divergence_free_field_is_unchanged_by_projection():
    n = 8
    kx, ky, kz = _wave_numbers(n)
    u, v, w = taylor_green_initial(n)

    uhat, vhat, what = np.fft.fftn(u), np.fft.fftn(v), np.fft.fftn(w)
    uhat_p, vhat_p, what_p = _project_incompressible(
        uhat, vhat, what, kx, ky, kz
    )

    np.testing.assert_allclose(uhat_p, uhat, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(vhat_p, vhat, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(what_p, what, rtol=1e-12, atol=1e-12)


def test_projection_preserves_zero_frequency_mean():
    n = 8
    kx, ky, kz = _wave_numbers(n)
    rng = np.random.default_rng(7)
    uhat = rng.standard_normal((n, n, n)) + 1j * rng.standard_normal((n, n, n))
    vhat = rng.standard_normal((n, n, n)) + 1j * rng.standard_normal((n, n, n))
    what = rng.standard_normal((n, n, n)) + 1j * rng.standard_normal((n, n, n))

    uhat_p, vhat_p, what_p = _project_incompressible(
        uhat, vhat, what, kx, ky, kz
    )

    assert uhat_p[0, 0, 0] == uhat[0, 0, 0]
    assert vhat_p[0, 0, 0] == vhat[0, 0, 0]
    assert what_p[0, 0, 0] == what[0, 0, 0]
    assert np.all(np.isfinite(uhat_p))
    assert np.all(np.isfinite(vhat_p))
    assert np.all(np.isfinite(what_p))


def test_taylor_green_3d_initial_divergence_and_finite_output():
    n = 16
    u, v, w = taylor_green_initial(n)
    kx, ky, kz = _wave_numbers(n)

    initial_div = _spectral_divergence(u, v, w, kx, ky, kz)
    assert initial_div < 1e-12
    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(v))
    assert np.all(np.isfinite(w))

    params = NS3DParams(n=n, nu=0.01, dt=0.0005, steps=20, dealias=True)
    with contextlib.redirect_stdout(io.StringIO()):
        stats = run_taylor_green_3d(params)

    assert np.all(np.isfinite(stats["energy"]))
    assert np.all(np.isfinite(stats["max_div"]))
    assert max(stats["max_div"]) < 1e-10
