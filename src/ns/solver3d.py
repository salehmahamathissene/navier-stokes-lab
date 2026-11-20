from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class NS3DParams:
    n: int = 32
    nu: float = 0.01
    dt: float = 5e-3
    steps: int = 200
    dealias: bool = True


def _fftfreqs(n: int) -> np.ndarray:
    # frequencies on [0, 2π)^3 grid
    return np.fft.fftfreq(n) * n  # gives integers


def _dealias_mask(n: int) -> np.ndarray:
    # 2/3 rule mask in 1D then outer product to 3D
    k = np.abs(_fftfreqs(n))
    kmax = (n // 3)
    m1 = (k <= kmax).astype(float)
    # 3D mask
    return m1[:, None, None] * m1[None, :, None] * m1[None, None, :]


def _spectral_kinetic_energy_dissipation(
    nu: float,
    uhat: np.ndarray,
    vhat: np.ndarray,
    what: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
) -> float:
    """Return the quadratic form of the solver's raw-k-squared diffusion operator.

    The raw FFT wavenumber is used at even-grid Nyquist bins, matching the
    implicit diffusion update even though the corresponding real-field
    first-derivative component uses a zero-derivative convention.
    """
    k2 = kx * kx + ky * ky + kz * kz
    spectral_speed_squared = np.abs(uhat) ** 2 + np.abs(vhat) ** 2 + np.abs(what) ** 2
    return float(nu * np.sum(k2 * spectral_speed_squared) / uhat.size**2)


def taylor_green_initial(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(u)
    return u, v, w


def abc_beltrami_initial(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the ABC/Beltrami initial velocity on an n-cubed periodic grid.

    Args:
        n: Number of grid points in each direction.

    Returns:
        The ``(u, v, w)`` velocity component arrays.
    """
    x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = np.sin(Z) + np.cos(Y)
    v = np.sin(X) + np.cos(Z)
    w = np.sin(Y) + np.cos(X)
    return u, v, w


def abc_beltrami_exact(
    n: int, nu: float, t: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the exact ABC velocity field.

    Args:
        n: Number of grid points in each direction.
        nu: Kinematic viscosity.
        t: Time at which to evaluate the reference.

    Returns:
        The exact ``(u, v, w)`` component arrays.
    """
    decay = np.exp(-nu * t)
    return tuple(decay * component for component in abc_beltrami_initial(n))


def abc_beltrami_energy(nu: float, t: float) -> float:
    """Return exact domain-mean energy.

    Args:
        nu: Kinematic viscosity.
        t: Evaluation time.

    Returns:
        Exact mean kinetic energy.
    """
    return 1.5 * float(np.exp(-2.0 * nu * t))


def _grad_hat(fhat: np.ndarray, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return 1j*kx*fhat, 1j*ky*fhat, 1j*kz*fhat


def _project_incompressible(uhat: np.ndarray, vhat: np.ndarray, what: np.ndarray, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # For even grids, the real-data Nyquist bin is self-conjugate.  Using
    # -N/2 directly as a signed derivative mode breaks Hermitian symmetry.
    # Its derivative convention is zero for real fields.
    nx, ny, nz = kx.shape
    projection_kx = kx.copy()
    projection_ky = ky.copy()
    projection_kz = kz.copy()
    if nx % 2 == 0:
        projection_kx[np.abs(projection_kx) == nx//2] = 0.0
    if ny % 2 == 0:
        projection_ky[np.abs(projection_ky) == ny//2] = 0.0
    if nz % 2 == 0:
        projection_kz[np.abs(projection_kz) == nz//2] = 0.0

    k2 = projection_kx*projection_kx + projection_ky*projection_ky + projection_kz*projection_kz
    div_hat = 1j*(projection_kx*uhat + projection_ky*vhat + projection_kz*what)
    uhat_p = uhat.copy()
    vhat_p = vhat.copy()
    what_p = what.copy()

    nonzero = k2 != 0.0
    uhat_p[nonzero] = uhat[nonzero] + 1j*projection_kx[nonzero]*div_hat[nonzero]/k2[nonzero]
    vhat_p[nonzero] = vhat[nonzero] + 1j*projection_ky[nonzero]*div_hat[nonzero]/k2[nonzero]
    what_p[nonzero] = what[nonzero] + 1j*projection_kz[nonzero]*div_hat[nonzero]/k2[nonzero]

    uhat_p[0, 0, 0] = uhat[0, 0, 0]
    vhat_p[0, 0, 0] = vhat[0, 0, 0]
    what_p[0, 0, 0] = what[0, 0, 0]
    return uhat_p, vhat_p, what_p


def run_spectral_3d(
    p: NS3DParams,
    initial_velocity: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    record_dissipation: bool = False,
) -> dict[str, list[float] | np.ndarray]:
    """Integrate an initial velocity and return histories plus final u, v, w arrays.

    Args:
        p: Grid, viscosity, timestep, step-count, and dealiasing parameters.
        initial_velocity: Three real velocity arrays, each with shape (p.n, p.n, p.n).
        record_dissipation: Include a spectral viscous-dissipation history.

    Returns:
        Energy and maximum-divergence histories, plus final velocity components.
        If requested, the result also contains ``dissipation`` at every step.
    """
    n = p.n
    expected_shape = (n, n, n)
    if any(component.shape != expected_shape for component in initial_velocity):
        raise ValueError(
            f"Initial velocity components must have shape {expected_shape}."
        )
    u, v, w = (
        np.array(component, dtype=float, copy=True) for component in initial_velocity
    )

    k1 = _fftfreqs(n)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    mask = _dealias_mask(n) if p.dealias else None

    energies: list[float] = []
    divs: list[float] = []
    dissipations: list[float] | None = [] if record_dissipation else None

    def energy(u_, v_, w_) -> float:
        return 0.5 * float(np.mean(u_**2 + v_**2 + w_**2))

    for step in range(p.steps + 1):
        # diagnostics
        uhat = np.fft.fftn(u); vhat = np.fft.fftn(v); what = np.fft.fftn(w)
        div_hat = 1j*(kx*uhat + ky*vhat + kz*what)
        div = np.fft.ifftn(div_hat).real
        energies.append(energy(u, v, w))
        divs.append(float(np.max(np.abs(div))))
        if dissipations is not None:
            dissipations.append(
                _spectral_kinetic_energy_dissipation(p.nu, uhat, vhat, what, kx, ky, kz)
            )

        if step % 20 == 0:
            print(f"step={step:04d}  KE={energies[-1]:.6e}  max|div|={divs[-1]:.3e}")

        if step == p.steps:
            break

        # Compute nonlinear term in physical space: (u·∇)u
        uhat = np.fft.fftn(u); vhat = np.fft.fftn(v); what = np.fft.fftn(w)

        ux = np.fft.ifftn(1j*kx*uhat).real
        uy = np.fft.ifftn(1j*ky*uhat).real
        uz = np.fft.ifftn(1j*kz*uhat).real

        vx = np.fft.ifftn(1j*kx*vhat).real
        vy = np.fft.ifftn(1j*ky*vhat).real
        vz = np.fft.ifftn(1j*kz*vhat).real

        wx = np.fft.ifftn(1j*kx*what).real
        wy = np.fft.ifftn(1j*ky*what).real
        wz = np.fft.ifftn(1j*kz*what).real

        Nu = u*ux + v*uy + w*uz
        Nv = u*vx + v*vy + w*vz
        Nw = u*wx + v*wy + w*wz

        Nu_hat = np.fft.fftn(Nu)
        Nv_hat = np.fft.fftn(Nv)
        Nw_hat = np.fft.fftn(Nw)

        if mask is not None:
            Nu_hat *= mask
            Nv_hat *= mask
            Nw_hat *= mask

        # semi-implicit diffusion (in Fourier): (I - dt*nu*k^2) u^{*} = u^n - dt*N
        k2 = kx*kx + ky*ky + kz*kz
        denom = (1.0 + p.dt * p.nu * k2)

        uhat_star = (np.fft.fftn(u) - p.dt * Nu_hat) / denom
        vhat_star = (np.fft.fftn(v) - p.dt * Nv_hat) / denom
        what_star = (np.fft.fftn(w) - p.dt * Nw_hat) / denom

        # projection to divergence-free
        uhat_new, vhat_new, what_new = _project_incompressible(uhat_star, vhat_star, what_star, kx, ky, kz)

        u = np.fft.ifftn(uhat_new).real
        v = np.fft.ifftn(vhat_new).real
        w = np.fft.ifftn(what_new).real

    result: dict[str, list[float] | np.ndarray] = {
        "energy": energies,
        "max_div": divs,
        "u": u,
        "v": v,
        "w": w,
    }
    if dissipations is not None:
        result["dissipation"] = dissipations
    return result


def run_taylor_green_3d(p: NS3DParams) -> dict[str, list[float]]:
    """Run parameters p; return the historical energy and max_div histories."""
    result = run_spectral_3d(p, taylor_green_initial(p.n))
    return {"energy": result["energy"], "max_div": result["max_div"]}


def run_abc_beltrami_3d(
    p: NS3DParams,
) -> dict[str, list[float] | np.ndarray | float]:
    """Run ABC/Beltrami and compare final velocity and energy to exact values.

    Args:
        p: Grid, viscosity, timestep, step-count, and dealiasing parameters.

    Returns:
        Histories, final u/v/w arrays, relative_l2_error (RMS vector error divided
        by exact RMS speed), linf_error (maximum pointwise Euclidean velocity
        error), and absolute/relative kinetic-energy errors.
    """
    result = run_spectral_3d(p, abc_beltrami_initial(p.n))
    u_exact, v_exact, w_exact = abc_beltrami_exact(p.n, p.nu, p.steps * p.dt)
    u_error = result["u"] - u_exact
    v_error = result["v"] - v_exact
    w_error = result["w"] - w_exact
    error_magnitude = np.sqrt(u_error**2 + v_error**2 + w_error**2)
    exact_magnitude = np.sqrt(u_exact**2 + v_exact**2 + w_exact**2)
    relative_l2_error = float(
        np.sqrt(np.mean(error_magnitude**2)) / np.sqrt(np.mean(exact_magnitude**2))
    )
    exact_energy = abc_beltrami_energy(p.nu, p.steps * p.dt)
    energy = result["energy"][-1]
    energy_absolute_error = float(abs(energy - exact_energy))
    result.update(
        {
            "relative_l2_error": relative_l2_error,
            "linf_error": float(np.max(error_magnitude)),
            "exact_energy": exact_energy,
            "energy_absolute_error": energy_absolute_error,
            "energy_relative_error": energy_absolute_error / exact_energy,
        }
    )
    return result
