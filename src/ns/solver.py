from __future__ import annotations
import numpy as np


def periodic_grad(f: np.ndarray, dx: float):
    # central difference, periodic
    dfdx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dx)
    dfdy = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dx)
    return dfdx, dfdy


def periodic_div(u: np.ndarray, v: np.ndarray, dx: float):
    dudx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dx)
    dvdy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2.0 * dx)
    return dudx + dvdy


def periodic_lap(f: np.ndarray, dx: float):
    return (
        np.roll(f, -1, axis=0) + np.roll(f, 1, axis=0) +
        np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1) -
        4.0 * f
    ) / (dx * dx)


def poisson_solve_fft(rhs: np.ndarray, L: float):
    """
    Solve Δp = rhs on a periodic square domain [0,L]^2 using FFT.
    We set mean(p)=0 by forcing p_hat[0,0]=0.
    """
    n = rhs.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=L/n)
    kx, ky = np.meshgrid(k, k)
    k2 = kx * kx + ky * ky

    rhs_hat = np.fft.fft2(rhs)
    p_hat = np.zeros_like(rhs_hat, dtype=np.complex128)

    # Avoid division by zero at k=0 by leaving p_hat[0,0]=0
    mask = k2 != 0.0
    p_hat[mask] = -rhs_hat[mask] / k2[mask]

    p = np.fft.ifft2(p_hat).real
    return p


def step_projection(u: np.ndarray, v: np.ndarray, nu: float, dt: float, dx: float, L: float):
    """
    One step of 2D incompressible Navier–Stokes on periodic domain using
    - explicit advection (central) + diffusion (explicit)
    - projection to divergence-free using FFT Poisson solve

    This is a simple educational solver (not production CFD).
    """
    # Advection terms: u·∇u and u·∇v
    dudx, dudy = periodic_grad(u, dx)
    dvdx, dvdy = periodic_grad(v, dx)

    adv_u = u * dudx + v * dudy
    adv_v = u * dvdx + v * dvdy

    # Diffusion
    lap_u = periodic_lap(u, dx)
    lap_v = periodic_lap(v, dx)

    # Tentative velocity (u*)
    u_star = u + dt * (-adv_u + nu * lap_u)
    v_star = v + dt * (-adv_v + nu * lap_v)

    # Pressure Poisson: Δp = (1/dt) div(u*)
    rhs = periodic_div(u_star, v_star, dx) / dt
    p = poisson_solve_fft(rhs, L)

    # Correct velocity: u^{n+1} = u* - dt ∇p
    dpdx, dpdy = periodic_grad(p, dx)
    u_new = u_star - dt * dpdx
    v_new = v_star - dt * dpdy

    return u_new, v_new, p
