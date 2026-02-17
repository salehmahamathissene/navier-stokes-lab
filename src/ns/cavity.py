from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class CavityParams:
    n: int = 129              # grid points per direction (odd is nice)
    re: float = 1000.0        # Reynolds number (U*L/nu), L=1, U=1
    dt: float = 2.5e-4        # timestep
    t_end: float = 10.0       # final time
    lid_u: float = 1.0        # lid velocity
    save_every: int = 200     # steps
    poisson_iters: int = 200  # Jacobi iterations per step
    poisson_tol: float = 0.0  # set >0 to early-stop (optional)


def _laplacian(f: np.ndarray, h: float) -> np.ndarray:
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = (
        (f[2:, 1:-1] - 2.0 * f[1:-1, 1:-1] + f[:-2, 1:-1]) +
        (f[1:-1, 2:] - 2.0 * f[1:-1, 1:-1] + f[1:-1, :-2])
    ) / (h * h)
    return lap


def _jacobi_poisson_dirichlet(psi: np.ndarray, rhs: np.ndarray, h: float, iters: int, tol: float) -> np.ndarray:
    # Solve ∇² psi = rhs with psi=0 on boundary (Dirichlet)
    # Jacobi: psi_new = 0.25*(psi_E+psi_W+psi_N+psi_S - h^2*rhs)
    psi_new = psi.copy()
    h2 = h * h
    for _ in range(iters):
        psi_new[1:-1, 1:-1] = 0.25 * (
            psi[1:-1, 2:] + psi[1:-1, :-2] + psi[2:, 1:-1] + psi[:-2, 1:-1]
            - h2 * rhs[1:-1, 1:-1]
        )
        # Dirichlet boundary psi = 0
        psi_new[0, :] = 0.0
        psi_new[-1, :] = 0.0
        psi_new[:, 0] = 0.0
        psi_new[:, -1] = 0.0

        if tol > 0.0:
            # residual check: r = ∇²psi - rhs
            r = _laplacian(psi_new, h) - rhs
            if np.max(np.abs(r[1:-1, 1:-1])) < tol:
                psi = psi_new
                break

        psi, psi_new = psi_new, psi
    return psi


def _velocity_from_psi(psi: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    # u = dpsi/dy, v = -dpsi/dx
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * h)
    v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * h)
    return u, v


def _apply_boundary_conditions(u: np.ndarray, v: np.ndarray, lid_u: float) -> None:
    # No-slip: u=v=0 on walls except top lid where u=lid_u, v=0
    u[0, :] = 0.0
    v[0, :] = 0.0

    u[-1, :] = lid_u
    v[-1, :] = 0.0

    u[:, 0] = 0.0
    v[:, 0] = 0.0

    u[:, -1] = 0.0
    v[:, -1] = 0.0


def _boundary_vorticity_from_psi(omega: np.ndarray, psi: np.ndarray, h: float, lid_u: float) -> None:
    # Thom boundary formula for vorticity, with psi=0 on all boundaries
    # Bottom y=0:
    omega[0, 1:-1] = -2.0 * (psi[1, 1:-1] - psi[0, 1:-1]) / (h * h)
    # Top y=1 (lid moving u=lid_u):
    omega[-1, 1:-1] = -2.0 * (psi[-2, 1:-1] - psi[-1, 1:-1]) / (h * h) - 2.0 * lid_u / h
    # Left x=0:
    omega[1:-1, 0] = -2.0 * (psi[1:-1, 1] - psi[1:-1, 0]) / (h * h)
    # Right x=1:
    omega[1:-1, -1] = -2.0 * (psi[1:-1, -2] - psi[1:-1, -1]) / (h * h)

    # Corners (simple average of adjacent edges)
    omega[0, 0] = 0.5 * (omega[0, 1] + omega[1, 0])
    omega[0, -1] = 0.5 * (omega[0, -2] + omega[1, -1])
    omega[-1, 0] = 0.5 * (omega[-2, 0] + omega[-1, 1])
    omega[-1, -1] = 0.5 * (omega[-2, -1] + omega[-1, -2])


def run_lid_driven_cavity(p: CavityParams) -> dict[str, np.ndarray]:
    n = p.n
    h = 1.0 / (n - 1)
    nu = 1.0 / p.re

    psi = np.zeros((n, n), dtype=float)
    omega = np.zeros((n, n), dtype=float)

    # initial velocity from psi (all zero)
    u, v = _velocity_from_psi(psi, h)
    _apply_boundary_conditions(u, v, p.lid_u)

    t = 0.0
    step = 0

    # for monitoring
    def kinetic_energy(u_: np.ndarray, v_: np.ndarray) -> float:
        return 0.5 * np.mean(u_**2 + v_**2)

    while t < p.t_end:
        # 1) solve Poisson for psi: ∇² psi = -omega
        psi = _jacobi_poisson_dirichlet(psi, rhs=-omega, h=h, iters=p.poisson_iters, tol=p.poisson_tol)

        # 2) compute velocity from psi
        u, v = _velocity_from_psi(psi, h)
        _apply_boundary_conditions(u, v, p.lid_u)

        # 3) boundary vorticity (depends on psi and lid speed)
        _boundary_vorticity_from_psi(omega, psi, h, p.lid_u)

        # 4) advance vorticity transport: ω_t + u ω_x + v ω_y = ν ∇² ω
        omega_x = np.zeros_like(omega)
        omega_y = np.zeros_like(omega)

        omega_x[1:-1, 1:-1] = (omega[2:, 1:-1] - omega[:-2, 1:-1]) / (2.0 * h)
        omega_y[1:-1, 1:-1] = (omega[1:-1, 2:] - omega[1:-1, :-2]) / (2.0 * h)

        adv = u * omega_x + v * omega_y
        diff = nu * _laplacian(omega, h)

        omega_new = omega.copy()
        omega_new[1:-1, 1:-1] = omega[1:-1, 1:-1] + p.dt * (-adv[1:-1, 1:-1] + diff[1:-1, 1:-1])

        omega = omega_new
        # reapply boundary vorticity after update
        _boundary_vorticity_from_psi(omega, psi, h, p.lid_u)

        t += p.dt
        step += 1

        if step % p.save_every == 0:
            ke = kinetic_energy(u, v)
            wmax = float(np.max(np.abs(omega)))
            print(f"step={step:06d} t={t:.4f}  KE={ke:.6e}  max|ω|={wmax:.3e}")

    return {"psi": psi, "omega": omega, "u": u, "v": v}
