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


def taylor_green_initial(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(u)
    return u, v, w


def _grad_hat(fhat: np.ndarray, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return 1j*kx*fhat, 1j*ky*fhat, 1j*kz*fhat


def _project_incompressible(uhat: np.ndarray, vhat: np.ndarray, what: np.ndarray, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k2 = kx*kx + ky*ky + kz*kz
    k2[0, 0, 0] = 1.0  # avoid divide-by-zero at mean mode

    div_hat = 1j*(kx*uhat + ky*vhat + kz*what)
    # subtract gradient of pressure: u <- u - k*(div)/k^2
    uhat_p = uhat - 1j*kx*div_hat / k2
    vhat_p = vhat - 1j*ky*div_hat / k2
    what_p = what - 1j*kz*div_hat / k2

    # set mean mode unchanged
    uhat_p[0, 0, 0] = uhat[0, 0, 0]
    vhat_p[0, 0, 0] = vhat[0, 0, 0]
    what_p[0, 0, 0] = what[0, 0, 0]
    return uhat_p, vhat_p, what_p


def run_taylor_green_3d(p: NS3DParams) -> dict[str, list[float]]:
    n = p.n
    u, v, w = taylor_green_initial(n)

    k1 = _fftfreqs(n)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    mask = _dealias_mask(n) if p.dealias else None

    energies: list[float] = []
    divs: list[float] = []

    def energy(u_, v_, w_) -> float:
        return 0.5 * float(np.mean(u_**2 + v_**2 + w_**2))

    for step in range(p.steps + 1):
        # diagnostics
        uhat = np.fft.fftn(u); vhat = np.fft.fftn(v); what = np.fft.fftn(w)
        div_hat = 1j*(kx*uhat + ky*vhat + kz*what)
        div = np.fft.ifftn(div_hat).real
        energies.append(energy(u, v, w))
        divs.append(float(np.max(np.abs(div))))

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

    return {"energy": energies, "max_div": divs}
