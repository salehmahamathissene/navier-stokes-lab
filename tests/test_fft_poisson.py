import numpy as np

def solve_poisson_fft(f):
    n, m = f.shape
    fk = np.fft.fftn(f)

    kx = 2*np.pi*np.fft.fftfreq(m)
    ky = 2*np.pi*np.fft.fftfreq(n)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX*KX + KY*KY
    K2[0, 0] = 1.0

    pk = -fk / K2
    pk[0, 0] = 0.0
    p = np.fft.ifftn(pk).real
    return p

def laplacian_fft(p):
    n, m = p.shape
    pk = np.fft.fftn(p)

    kx = 2*np.pi*np.fft.fftfreq(m)
    ky = 2*np.pi*np.fft.fftfreq(n)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX*KX + KY*KY

    lk = -K2 * pk
    l = np.fft.ifftn(lk).real
    return l

def test_poisson_fft_residual_small():
    n = 32
    rng = np.random.default_rng(0)
    f = rng.standard_normal((n, n))
    f = f - f.mean()  # solvability for periodic Poisson
    p = solve_poisson_fft(f)
    f2 = laplacian_fft(p)
    err = np.linalg.norm(f2 - f) / np.linalg.norm(f)
    assert err < 1e-10
