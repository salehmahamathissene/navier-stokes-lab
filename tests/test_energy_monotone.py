import numpy as np

def test_energy_nonnegative():
    u = np.random.randn(16, 16)
    v = np.random.randn(16, 16)
    ke = 0.5 * np.mean(u*u + v*v)
    assert ke >= 0.0
