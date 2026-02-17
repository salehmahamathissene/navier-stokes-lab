import numpy as np

def periodic_div(u, v, dx):
    dudx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
    dvdy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dx)
    return dudx + dvdy

def test_zero_field_divergence():
    n = 32
    dx = 1.0
    u = np.zeros((n, n))
    v = np.zeros((n, n))
    div = periodic_div(u, v, dx)
    assert np.max(np.abs(div)) == 0.0
