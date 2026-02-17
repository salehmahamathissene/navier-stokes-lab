import numpy as np
from ns.cavity import CavityParams, run_lid_driven_cavity


def test_cavity_runs_and_stays_finite():
    p = CavityParams(n=65, re=100.0, dt=5e-4, t_end=0.05, poisson_iters=80, save_every=999999)
    sol = run_lid_driven_cavity(p)
    for k in ["psi", "omega", "u", "v"]:
        a = sol[k]
        assert np.isfinite(a).all()
