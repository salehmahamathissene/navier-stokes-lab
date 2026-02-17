# navier-stokes-lab

**Educational, research-style** 2D incompressible Navier–Stokes solver on a periodic square.
Implements a **fractional-step (projection) method** with an FFT Poisson solve for pressure,
and validates using the **Taylor–Green vortex** benchmark.

> Goal: demonstrate numerical PDE + scientific computing skills with clean, reproducible experiments.

---

## Mathematical model

We solve the incompressible Navier–Stokes equations:
\[
\partial_t \mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u} = -\nabla p + \nu \Delta \mathbf{u},\qquad
\nabla\cdot\mathbf{u}=0.
\]

- \(\mathbf{u}(x,y,t)\): velocity
- \(p(x,y,t)\): pressure
- \(\nu\): kinematic viscosity

Domain: periodic square \([0,2\pi]^2\).

---

## Numerical method (projection)

At each step:
1. **Advection + diffusion** to compute intermediate velocity \( \mathbf{u}^\* \)
2. **Pressure Poisson solve**
\[
\Delta p = \frac{1}{\Delta t}\nabla\cdot \mathbf{u}^\*
\]
3. **Projection**
\[
\mathbf{u}^{n+1} = \mathbf{u}^\* - \Delta t \nabla p
\]
so that \(\nabla\cdot \mathbf{u}^{n+1} \approx 0\).

---

## Benchmark: Taylor–Green vortex

The solver includes a Taylor–Green vortex run that reports:
- kinetic energy decay (sanity/physics check)
- maximum divergence \(\max|\nabla\cdot u|\) (incompressibility check)
- vorticity snapshots

Example run output:
- `max|div| ~ 1e-5` (expected small nonzero due to discretization & timestep)

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
python scripts/run_taylor_green.py
