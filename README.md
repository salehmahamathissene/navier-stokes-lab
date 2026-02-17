# navier-stokes-lab

**Educational, research-style** 2D incompressible Navier–Stokes solver on a periodic square.
Implements a **fractional-step (projection) method** with an FFT Poisson solve for pressure,
and validates using the **Taylor–Green vortex** benchmark.

> Goal: demonstrate numerical PDE + scientific computing skills with clean, reproducible experiments.

---

## Mathematical model

We solve the 2D incompressible Navier–Stokes equations on a periodic square:

∂t u + (u·∇)u = −∇p + ν Δu  
∇·u = 0

where:
- u(x,y,t): velocity
- p(x,y,t): pressure
- ν: kinematic viscosity
- domain: [0, 2π]² (periodic)

## Numerical method (projection / fractional step)

At each time step:

1) Advection + diffusion → intermediate velocity u*  
2) Pressure Poisson equation:

Δp = (1/Δt) ∇·u*

3) Projection:

uⁿ⁺¹ = u* − Δt ∇p

This enforces ∇·uⁿ⁺¹ ≈ 0 up to discretization error.

---


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
