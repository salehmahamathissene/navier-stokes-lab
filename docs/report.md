# navier-stokes-lab: short technical report

## Objective
Implement and validate an incompressible 2D Navier–Stokes solver suitable for
scientific computing practice: reproducible runs, numerical sanity checks, and tests.

## Method
- Fractional-step / projection scheme
- Periodic domain [0, 2π]²
- Poisson equation solved with FFT

## Benchmark: Taylor–Green vortex
Metrics:
- kinetic energy decay (physics sanity)
- max divergence max|∇·u| (incompressibility)

Expected behavior:
- energy decreases monotonically with viscosity
- divergence remains small (near 1e−5–1e−8 depending on dt/grid)

## Outputs
- results/energy.png
- results/divergence.png
- results/vorticity.png

## Next steps
- convergence study vs grid size (L2 error)
- RK2 / RK3 time stepping
- lid-driven cavity (non-periodic BCs)
