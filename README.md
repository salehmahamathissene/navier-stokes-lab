![CI](https://github.com/salehmahamathissene/navier-stokes-lab/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# Navier–Stokes Lab

## About

Navier–Stokes Lab is a Python scientific-computing project containing a 2D
incompressible Navier–Stokes solver using finite differences and a 3D
incompressible Navier–Stokes solver using a Fourier pseudospectral method. Both
are intended for education and numerical experimentation. The 2D solver uses a
fractional-step projection method on a periodic square; the 3D solver uses a
Fourier pseudospectral method on a periodic cube. The repository also includes
a lid-driven-cavity example. It is not presented as a production CFD package or
as a certified solver.

## Numerical methods: 2D and 3D solvers

The two solvers use distinct spatial discretizations and pressure treatments:

**2D periodic solver** (`src/ns/solver.py`, demonstrated by
`scripts/run_taylor_green.py`)

- Advances incompressible flow on a periodic square.
- Uses central finite differences for derivatives, advection, and diffusion.
- Treats both advection and diffusion explicitly to form an intermediate
  velocity.
- Solves a pressure Poisson equation with an FFT, then corrects velocity using
  the pressure gradient.
- Includes a Taylor–Green vortex demonstration.

**3D spectral solver** (`src/ns/solver3d.py`)

- Uses a periodic three-dimensional Fourier pseudospectral formulation with
  spectral derivatives.
- Forms the nonlinear advective term in physical space and treats it
  explicitly.
- Treats diffusion implicitly with backward Euler in Fourier space.
- Projects velocity onto the divergence-free subspace in Fourier space.
- Applies optional 2/3-rule dealiasing to the nonlinear term; it is enabled by
  default.
- Includes a Taylor–Green initial condition and an ABC Beltrami exact-solution
  benchmark.

## Scientific overview

The 3D solver advances the incompressible Navier–Stokes equations,

$$
\partial_t \mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u}
= -\nabla p + \nu \nabla^2\mathbf{u},
\qquad \nabla\cdot\mathbf{u}=0,
$$

on the periodic cube $[0,2\pi)^3$, sampled on an $n\times n\times n$ grid.
Spatial derivatives are evaluated spectrally using FFTs. The nonlinear
advective term is formed in physical space from the Fourier derivatives, then
transformed back to Fourier space. When dealiasing is enabled, a 2/3-rule mask
is applied to its Fourier coefficients; the mask retains coordinate modes
through $\lfloor n/3\rfloor$.

For each step, the solver treats advection explicitly and diffusion implicitly:

$$
\widehat{\mathbf{u}}^{*}
=
\frac{
\widehat{\mathbf{u}}^{\,n}
-
\Delta t\,\widehat{(\mathbf{u}\cdot\nabla)\mathbf{u}}^{\,n}
}{
1+\nu\Delta t|\mathbf{k}|^2
}.
$$

It then projects the intermediate velocity onto the divergence-free subspace
in Fourier space. For nonzero wavevectors the projection is

$$
\widehat{\mathbf{u}}^{\,n+1}
= \left(I-\frac{\mathbf{k}\mathbf{k}^{\mathsf T}}{|\mathbf{k}|^2}\right)
\widehat{\mathbf{u}}^{*}, \qquad \mathbf{k}\ne 0.
$$

The zero-frequency mean is preserved. This is explicit Euler treatment of the
nonlinear term with backward Euler diffusion and a subsequent incompressibility
projection. The 3D implementation does not solve for or return a pressure
field.

## Implemented benchmarks

### Taylor–Green initial condition

The 3D solver provides a Taylor–Green initial condition and a runner that
records kinetic-energy and maximum-divergence histories. Run
`python scripts/run_taylor_green_3d.py` to generate those histories and save
plots under `results/`. A separate 2D periodic projection demo is available as
`python scripts/run_taylor_green.py`; it reports energy and divergence and
saves energy, divergence, and vorticity plots.

### ABC Beltrami exact solution

The 3D ABC benchmark uses the initial velocity field

$$
\mathbf{A}(x,y,z)
= \bigl(\sin z+\cos y,\ \sin x+\cos z,\ \sin y+\cos x\bigr).
$$

It satisfies

$$
\nabla\cdot\mathbf{A}=0,\qquad
\nabla\times\mathbf{A}=\mathbf{A},\qquad
\nabla^2\mathbf{A}=-\mathbf{A}.
$$

For kinematic viscosity $\nu$, the exact velocity and domain-mean kinetic
energy are

$$
\mathbf{u}(\mathbf{x},t)=e^{-\nu t}\mathbf{A}(\mathbf{x})
$$

$$
E(t)=\frac{3}{2}e^{-2\nu t}
$$

An analytical pressure satisfying the momentum equation is

$$
p(\mathbf{x},t)
=-\frac{1}{2}e^{-2\nu t}|\mathbf{A}(\mathbf{x})|^2+C(t),
$$

where $C(t)$ is an arbitrary spatially uniform pressure gauge. This is an
analytical reference only: the 3D solver does not reconstruct pressure, and
the benchmark does not compare a numerical pressure field.

## Installation and setup

Python 3.10 or later is required. The project declares NumPy, SciPy, and
Matplotlib as runtime dependencies, and provides pytest as an optional
development dependency (`pytest>=8`). CI is configured for Python 3.11 and
3.12.

```bash
git clone https://github.com/salehmahamathissene/navier-stokes-lab.git
cd navier-stokes-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1` instead of the `source` command.

## Running the ABC benchmark

Run the benchmark with the repository's script:

```bash
python scripts/run_abc_beltrami_3d.py
```

The available arguments and defaults are:

| Argument | Default | Description |
| --- | ---: | --- |
| `--n` | `16` | Grid points in each direction |
| `--nu` | `0.01` | Kinematic viscosity |
| `--dt` | `0.0025` | Time step |
| `--steps` | `40` | Number of time steps |
| `--no-dealias` | off | Disable the default 2/3-rule dealiasing |

For example:

```bash
python scripts/run_abc_beltrami_3d.py --n 16 --nu 0.01 --dt 0.0025 --steps 40
```

The script prints a JSON report. Its diagnostic fields are:

| Output field | Meaning |
| --- | --- |
| `relative_l2_error` | RMS Euclidean velocity error divided by RMS exact speed: $\sqrt{\langle\lVert\mathbf{u}_{\mathrm{num}}-\mathbf{u}_{\mathrm{exact}}\rVert_2^2\rangle}/\sqrt{\langle\lVert\mathbf{u}_{\mathrm{exact}}\rVert_2^2\rangle}$ |
| `linf_error` | Maximum pointwise Euclidean velocity error: $\max_{\mathbf{x}}\lVert\mathbf{u}_{\mathrm{num}}-\mathbf{u}_{\mathrm{exact}}\rVert_2$ |
| `numerical_energy` | Final numerical mean kinetic energy, one half of the spatial average of squared velocity magnitude |
| `exact_energy` | Exact mean kinetic energy at the final time |
| `energy_absolute_error` | Absolute difference between numerical and exact energy |
| `energy_relative_error` | Absolute energy error divided by exact energy |
| `final_max_divergence` | Maximum absolute spectral divergence at the final time: $\max_{\mathbf{x}}\lvert\nabla\cdot\mathbf{u}\rvert$ |

The report also includes the grid size, viscosity, time step, step count, final
time, and whether dealiasing is enabled.

## Validation and tests

The automated test suite covers solver behavior and benchmark properties. Its
ABC tests check the initial field, divergence, a short exact-solution
comparison, and energy. The suite also includes projection and existing 2D
solver tests.

The ABC temporal-refinement test holds the final time and grid fixed
($n=8$, $\nu=0.1$, final time $0.1$) while reducing $\Delta t$ from
0.01 to 0.005 to 0.0025. It checks decreasing errors and successive error
ratios between 1.8 and 2.2, consistent with approximately first-order
temporal behavior for that tested configuration. A separate low-mode test
checks the ABC error on grids $n=8,12,16$. Because this field is resolved on
each grid and the test checks grid-independence rather than an asymptotic error
rate, it is not evidence of spectral convergence.

The published test suite reports **15 tests passed**. These checks, including
exact-solution and temporal-refinement tests, are limited to their tested
cases; they do not establish accuracy for every flow or grid.

## Project structure

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── src/
│   ├── ns/
│   │   ├── solver.py
│   │   ├── solver3d.py
│   │   └── cavity.py
│   └── navier_stokes_lab/
│       ├── cavity.py
│       └── scripts/run_cavity_fast.py
├── tests/
│   ├── test_abc_beltrami.py
│   ├── test_cavity_smoke.py
│   ├── test_divergence.py
│   ├── test_energy_monotone.py
│   ├── test_fft_poisson.py
│   └── test_solver3d_projection.py
└── scripts/
    ├── run_abc_beltrami_3d.py
    ├── run_cavity.py
    ├── run_taylor_green.py
    └── run_taylor_green_3d.py
```

## Numerical limitations and scope

- The 3D spectral solver uses a periodic Fourier formulation on a cube; it does
  not implement non-periodic boundary conditions.
- The ABC benchmark verifies one specific smooth exact flow and does not
  establish accuracy for turbulent, high-Reynolds-number, or otherwise complex
  flows.
- Pressure reconstruction is not established by this benchmark: the 3D solver
  projects velocity in Fourier space and does not return pressure.
- The included low-mode grid comparison is not a spatial-convergence study.
- The repository does not claim production CFD certification or industrial
  readiness.

## Roadmap

The following items are proposed future work, not implemented features:

- Add a more demanding Taylor–Green vortex validation study.
- Conduct broader temporal- and spatial-convergence studies.
- Implement and verify pressure reconstruction.
- Add conservation and stability diagnostics.
- Improve scientific reporting and experiment reproducibility.
