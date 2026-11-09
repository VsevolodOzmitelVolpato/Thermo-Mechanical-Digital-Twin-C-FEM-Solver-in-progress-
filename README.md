# thermomech-twin

A thermo-mechanical *digital twin* of a heated component: a C++ finite-element core, with a Python reduced-order model and uncertainty quantification planned on top. The goal is a small simulation stack in which every layer is checked against hand calculations, exact solutions and measured convergence orders before the next layer is built.

**Status: work in progress.** The 1D layer is complete and verified: heat conduction, and the chain temperature -> thermal expansion -> stress (see below). 2D/3D and the Python layer are next.

## What works today

- Sparse matrices in CSR format, Conjugate Gradient (CG) and Jacobi-preconditioned CG.
- 1D linear (P1) finite elements for `rho_c dT/dt - k d2T/dx2 = q`: stiffness, consistent mass and load assembly.
- Dirichlet conditions by symmetric elimination, so the system matrix stays symmetric and CG applies.
- Steady-state solver, and time stepping with the theta scheme (`theta = 1`: backward Euler, `theta = 0.5`: Crank-Nicolson):

  `(M + theta dt K) T_new = (M - (1 - theta) dt K) T_old + dt F`
- Thermal expansion and thermal stress in 1D (linear elasticity, `sigma = E (u' - alpha (T - T_ref))`): equivalent thermal nodal forces, the displacement of a bar clamped at the left end and free or clamped at the right end, and the stress in every element.
- The whole chain in one call: `solve_thermomechanical(problem)` runs the transient heat solver, gives its final temperature to the expansion solver, and returns temperature, displacement and stress. The coupling is one-way: heat deforms the bar, the deformation does not heat it.
- Failures are reported, not hidden: `CGSolver` exposes `has_converged()`, and the steady, transient and thermal-expansion solvers throw `std::runtime_error` when the linear solver does not converge (for example with NaN in the data).
- 73 unit tests (GoogleTest) and a program that prints the convergence tables below.

## Build and test

Requirements: CMake 3.20 or newer and a C++17 compiler. Developed with GCC 14 (MinGW-w64, Windows); it also builds and passes all tests with GCC 13 on Linux. GoogleTest is downloaded by CMake the first time you configure, so an internet connection is needed once.

```
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

`-G Ninja` is optional; any CMake generator works. To reproduce the tables below, run `./build/convergence_study` (`.\build\convergence_study.exe` on Windows).

## Example

A steel bar 1 m long. One end is held at 20 degrees C and the other at 120 degrees C; initially everything is at 20 degrees C except the hot end. After 100 000 s (about twelve of the slowest thermal time constants) the temperature is practically the steady linear profile, so the results can be checked by hand: the mean temperature rise is 50 K.

```cpp
#include "Thermoelastic1D.h"

ThermoMechanicalProblem p;
p.x = uniform_mesh(1.0, 100);          // 100 elements, 1 m
p.rho_c = 3.8e6;                       // density * specific heat [J/(m3 K)]
p.k = 45.0;                            // conductivity [W/(m K)]
p.q = 0.0;                             // no heat source
p.T_left = 20.0;                       // temperatures of the two ends
p.T_right = 120.0;
p.T0 = DVector(p.x.size(), 20.0);      // initial temperature, one value per node
p.T0.back() = p.T_right;
p.dt = 1000.0;                         // 100 steps of 1000 s
p.n_steps = 100;
p.theta = 1.0;                         // backward Euler
p.E = 200e9;                           // Young modulus [Pa]
p.alpha = 12e-6;                       // thermal expansion [1/K]
p.T_ref = 20.0;                        // temperature at which the bar has no stress

p.right_end = RightEnd::Fixed;         // or RightEnd::Free
ThermoMechanicalResult r = solve_thermomechanical(p);
// r.T[i]: temperature [deg C], r.u[i]: displacement [m], r.sigma[e]: stress [Pa] (one per element)
```

With `RightEnd::Free` the tip moves by 0.6000 mm (`alpha * 50 K * 1 m`) and the stress is zero up to round-off. With `RightEnd::Fixed` the bar cannot expand and the stress is -120.0 MPa in every element (`-E * alpha * 50 K`), i.e. compression.

## Verification

How do we know the solver is right? Three kinds of evidence, from simple to demanding.

1. **Hand calculations.** Tiny meshes whose results can be written on paper: stiffness, load, mass, Dirichlet elimination, the L2 error and the observed order, thermal forces, displacements and stresses of bars with a few elements, and a complete one-step heat -> expansion -> stress example.
2. **Exact solutions.**
   - Steady state with constant `k` and `q`: `T(x) = T_left + (T_right - T_left) x / L + q x (L - x) / (2k)`. For this problem linear elements give the exact nodal values on any mesh, including graded ones, so the tests use a tolerance of 1e-9.
   - Time stepping: on a uniform mesh, `sin(pi x / L)` is an eigenvector of both `K` and `M`, so the theta scheme multiplies it by a known factor `g` at every step. This gives the exact solution of the *discrete scheme* (not of the PDE), and the tests compare the solver with `g^n sin(pi x / L)` to within 1e-9, for three values of `theta`.
   - Properties: the stationary solution is a fixed point of the scheme, any initial condition relaxes to it, the mass matrix sums to the total heat capacity.
   - Thermal stress: a bar heated by the same `dT` everywhere and clamped at both ends has the uniform stress `-E alpha dT` (240 MPa of compression for steel heated by 100 K); with a free end the stress is zero. A free bar has zero stress for *any* temperature field on *any* mesh. A linear temperature on a graded mesh gives the clamped bar a uniform stress, equal to `-E alpha` times the mean temperature rise.
3. **Convergence orders** (below).

### Convergence study

Test problem: a bar of length 1, `k = 1`, `rho_c = 1`, no source, both ends held at 0, initial temperature `sin(pi x)`. The exact solution is `T(x, t) = exp(-pi^2 t) sin(pi x)`.

- **Error:** the L2 norm of the difference between the numerical and the exact nodal values at `t = 0.1`. Because both fields are piecewise linear, it is computed as `sqrt(e^T M e)`, with `M` the mass matrix for `rho_c = 1`.
- **Observed order:** if the error behaves like `C h^p`, then `p = ln(e_coarse / e_fine) / ln(ratio)`, where `ratio` is the refinement factor (here always 2).
- **Separating the two error sources:** the total error is a mesh error plus a time error. The spatial study uses a tiny time step so that the time error is negligible. The temporal studies use a fine mesh (`N = 800`) so that the mesh error (about 3.5e-7) is far below the time error.

### Spatial convergence (Crank-Nicolson, dt = 1e-4, t = 0.1)

|     N |        h | L2 error | order |
|------:|---------:|---------:|------:|
|    10 | 1.00e-01 | 2.12e-03 |     - |
|    20 | 5.00e-02 | 5.34e-04 |  1.99 |
|    40 | 2.50e-02 | 1.34e-04 |  2.00 |
|    80 | 1.25e-02 | 3.34e-05 |  2.00 |
|   160 | 6.25e-03 | 8.38e-06 |  2.00 |

### Temporal convergence (backward Euler, N = 800, t = 0.1)

| steps |       dt | L2 error | order |
|------:|---------:|---------:|------:|
|     5 | 2.00e-02 | 2.37e-02 |     - |
|    10 | 1.00e-02 | 1.23e-02 |  0.94 |
|    20 | 5.00e-03 | 6.29e-03 |  0.97 |
|    40 | 2.50e-03 | 3.18e-03 |  0.99 |

### Temporal convergence (Crank-Nicolson, N = 800, t = 0.1)

| steps |       dt | L2 error | order |
|------:|---------:|---------:|------:|
|     5 | 2.00e-02 | 8.49e-04 |     - |
|    10 | 1.00e-02 | 2.12e-04 |  2.00 |
|    20 | 5.00e-03 | 5.31e-05 |  1.99 |
|    40 | 2.50e-03 | 1.35e-05 |  1.97 |

**Reading the tables.**
- Linear elements are second order in space: halving `h` divides the error by 4.
- Backward Euler is first order in time. The observed order approaches 1 from below as `dt` decreases (0.94, 0.97, 0.99): an order is a statement about the limit `dt -> 0`.
- Crank-Nicolson is second order in time. The small drop at the finest `dt` (1.97) is the mesh error starting to show next to a very small time error, not a loss of accuracy of the scheme: on a finer mesh (`N = 1600`) the three orders are 2.00, 2.00 and 1.99.

### Convergence study of the thermo-mechanical chain

The same decaying-sine problem, followed through the whole chain: temperature, then expansion, then stress. With `T_ref = 0`, `E = 3` and `alpha = 2` the chain has two exact results at time `t`:

- Bar with a free right end: the tip moves by `alpha` times the integral of `T`, that is `2 alpha exp(-pi^2 t) / pi`.
- Bar clamped at both ends: the stress is the same in every element, `-E alpha` times the mean of `T`, that is `-2 E alpha exp(-pi^2 t) / pi`.

The tables report the absolute error of the tip displacement (free bar) and the largest absolute error of the stress over the elements (clamped bar) at `t = 0.1`. The observed order is computed on the tip error. The spatial and temporal setups are the same as above.

### Chain: spatial convergence (Crank-Nicolson, dt = 1e-4, t = 0.1)

|     N |        h | tip error | order | stress error |
|------:|---------:|----------:|------:|-------------:|
|    10 | 1.00e-01 |  7.73e-03 |     - |     2.32e-02 |
|    20 | 5.00e-02 |  1.94e-03 |  2.00 |     5.81e-03 |
|    40 | 2.50e-02 |  4.85e-04 |  2.00 |     1.45e-03 |
|    80 | 1.25e-02 |  1.21e-04 |  2.00 |     3.64e-04 |
|   160 | 6.25e-03 |  3.03e-05 |  2.00 |     9.10e-05 |

### Chain: temporal convergence (backward Euler, N = 800, t = 0.1)

| steps |       dt | tip error | order | stress error |
|------:|---------:|----------:|------:|-------------:|
|     5 | 2.00e-02 |  4.27e-02 |     - |     1.28e-01 |
|    10 | 1.00e-02 |  2.22e-02 |  0.94 |     6.66e-02 |
|    20 | 5.00e-03 |  1.13e-02 |  0.97 |     3.40e-02 |
|    40 | 2.50e-03 |  5.72e-03 |  0.99 |     1.72e-02 |

### Chain: temporal convergence (Crank-Nicolson, N = 800, t = 0.1)

| steps |       dt | tip error | order | stress error |
|------:|---------:|----------:|------:|-------------:|
|     5 | 2.00e-02 |  1.53e-03 |     - |     4.59e-03 |
|    10 | 1.00e-02 |  3.82e-04 |  2.00 |     1.15e-03 |
|    20 | 5.00e-03 |  9.63e-05 |  1.99 |     2.89e-04 |
|    40 | 2.50e-03 |  2.50e-05 |  1.95 |     7.49e-05 |

**Reading the tables.**
- Displacement and stress inherit the orders of the temperature they come from: second order in space, first order in time for backward Euler (0.94, 0.97, 0.99, as above) and second order in time for Crank-Nicolson. The 1.95 in the last row is again the mesh error showing next to a very small time error (with `N = 1600` the three orders are 2.00, 2.00 and 1.99).
- The stress error is `E / L = 3` times the tip error in every row (up to the printed digits). In the discrete problem, the equilibrium of the clamped bar forces a uniform stress equal to `-(E / L)` times the tip displacement of the free bar. The two columns come from two different solves, free and clamped, so the constant ratio is a free consistency check of the mechanics.

## Project layout

```
include/   CSR_and_Vectors.h, CGSolver.h, JacobiPreconditioner.h, Fem1D.h, Thermoelastic1D.h
src/       implementations
tests/     GoogleTest unit tests: test_csr.cpp, test_cg.cpp, test_fem1d.cpp, test_thermoelastic1d.cpp
apps/      convergence_study.cpp  (prints the tables above)
```

## Design notes

- **Parameter structs instead of long argument lists.** `solve_thermomechanical` takes one `ThermoMechanicalProblem` and returns one `ThermoMechanicalResult`. The alternative is 14 arguments, 10 of them `double`, where swapping two of them compiles without a warning. The fields are public and independent of each other: while a problem is being filled in nothing has to hold between them, so there is no invariant for a class to protect. Validity (matching vector sizes, `dt > 0`, `theta` in [0, 1], `E > 0`, ...) is checked where the data is used, by the three solvers, which throw `std::invalid_argument` or `std::runtime_error`; the tests check that these errors reach the caller. A parameter sweep is then a copy of the struct and a change of one field.
- **Classes where there is state to protect.** `CSR`, `CGSolver` and `JacobiPreconditioner` are classes: their internal data must stay consistent with each other.

## Known limitations

- Matrices are assembled as dense tables, and the system matrix of the time loop is rebuilt and converted to CSR at every step even though it does not change. This is fine in 1D (a few seconds for the whole study) and will be replaced by sparse assembly in 2D.
- There is no stability check for `theta < 0.5`. With a time step above the stability limit the temperatures grow at every step. The solver throws only when the growth produces NaN or infinity; otherwise it returns huge but finite numbers. The schemes used in the convergence study (`theta = 1` and `theta = 0.5`) are unconditionally stable.
- The thermo-mechanical model is deliberately simple: one-dimensional, linear elastic, small strain, quasi-static, with constant material properties. The left end of the bar is always clamped. The coupling is one-way: the deformation does not heat the material. For steel the size of that effect, `E alpha^2 T / (rho c)`, is about 2e-3, which is why it is commonly neglected.
- The heat equation accepts only fixed-temperature ends (Dirichlet conditions). Convection and heat-flux conditions are not implemented.

## Roadmap

Planned, in this order:

1. 2D triangular elements with sparse assembly.
2. Python interface and a proper-orthogonal-decomposition (POD) reduced-order model.
3. Sobol sensitivity analysis on the reduced model.

## Author

Vsevolod Ozmitel Volpato, (for my contacts see my main github page)

note : la matrice del ciclo temporale viene ricostruita a ogni passo scrivilo
#   T h e r m o - M e c h a n i c a l - D i g i t a l - T w i n - C - F E M - S o l v e r - i n - p r o g r e s s -  
 