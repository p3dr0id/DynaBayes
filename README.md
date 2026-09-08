# DynaBayes

**Dynamic Bayesian Inference for Coupled Phase Oscillators**

DynaBayes is a Python package for simulation and time-evolving Bayesian inference of parameters in networks of coupled phase oscillators. The package provides a compact interface for a phase model containing direct phase terms and phase-difference coupling, together with posterior uncertainty and noise-covariance diagnostics.

---

## Model formulation

The current model is

```math
\frac{d\phi_i}{dt} = \omega_i(t)
+ \sum_{j=1}^{N} A_{ij}(t)\sin(\phi_j)
+ \sum_{j=1}^{N} B_{ij}(t)\sin(\phi_j - \phi_i)
+ \xi_i(t)
```

with stochastic terms described by

```math
\langle \xi_i(t)\xi_j(t') \rangle
= E_{ij}\,\delta(t-t')
```

Here:

- `phi_i` is the phase of oscillator `i`;
- `omega_i(t)` is its time-dependent intrinsic frequency;
- `A_ij(t)` multiplies the direct phase term `sin(phi_j)`;
- `B_ij(t)` multiplies the phase-difference term `sin(phi_j - phi_i)`;
- `E` is the inferred noise covariance matrix.

Because

```math
B_{ii}\sin(\phi_i-\phi_i) = 0
```

the diagonal `B_ii` terms are structural zeros. They are retained in the historical output layout for clarity and compatibility, but they are not included in the inverse problem.

---

## Methodology and validation

DynaBayes is an independent Python implementation of the time-evolving dynamical Bayesian inference methodology described in:

> Stankovski, T., Ticcinelli, V., McClintock, P. V. E. & Stefanovska, A. (2014). A tutorial on time-evolving dynamical Bayesian inference. *European Physical Journal Special Topics*, 222, 2467–2485. https://doi.org/10.1140/epjst/e2014-02286-7

The sequential inference implementation uses midpoint discretization, iterative estimation of the full noise covariance matrix, a fixed prior during the internal parameter/noise iteration, and posterior-to-prior covariance propagation between consecutive non-overlapping windows.

The numerical behaviour of the implementation has been cross-validated against the independently developed MODA reference software and against synthetic systems with known time-varying parameters. MODA source code is **not** incorporated into DynaBayes and MODA is not a package dependency.

Automated regression tests verify parameter tracking, passage of a coupling coefficient through zero, noise-covariance recovery, structural zeros, convergence, and public-API behaviour.

---

## Installation

Install the released package from PyPI:

```bash
pip install dynabayes
```

For development from GitHub:

```bash
git clone https://github.com/p3dr0id/DynaBayes.git
cd DynaBayes
pip install -e .
```

To run the development test suite:

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## Quick example

The example below uses the two-oscillator model associated with the tutorial benchmark, with a time-varying frequency and a time-varying direct phase coefficient.

```python
import numpy as np
import dynabayes as db

# Time-dependent model parameters
omega = [
    lambda t: 2 - 0.5 * np.sin(2 * np.pi * 0.00151 * t),
    db.const(4.53),
]

A = [
    [
        db.const(0.8),
        lambda t: 0.8 - 0.3 * np.sin(2 * np.pi * 0.0012 * t),
    ],
    [db.const(0.0), db.const(0.6)],
]

B = [
    [db.const(0.0), db.const(0.0)],
    [db.const(0.0), db.const(0.0)],
]

# Simulate synthetic phase data
phi, true_funcs, t = db.simulate_model(
    omega,
    A,
    B,
    E=[0.03, 0.01],
    t_max=2000,
    dt=0.01,
)

# Sequential dynamic Bayesian inference
result = db.run_inference(
    phi,
    dt=0.01,
    t=t,
    window_seconds=40.0,
    pr=0.20,
)

# Parameter evolution with posterior uncertainty
db.plot_parameters(
    result,
    true_funcs=true_funcs,
    uncertainty="95",
)

# Statistical summary across all windows
summary = db.show_summary(
    result,
    true_funcs=true_funcs,
)

# Inferred noise covariance and normalized cross-noise correlation
db.plot_noise(result)
```

The returned `InferenceResult` contains, among other fields:

```python
result.params
result.time
result.sigma
result.covariance
result.E
result.iterations
result.criterion
result.pr
result.p_internal
```

For a window duration `T_w`, the propagation parameter used internally is

```math
p_{\mathrm{internal}} = T_w p_r
```

For example, `window_seconds=40` and `pr=0.20` correspond to

```math
p_{\mathrm{internal}} = 8
```

---

## Inference from external phase data

If phase time series have already been extracted from empirical signals, they can be supplied directly:

```python
# phi shape: (n_oscillators, n_time_points)
result = db.run_inference(
    phi,
    dt=0.1,
    t=your_time_vector,
    window_seconds=40.0,
    pr=0.20,
)

db.plot_parameters(result, uncertainty="sd")
db.plot_noise(result)
summary = db.show_summary(result)
```

Sequential posterior propagation currently uses consecutive **non-overlapping windows**. This avoids repeatedly propagating information from strongly overlapping observations as though those observations were independent new data.

---

## Backward compatibility

The previous two-value unpacking remains temporarily available:

```python
params, centers = db.run_inference(...)
```

The old `pw` argument is accepted as a deprecated alias for `pr` and emits a `DeprecationWarning`.

The old `E_true` inference argument is also accepted temporarily for compatibility, but it is ignored because the full noise covariance matrix is now inferred internally.

New code should use the `InferenceResult` interface.

---

## Networks with 3+ oscillators

The inference code is written for a general number of phase oscillators. For each equation, the active basis contains

```math
1,\qquad \sin(\phi_j),\qquad \sin(\phi_j-\phi_i),\quad j\ne i
```

The package preserves the full historical `[omega_i, A_i1, ..., A_iN, B_i1, ..., B_iN]` output layout, explicitly marking `B_ii` as structural zeros.

---

## Development and continuous integration

Tests are executed automatically through GitHub Actions on supported development Python versions. The suite contains both API-level regression tests and independent synthetic scientific benchmarks.

Before merging changes to the inference core, all CI checks should pass.

---

## License

MIT License.

The MIT license covers DynaBayes source code. Scientific methods implemented by the package should be cited through their original methodological references.
