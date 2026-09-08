# Validation of the revised DBI core

This document records the validation steps used before the DynaBayes 0.2.0 release.

## Methodological source

DynaBayes is an independent Python implementation of the time-evolving dynamical Bayesian inference methodology described by Stankovski et al. (2014).

The MODA software was used only as an independent numerical reference during development and cross-validation. No MODA source code is incorporated into DynaBayes.

Reference:

> Stankovski, T., Duggento, A., McClintock, P. V. E., & Stefanovska, A. (2014). A tutorial on time-evolving dynamical Bayesian inference. *European Physical Journal Special Topics*, 223, 2685-2703. https://doi.org/10.1140/epjst/e2014-02286-7

## 1. Automated synthetic benchmarks

The repository contains independent synthetic tests written from the published phase model. They verify:

- recovery of a time-varying intrinsic frequency;
- recovery of a time-varying coupling coefficient;
- tracking of a coupling coefficient through zero;
- recovery of the inferred noise covariance scale;
- structural treatment of B_ii terms;
- stable internal convergence.

These tests are executed automatically with `pytest` by GitHub Actions on Python 3.11 and Python 3.13.

## 2. Cross-validation against an independent reference implementation

During development, the numerical behaviour of the revised inference algorithm was compared with the independently developed MODA implementation. This comparison was used to resolve implementation details such as:

- midpoint discretization;
- keeping the prior fixed during the internal E <-> C iterations;
- full noise covariance inference;
- covariance-based posterior-to-prior propagation;
- the relationship `p_internal = window_duration * pr`.

The DynaBayes source was written independently from the published mathematical formulation and the project-specific reference implementation.

## 3. Empirical S01 equivalence test

The packaged DynaBayes implementation was compared directly with the already validated self-contained reference implementation used in the S01 cardiorespiratory notebook, using the same phase data and inference configuration.

Configuration:

- two oscillators;
- sampling interval: 0.1 s;
- inference window: 40 s;
- non-overlapping windows;
- `pr = 0.20`;
- `p_internal = 8`;
- 12 complete inference windows.

Maximum absolute differences between the packaged implementation and the self-contained reference were:

| Quantity | Maximum absolute difference |
| --- | ---: |
| Inferred parameters | 2.6645352591003757e-15 |
| Posterior standard deviations | 1.3877787807814457e-17 |
| Noise covariance matrix E | 2.7755575615628914e-17 |
| Convergence criterion | 4.3495141841508322e-18 |

The internal iteration sequence was exactly identical in both implementations:

```text
[4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]
```

Therefore the DBI core is numerically equivalent to the validated reference implementation to substantially better than 1e-12.

### Time-vector offset observed during the validation notebook

A constant 5 ms time offset was observed only because the temporary validation cell reconstructed the initial time from a value printed to two decimal places (`10.92 s`) rather than using the original full-precision time vector.

The offset was constant in all windows and does not affect inferred parameters, posterior uncertainty, noise covariance, or convergence. The original reference time origin corresponds to `10.925 s`.

The public API should therefore be supplied with the original time vector when absolute time coordinates matter.

## Validation conclusion

The revised DynaBayes inference core passed:

1. automated unit/regression tests;
2. independent synthetic scientific benchmarks;
3. numerical cross-validation against an independent reference implementation;
4. empirical numerical-equivalence testing against the validated S01 reference notebook.

This validation record supports the 0.2.0 inference-core release.