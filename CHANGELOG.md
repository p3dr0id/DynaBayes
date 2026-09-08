# Changelog

All notable changes to DynaBayes will be documented in this file.

The format follows the spirit of [Keep a Changelog](https://keepachangelog.com/), and version numbers follow semantic versioning.

## [Unreleased]

### Changed

- Reworked the sequential Dynamic Bayesian Inference core following the mathematical formulation described by Stankovski et al. (2014).
- Switched to midpoint phase discretization for the inference update.
- Kept the prior mean and prior concentration fixed during each internal parameter/noise iteration.
- Changed posterior-to-prior propagation to covariance-based diffusion between consecutive non-overlapping windows.
- Replaced the public propagation name `pw` with `pr`; the old name is temporarily retained as a deprecated alias.
- Defined the internal propagation scale as `p_internal = window_duration * pr`.
- Changed inference from per-oscillator scalar noise estimates to a jointly inferred full noise covariance matrix `E`.
- Removed structurally unidentifiable `B_ii sin(phi_i - phi_i)` terms from the inverse problem while preserving them as explicit structural zeros in the historical output layout.
- `run_inference()` now returns an `InferenceResult` containing parameters, time centers, posterior standard deviations, posterior covariance, inferred noise covariance, convergence diagnostics, and propagation metadata.
- `show_summary()` now summarizes all inference windows and includes posterior uncertainty diagnostics when an `InferenceResult` is supplied.
- `plot_parameters()` now supports posterior uncertainty bands and marks the zero reference level.

### Added

- `InferenceResult` as the structured public result object for inference.
- `plot_noise()` for diagonal noise variances, off-diagonal noise covariances, and normalized cross-noise correlations.
- Automated API and inference regression tests using `pytest`.
- Independent synthetic scientific benchmarks for time-varying parameter recovery and coupling passage through zero.
- GitHub Actions continuous integration on Python 3.11 and 3.13.
- `requirements-dev.txt` for development/test dependencies.
- Explicit methodology/licensing documentation distinguishing the published DBI methodology from independent numerical cross-validation against MODA.

### Deprecated

- `pw` as a public argument to `run_inference()`; use `pr` instead.
- `E_true` as an inference argument. It is accepted temporarily but ignored because `E` is now inferred internally.

### Compatibility

- Historical two-value unpacking remains available temporarily:

  ```python
  params, centers = run_inference(...)
  ```

- The historical full parameter layout is preserved, including structural `B_ii = 0` entries.

### Validation

- The revised inference core is tested against independently generated synthetic systems with known time-dependent parameters and known noise covariance.
- Numerical behaviour was cross-validated against the independently developed MODA reference implementation. No MODA source code is incorporated into DynaBayes.

## [0.1.0]

- Initial public release.
