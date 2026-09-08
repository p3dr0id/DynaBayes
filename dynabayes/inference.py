from dataclasses import dataclass
import warnings

import numpy as np


@dataclass
class InferenceResult:
    """Container returned by :func:`run_inference`.

    ``params`` keeps the historical DynaBayes layout
    ``(n_windows, n_oscillators, 1 + 2*n_oscillators)``. Terms
    ``B_ii * sin(phi_i - phi_i)`` are structural zeros and are stored as zero.

    For backward compatibility, the result can still be unpacked as
    ``params, time = run_inference(...)``.
    """

    params: np.ndarray
    time: np.ndarray
    sigma: np.ndarray
    covariance: np.ndarray
    E: np.ndarray
    iterations: np.ndarray
    criterion: np.ndarray
    window_size: int
    step_size: int
    dt: float
    pr: float
    p_internal: float
    structural_zero_mask: np.ndarray
    active_parameter_indices: tuple

    def __iter__(self):
        # Backward-compatible two-value unpacking.
        yield self.params
        yield self.time

    def __getitem__(self, key):
        if isinstance(key, str):
            aliases = {
                "cov": "covariance",
                "cov_internal": "covariance",
            }
            return getattr(self, aliases.get(key, key))
        raise TypeError("InferenceResult indices must be string field names.")

    def keys(self):
        return (
            "params",
            "time",
            "sigma",
            "covariance",
            "E",
            "iterations",
            "criterion",
            "window_size",
            "step_size",
            "dt",
            "pr",
            "p_internal",
            "structural_zero_mask",
            "active_parameter_indices",
        )

    def as_dict(self):
        return {key: self[key] for key in self.keys()}


def _validate_phase_array(phi):
    phi = np.asarray(phi, dtype=float)

    if phi.ndim != 2:
        raise ValueError(
            "phi must be a 2D array with shape "
            "(n_oscillators, n_time_points)."
        )

    if phi.shape[0] < 1 or phi.shape[1] < 3:
        raise ValueError("phi does not contain enough phase samples.")

    if not np.all(np.isfinite(phi)):
        raise ValueError("phi contains NaN or infinite values.")

    return phi


def _active_parameter_indices(n_osc, equation):
    """Indices of the full AB parameter row that are actually inferred.

    Full historical layout for equation i:
        [omega_i, A_i1, ..., A_iN, B_i1, ..., B_iN]

    B_ii is omitted from inference because
        sin(phi_i - phi_i) == 0
    identically.
    """

    indices = [0]
    indices.extend(range(1, 1 + n_osc))
    indices.extend(
        1 + n_osc + j
        for j in range(n_osc)
        if j != equation
    )
    return tuple(indices)


def _basis_and_divergence(phi_mid, equation):
    """Evaluate active basis functions and dPhi/dphi_i at midpoints."""

    n_osc, n_samples = phi_mid.shape

    basis = [np.ones(n_samples)]
    divergence = [np.zeros(n_samples)]

    # A_ij sin(phi_j)
    for j in range(n_osc):
        basis.append(np.sin(phi_mid[j]))
        if j == equation:
            divergence.append(np.cos(phi_mid[j]))
        else:
            divergence.append(np.zeros(n_samples))

    # B_ij sin(phi_j - phi_i), excluding j == i (structural zero)
    for j in range(n_osc):
        if j == equation:
            continue

        delta = phi_mid[j] - phi_mid[equation]
        basis.append(np.sin(delta))
        divergence.append(-np.cos(delta))

    return np.vstack(basis), np.vstack(divergence)


def _regularize_positive_definite(matrix, floor=1e-12):
    """Apply only the minimal diagonal jitter needed for inversion."""

    matrix = 0.5 * (matrix + matrix.T)
    eig_min = np.min(np.linalg.eigvalsh(matrix))

    if eig_min <= floor:
        matrix = matrix + np.eye(matrix.shape[0]) * (
            floor - eig_min + floor
        )

    return matrix


def _infer_one_window(
    phi_window,
    dt,
    C_prior,
    Xi_prior,
    max_iter=500,
    tol=1e-5,
):
    """Dynamic Bayesian update for one non-overlapping time window.

    The prior mean and prior concentration are fixed throughout the internal
    E <-> C iterations. Only after convergence is the posterior propagated to
    the next window.
    """

    phi_mid = 0.5 * (
        phi_window[:, 1:] + phi_window[:, :-1]
    )
    phi_dot = (
        phi_window[:, 1:] - phi_window[:, :-1]
    ) / dt

    n_osc = phi_window.shape[0]
    n_increments = phi_dot.shape[1]
    n_active = 2 * n_osc

    basis = []
    divergence_sum = []

    for i in range(n_osc):
        P_i, dP_i = _basis_and_divergence(phi_mid, i)
        basis.append(P_i)
        divergence_sum.append(np.sum(dP_i, axis=1))

    C_prior = np.asarray(C_prior, dtype=float).copy()
    Xi_prior = np.asarray(Xi_prior, dtype=float).copy()

    C_current = C_prior.copy()
    C_old = C_current.copy()

    criterion = np.inf
    Xi_post = None
    E = None

    for iteration in range(1, max_iter + 1):

        residual = np.vstack(
            [
                phi_dot[i] - C_current[i] @ basis[i]
                for i in range(n_osc)
            ]
        )

        # Full noise covariance matrix (Eq. 6 form).
        E = (
            dt / n_increments
        ) * (residual @ residual.T)

        E = _regularize_positive_definite(E)
        E_inv = np.linalg.inv(E)

        # Joint posterior concentration matrix.
        Xi_post = Xi_prior.copy()

        for i in range(n_osc):
            for j in range(n_osc):
                Xi_post[
                    i*n_active:(i+1)*n_active,
                    j*n_active:(j+1)*n_active,
                ] += (
                    dt
                    * E_inv[i, j]
                    * (basis[i] @ basis[j].T)
                )

        # Posterior r vector. The prior is deliberately NOT updated here.
        r = np.zeros((n_osc, n_active))

        for i in range(n_osc):
            prior_term = np.zeros(n_active)

            for j in range(n_osc):
                prior_term += (
                    Xi_prior[
                        i*n_active:(i+1)*n_active,
                        j*n_active:(j+1)*n_active,
                    ]
                    @ C_prior[j]
                )

            weighted_velocity = E_inv[i] @ phi_dot

            r[i] = prior_term + dt * (
                basis[i] @ weighted_velocity
                - 0.5 * divergence_sum[i]
            )

        C_vector = np.linalg.solve(
            Xi_post,
            np.concatenate(r),
        )
        C_new = C_vector.reshape(n_osc, n_active)

        criterion = np.sum(
            (C_old - C_new) ** 2
            / (C_new ** 2 + 1e-12)
        )

        C_current = C_new

        if criterion < tol:
            break

        C_old = C_new.copy()

    Sigma_post = np.linalg.inv(Xi_post)

    return {
        "C": C_current,
        "Xi": Xi_post,
        "Sigma": Sigma_post,
        "E": E,
        "iterations": iteration,
        "criterion": criterion,
    }


def _expand_active_parameters(C_active, Sigma_active):
    """Insert structural B_ii zeros into the historical full AB layout."""

    n_osc, n_active = C_active.shape
    n_full = 1 + 2 * n_osc

    params = np.zeros((n_osc, n_full))
    sigma = np.zeros((n_osc, n_full))
    structural_zero_mask = np.zeros((n_osc, n_full), dtype=bool)

    sd_active = np.sqrt(
        np.clip(np.diag(Sigma_active), 0.0, None)
    ).reshape(n_osc, n_active)

    active_indices = []

    for i in range(n_osc):
        indices = _active_parameter_indices(n_osc, i)
        active_indices.append(indices)

        params[i, list(indices)] = C_active[i]
        sigma[i, list(indices)] = sd_active[i]

        structural_index = 1 + n_osc + i
        structural_zero_mask[i, structural_index] = True

    return (
        params,
        sigma,
        structural_zero_mask,
        tuple(active_indices),
    )


def run_inference(
    phi,
    dt,
    E_true=None,
    pw=None,
    t=None,
    window_size=None,
    step_size=None,
    *,
    pr=None,
    window_seconds=None,
    max_iter=500,
    tol=1e-5,
):
    """Infer time-evolving phase dynamics using sequential Bayesian updates.

    Parameters
    ----------
    phi : array_like, shape (n_oscillators, n_time_points)
        Unwrapped phase time series.
    dt : float
        Sampling interval in seconds.
    E_true : optional
        Deprecated compatibility argument. Noise is now inferred internally as
        a full covariance matrix and this value is ignored.
    pw : float, optional
        Deprecated alias for ``pr``.
    t : array_like, optional
        Time vector aligned with ``phi``.
    window_size : int, optional
        Window length in samples. Prefer ``window_seconds`` in new code.
    step_size : int, optional
        Step in samples. Sequential posterior propagation currently requires
        non-overlapping consecutive windows, therefore it must equal
        ``window_size``.
    pr : float, optional
        Propagation parameter exposed by the DynaBayes interface.
        The internal diffusion scale is
        ``p_internal = window_duration * pr``.
    window_seconds : float, optional
        Window duration in seconds. Default is 40 s.
    max_iter : int, optional
        Maximum number of internal E <-> C iterations per window.
    tol : float, optional
        Relative convergence criterion.

    Returns
    -------
    InferenceResult
        Contains inferred parameters, marginal posterior standard deviations,
        joint posterior covariance, full noise covariance, convergence
        diagnostics, time centers, and propagation metadata.

    Notes
    -----
    The inference uses:
    * midpoint discretization;
    * a full inferred noise covariance matrix;
    * a fixed prior during the internal E <-> C iterations;
    * posterior-to-prior propagation only between windows;
    * covariance-based diffusion between windows.

    The terms ``B_ii sin(phi_i - phi_i)`` are structural zeros and are not
    included in the linear solve.
    """

    phi = _validate_phase_array(phi)

    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive finite number.")

    if E_true is not None:
        warnings.warn(
            "E_true is deprecated and ignored. DynaBayes now infers the "
            "full noise covariance matrix E inside each window.",
            DeprecationWarning,
            stacklevel=2,
        )

    if pr is not None and pw is not None:
        raise ValueError("Use either pr or deprecated pw, not both.")

    if pr is None:
        if pw is not None:
            warnings.warn(
                "pw is deprecated; use pr instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            pr = float(pw)
        else:
            pr = 0.20

    pr = float(pr)
    if not np.isfinite(pr) or pr < 0:
        raise ValueError("pr must be a finite non-negative number.")

    if window_seconds is not None and window_size is not None:
        raise ValueError(
            "Specify only one of window_seconds or window_size."
        )

    if window_seconds is None and window_size is None:
        window_seconds = 40.0

    if window_seconds is not None:
        window_seconds = float(window_seconds)
        if not np.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        window_size = int(round(window_seconds / dt))
    else:
        window_size = int(window_size)
        window_seconds = window_size * dt

    if window_size < 3:
        raise ValueError("Inference window is too short.")

    if step_size is None:
        step_size = window_size

    step_size = int(step_size)

    if step_size != window_size:
        raise ValueError(
            "Sequential posterior propagation currently requires "
            "non-overlapping consecutive windows, so step_size must equal "
            "window_size."
        )

    if phi.shape[1] < window_size:
        raise ValueError(
            "phi is shorter than one complete inference window."
        )

    if t is not None:
        t = np.asarray(t, dtype=float).squeeze()
        if t.ndim != 1 or len(t) != phi.shape[1]:
            raise ValueError(
                "t must be a 1D vector with the same number of samples as phi."
            )
        if not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
            raise ValueError("t must be finite and strictly increasing.")

    starts = np.arange(
        0,
        phi.shape[1] - window_size + 1,
        step_size,
    )

    n_osc = phi.shape[0]
    n_active = 2 * n_osc

    C_prior = np.zeros((n_osc, n_active))
    Xi_prior = np.zeros(
        (n_osc*n_active, n_osc*n_active)
    )

    p_internal = window_seconds * pr

    param_seq = []
    sigma_seq = []
    covariance_seq = []
    noise_seq = []
    iterations_seq = []
    criterion_seq = []
    time_centers = []

    structural_zero_mask = None
    active_indices = None

    for start in starts:

        stop = start + window_size
        result = _infer_one_window(
            phi[:, start:stop],
            dt,
            C_prior,
            Xi_prior,
            max_iter=max_iter,
            tol=tol,
        )

        (
            params_full,
            sigma_full,
            structural_zero_mask,
            active_indices,
        ) = _expand_active_parameters(
            result["C"],
            result["Sigma"],
        )

        param_seq.append(params_full)
        sigma_seq.append(sigma_full)
        covariance_seq.append(result["Sigma"])
        noise_seq.append(result["E"])
        iterations_seq.append(result["iterations"])
        criterion_seq.append(result["criterion"])

        center_index = start + window_size / 2.0

        if t is None:
            time_centers.append(center_index * dt)
        else:
            time_centers.append(
                np.interp(
                    center_index,
                    np.arange(len(t)),
                    t,
                )
            )

        # Posterior -> prior only after internal convergence.
        C_prior = result["C"].copy()

        Sigma_post = result["Sigma"]

        # MODA-compatible covariance diffusion.
        Sigma_diff = np.diag(
            p_internal**2 * np.diag(Sigma_post)
        )
        Sigma_prior = Sigma_post + Sigma_diff
        Xi_prior = np.linalg.inv(Sigma_prior)

    return InferenceResult(
        params=np.asarray(param_seq),
        time=np.asarray(time_centers),
        sigma=np.asarray(sigma_seq),
        covariance=np.asarray(covariance_seq),
        E=np.asarray(noise_seq),
        iterations=np.asarray(iterations_seq),
        criterion=np.asarray(criterion_seq),
        window_size=window_size,
        step_size=step_size,
        dt=dt,
        pr=pr,
        p_internal=p_internal,
        structural_zero_mask=structural_zero_mask,
        active_parameter_indices=active_indices,
    )


# Public alias useful for tests and advanced users.
bayesian_inference_general = _infer_one_window
