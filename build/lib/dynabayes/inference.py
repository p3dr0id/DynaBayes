import numpy as np
from .utils import numerical_derivative, base_funcs_from_AB
from .model import create_model

def run_inference(phi, dt, E_true, pw, t=None, window_size=None, step_size=None):
    if window_size is None:
        window_size = int(40 / dt)
    if step_size is None:
        step_size = int(1 / dt)
    indices = np.arange(0, phi.shape[1] - window_size, step_size)
    time_centers = t[indices + window_size // 2] if t is not None else np.arange(len(indices))
    base_funcs_all = base_funcs_from_AB(phi.shape[0])
    param_seq = []
    for i in indices:
        phi_win = phi[:, i:i + window_size]
        params = bayesian_inference_general(phi_win, dt, E_true, pw, base_funcs_all)
        param_seq.append(params)
    return np.array(param_seq), time_centers

def bayesian_inference_general(phi_window, dt, E_init, pw, base_funcs_all):
    N = phi_window.shape[1] - 1
    dot_phi = (phi_window[:, 1:] - phi_window[:, :-1]) / dt
    n_osc = phi_window.shape[0]
    results = []

    for i in range(n_osc):
        y = dot_phi[i]
        base_funcs = base_funcs_all[i]
        P = np.vstack([f(phi_window[:, :-1]) for f in base_funcs])
        dP_dt = np.vstack([numerical_derivative(f, phi_window, dt, i, y) for f in base_funcs])
        v_term = np.sum(dP_dt, axis=1)

        Xi_prior = np.eye(len(base_funcs)) * 1e-6
        c_prior = np.zeros(len(base_funcs))
        E_scalar = E_init[i]

        for _ in range(100):
            Xi = Xi_prior + (dt / E_scalar) * (P @ P.T)
            r = Xi_prior @ c_prior + (dt / E_scalar) * (P @ y) - (dt / 2) * v_term
            c_new = np.linalg.solve(Xi, r)
            residuals = y - P.T @ c_new
            E_scalar_new = (dt / N) * np.sum(residuals**2)

            if np.linalg.norm(c_new - c_prior) < 1e-6:
                break

            Xi_prior = np.linalg.inv(np.linalg.inv(Xi) + pw * np.diag(np.diag(np.linalg.inv(Xi))))
            c_prior = c_new
            E_scalar = E_scalar_new

        results.append(c_new)
    return np.array(results)