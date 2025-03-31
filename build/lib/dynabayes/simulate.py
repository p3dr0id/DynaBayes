import numpy as np
from .model import create_model
from .utils import base_funcs_from_AB, gerar_true_funcs_from_matrices

def simulate_model(omega, A, B, E, t_max, dt):
    t = np.arange(0, t_max, dt)
    n_osc = len(omega)
    phi = np.zeros((n_osc, len(t)))
    phi[:, 0] = np.random.rand(n_osc) * 2 * np.pi
    for i in range(1, len(t)):
        dphi = create_model(phi[:, i-1], t[i-1], omega, A, B)
        noise = np.sqrt(dt) * np.random.randn(n_osc) * np.sqrt(E)
        phi[:, i] = phi[:, i-1] + dt * dphi + noise
    true_funcs = gerar_true_funcs_from_matrices(omega, A, B)
    return phi, true_funcs, t