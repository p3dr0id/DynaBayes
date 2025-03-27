import numpy as np

def const(c):
    return lambda t: c

def base_funcs_from_AB(n_osc):
    base_funcs_all = []
    for i in range(n_osc):
        funcs_i = [lambda phi: np.ones(phi.shape[1])]
        funcs_i += [lambda phi, j=j: np.sin(phi[j]) for j in range(n_osc)]
        funcs_i += [lambda phi, i=i, j=j: np.sin(phi[j] - phi[i]) for j in range(n_osc)]
        base_funcs_all.append(funcs_i)
    return base_funcs_all

def numerical_derivative(f, phi_window, dt, i, y):
    eps = 1e-6
    base_val = f(phi_window[:, :-1])
    phi_eps = np.copy(phi_window[:, :-1])
    phi_eps[i] += eps
    base_eps = f(phi_eps)
    return ((base_eps - base_val) / eps) * y

def gerar_true_funcs_from_matrices(omega, A, B):
    N = len(omega)
    true_funcs = []
    for i in range(N):
        row = [omega[i]]
        for j in range(N):
            row.append(A[i][j])
        for j in range(N):
            row.append(B[i][j])
        true_funcs.append(row)
    return true_funcs

def generate_param_names(n_osc):
    names = []
    for i in range(n_osc):
        row = [f'ω{i+1}']  # Intrinsic frequency
        for j in range(n_osc):
            row.append(f'A{i+1}{j+1}')
        for j in range(n_osc):
            row.append(f'B{i+1}{j+1}')
        names.append(row)
    return names
