import numpy as np

def const(c):
    return lambda t: c

def create_model(phi, t, omega, A, B):
    N = len(phi)
    return np.array([
        omega[i](t) +
        sum(A[i][j](t) * np.sin(phi[j]) for j in range(N)) +
        sum(B[i][j](t) * np.sin(phi[j] - phi[i]) for j in range(N))
        for i in range(N)
    ])