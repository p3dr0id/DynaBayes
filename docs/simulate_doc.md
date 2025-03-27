"""Synthetic data generation for oscillator models."""

def simulate_model(omega, A, B, E, t_max, dt):
    """
    Simulates phase evolution with noise over time.

    Args:
        omega, A, B: model parameters
        E (list): noise level per oscillator
        t_max (float): total simulation time
        dt (float): time step

    Returns:
        phi (ndarray): phase matrix
        true_funcs (list): functions used
        t (ndarray): time vector
    """