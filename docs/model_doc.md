"""Model formulation for coupled phase oscillators."""

def const(c):
    """
    Returns a constant function over time.
    
    Args:
        c (float): constant value
        
    Returns:
        function: function(t) returning c
    """

def create_model(phi, t, omega, A, B):
    """
    Computes dphi/dt using the general oscillator model.

    Args:
        phi (ndarray): phases of oscillators at time t
        t (float): current time
        omega (list): list of functions ω_i(t)
        A (list): matrix A_ij(t) multiplying sin(φ_j)
        B (list): matrix B_ij(t) multiplying sin(φ_j - φ_i)

    Returns:
        ndarray: time derivatives dphi/dt
    """