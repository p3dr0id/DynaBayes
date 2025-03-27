import numpy as np
import matplotlib.pyplot as plt
import bayesinf as bi

# Example: Inference on 2 coupled phase oscillators

omega = [lambda t: 2 - 0.5 * np.sin(2 * np.pi * 0.00151 * t), bi.const(4.53)]
A = [
    [bi.const(0.8), lambda t: 0.8 - 0.3 * np.sin(2 * np.pi * 0.0012 * t)],
    [bi.const(0.0), bi.const(0.6)]
]
B = [
    [bi.const(0.0), bi.const(0.0)],
    [bi.const(0.0), bi.const(0.0)]
]

# Simulate system
phi, true_funcs, t = bi.simulate_model(omega, A, B, E=[0.03, 0.01], t_max=2000, dt=0.01)

# Run inference
params, centers = bi.run_inference(phi, dt=0.01, E_true=[0.03, 0.01], pw=0.2, t=t)

# Plot results
bi.plot_parameters(params, true_funcs, centers)
plt.savefig("params_inference.png")
bi.show_summary(params, true_funcs, centers)