import numpy as np
import matplotlib.pyplot as plt
import dynabayes as db

# Example: Inference on 2 coupled phase oscillators

omega = [lambda t: 2 - 0.5 * np.sin(2 * np.pi * 0.00151 * t), db.const(4.53)]
A = [
    [db.const(0.8), lambda t: 0.8 - 0.3 * np.sin(2 * np.pi * 0.0012 * t)],
    [db.const(0.0), db.const(0.6)]
]
B = [
    [db.const(0.0), db.const(0.0)],
    [db.const(0.0), db.const(0.0)]
]

# Simulate system
phi, true_funcs, t = db.simulate_model(omega, A, B, E=[0.03, 0.01], t_max=2000, dt=0.01)

# Run inference
params, centers = db.run_inference(phi, dt=0.01, E_true=[0.03, 0.01], pw=0.2, t=t)

# Plot results
db.plot_parameters(params, true_funcs, centers)
plt.savefig("params_inference.png")
db.show_summary(params, true_funcs, centers)