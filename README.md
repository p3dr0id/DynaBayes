# DynaBayes

**Dynamic Bayesian Inference for Coupled Phase Oscillators**

This Python package implements a method for dynamic Bayesian inference of parameters in networks of coupled phase oscillators. It allows simulation and inference using a general model of phase interaction, including both direct phase coupling and coupling through phase differences.

---

## 📘 Model Formulation

The general form of the system implemented is:

$$
\frac{d\phi_i}{dt} = \omega_i(t) + \sum_{j=1}^{N} A_{ij}(t)\sin(\phi_j) + \sum_{j=1}^{N} B_{ij}(t)\sin(\phi_j - \phi_i) + \sqrt{E_{ii}} \xi_i(t)
$$

Where:

- $\phi_i$ is the phase of the $i$-th oscillator.
- $\omega_i(t)$ is the intrinsic frequency (can be time-dependent).
- $A_{ij}(t)$ controls influence from $\sin(\phi_j)$.
- $B_{ij}(t)$ controls influence from phase differences $\sin(\phi_j - \phi_i)$.

---

## Reference

This package is based on the dynamic Bayesian inference approach described in:

> Stankovski, T., Ticcinelli, V., McClintock, P.V.E. & Stefanovska, A. (2014). A tutorial on time-evolving dynamical Bayesian inference. Eur. Phys. J. Special Topics 222, 2467–2485. https://doi.org/10.1140/epjst/e2014-02286-7



## 🔧 Installation

```bash
git clone https://github.com/p3dr0id/DynaBayes.git
cd DynaBayes
pip install -e .
```

---

## ⚡ Quick Example

The model used in the example below replicates Equation (10) from the paper

The system is:

$$
\frac{d\phi_1}{dt} = \omega_1(t) + a_1 \sin(\phi_1) + a_3(t) \sin(\phi_2) + \sqrt{E_{11}}\xi_1(t)
$$

$$
\frac{d\phi_2}{dt} = \omega_2 + a_2 \sin(\phi_1) + a_4 \sin(\phi_2) + \sqrt{E_{22}}\xi_2(t)
$$


Where:

- $\omega_1(t) = 2 - 0.5 \sin(2\pi \cdot 0.00151 \cdot t)$
- $a_3(t) = 0.8 - 0.3 \sin(2\pi \cdot 0.0012 \cdot t)$
- $\omega_2 = 4.53$, $a_1 = 0.8$, $a_2 = 0.0$, $a_4 = 0.6$
- $E_{11} = 0.03$, $E_{22} = 0.01$

---

```python
import numpy as np
import dynabayes as db

# Define time-dependent parameters
omega = [lambda t: 2 - 0.5 * np.sin(2 * np.pi * 0.00151 * t), db.const(4.53)]
A = [
    [db.const(0.8), lambda t: 0.8 - 0.3 * np.sin(2 * np.pi * 0.0012 * t)],
    [db.const(0.0), db.const(0.6)]
]
B = [
    [db.const(0.0), db.const(0.0)],
    [db.const(0.0), db.const(0.0)]
]

# Simulate synthetic phase data
phi, true_funcs, t = db.simulate_model(omega, A, B, E=[0.03, 0.01], t_max=2000, dt=0.01)

# Run inference
params, centers = db.run_inference(phi, dt=0.01, E_true=[0.03, 0.01], pw=0.2, t=t)

# Plot parameter evolution
db.plot_parameters(params, true_funcs, centers)

# Show summary dataframe
db.show_summary(params, true_funcs, centers)
```

---

## 📈 Inference from External Phase Data

If you already have phase time series (e.g., from empirical data), you can apply inference directly:

```python
# phi: 2D array of shape (n_oscillators, time_points)
params, centers = db.run_inference(phi, dt=0.01, E_true=[0.02, 0.01], pw=0.1, t=your_time_vector)

# You can still visualize even without a model:
db.plot_parameters(params, true_funcs=None, time_centers=centers)
```

Note: If `true_funcs=None`, only inferred parameters will be plotted.

---

## 🧪 Testing Models with 3+ Oscillators

You can define and simulate networks of 3 or more coupled oscillators by adjusting the size of `omega`, `A`, and `B`. The package automatically handles the dimensionality.

---

## 📄 License

MIT License
