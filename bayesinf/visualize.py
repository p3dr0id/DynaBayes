import matplotlib.pyplot as plt
import pandas as pd

def plot_parameters(param_seq, true_funcs, time_centers):
    n_osc, n_params = param_seq.shape[1], param_seq.shape[2]
    total = n_osc * n_params
    fig, axs = plt.subplots(total, 1, figsize=(10, 2.5 * total), squeeze=False)
    axs = axs.flatten()
    for i in range(n_osc):
        for j in range(n_params):
            idx = i * n_params + j
            if true_funcs:
                true_vals = [true_funcs[i][j](t) for t in time_centers]
                axs[idx].plot(time_centers, true_vals, 'k-', label='True')
            axs[idx].plot(time_centers, param_seq[:, i, j], 'r--', label='Inferred')
            axs[idx].set_ylabel(f'Osc {i+1} - Param {j}')
            axs[idx].legend()
            axs[idx].grid()
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

def show_summary(param_seq, true_funcs, time_centers):
    ultimo = param_seq[-1]
    n_osc, n_params = ultimo.shape
    df = pd.DataFrame(columns=['Oscillator', 'Parameter', 'True', 'Inferred'])
    for i in range(n_osc):
        for j in range(n_params):
            name = 'ω' if j == 0 else f'a{j}'
            true_val = true_funcs[i][j](time_centers[-1]) if true_funcs else "-"
            inf_val = ultimo[i, j]
            df.loc[len(df)] = [f'Osc {i+1}', name, true_val, round(inf_val, 3)]
    print(df)