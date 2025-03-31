import matplotlib.pyplot as plt
import pandas as pd
from .utils import generate_param_names

def plot_parameters(param_seq, true_funcs, time_centers):
    n_osc, n_params = param_seq.shape[1], param_seq.shape[2]
    names = generate_param_names(n_osc)
    total = n_osc * n_params
    fig, axs = plt.subplots(total, 1, figsize=(10, 2.5 * total), squeeze=False)
    axs = axs.flatten()
    
    for i in range(n_osc):
        for j in range(n_params):
            idx = i * n_params + j
            axs[idx].set_ylabel(f'{names[i][j]}')

            if true_funcs:
                true_vals = [true_funcs[i][j](t) for t in time_centers]
                axs[idx].plot(time_centers, true_vals, 'k-', label='True')

            axs[idx].plot(time_centers, param_seq[:, i, j], 'r--', label='Inferred')
            axs[idx].legend()
            axs[idx].grid()
    
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

def show_summary(param_seq, true_funcs, time_centers):
    last_params = param_seq[-1]
    n_osc, n_params = last_params.shape
    names = generate_param_names(n_osc)

    df = pd.DataFrame(columns=['Oscillator', 'Parameter', 'True Value', 'Inferred'])

    for i in range(n_osc):
        for j in range(n_params):
            name = names[i][j]
            if true_funcs:
                true_val = round(true_funcs[i][j](time_centers[-1]), 4)
            else:
                true_val = 'N/A'
            inf_val = round(last_params[i, j], 4)
            df.loc[len(df)] = [f'{i+1}', name, true_val, inf_val]

    print(df)
