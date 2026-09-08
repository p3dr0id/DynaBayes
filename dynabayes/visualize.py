import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .inference import InferenceResult
from .utils import generate_param_names


def _coerce_result(result_or_params, time_centers=None):
    """Normalize new InferenceResult and legacy array inputs."""
    if isinstance(result_or_params, InferenceResult):
        return {
            "params": result_or_params.params,
            "time": result_or_params.time,
            "sigma": result_or_params.sigma,
            "structural_zero_mask": result_or_params.structural_zero_mask,
            "result": result_or_params,
        }

    params = np.asarray(result_or_params, dtype=float)

    if params.ndim != 3:
        raise ValueError(
            "Legacy parameter input must have shape "
            "(n_windows, n_oscillators, n_parameters)."
        )

    if time_centers is None:
        raise ValueError(
            "time_centers is required when passing a legacy parameter array."
        )

    time_centers = np.asarray(time_centers, dtype=float)

    if time_centers.ndim != 1 or len(time_centers) != params.shape[0]:
        raise ValueError(
            "time_centers must be a 1D vector with one value per window."
        )

    return {
        "params": params,
        "time": time_centers,
        "sigma": None,
        "structural_zero_mask": None,
        "result": None,
    }


def plot_parameters(
    result_or_params,
    true_funcs=None,
    time_centers=None,
    *,
    uncertainty=None,
    include_structural=False,
    show=True,
):
    """Plot the temporal evolution of inferred parameters.

    Parameters
    ----------
    result_or_params : InferenceResult or ndarray
        Prefer passing the full ``InferenceResult`` returned by
        :func:`dynabayes.run_inference`. Legacy parameter arrays remain
        supported when ``time_centers`` is also supplied.
    true_funcs : sequence, optional
        Optional reference functions in the historical DynaBayes layout.
    time_centers : array_like, optional
        Required only for legacy array input.
    uncertainty : {None, "sd", "95"}, optional
        Posterior uncertainty band. Available only with ``InferenceResult``.
        ``"sd"`` plots +/- 1 posterior standard deviation and ``"95"`` plots
        +/- 1.96 posterior standard deviations.
    include_structural : bool, optional
        If False, structural B_ii zeros are omitted from the figure.
    show : bool, optional
        Call ``plt.show()`` before returning.

    Returns
    -------
    fig, axes
        Matplotlib figure and the active axes array.
    """
    data = _coerce_result(result_or_params, time_centers=time_centers)

    params = data["params"]
    times = data["time"]
    sigma = data["sigma"]
    structural_mask = data["structural_zero_mask"]

    if uncertainty not in (None, "sd", "95"):
        raise ValueError("uncertainty must be None, 'sd', or '95'.")

    if uncertainty is not None and sigma is None:
        raise ValueError(
            "Posterior uncertainty is available only when an InferenceResult "
            "is passed."
        )

    n_osc, n_params = params.shape[1], params.shape[2]
    names = generate_param_names(n_osc)

    active = []
    for i in range(n_osc):
        for j in range(n_params):
            is_structural = (
                structural_mask is not None
                and bool(structural_mask[i, j])
            )

            if is_structural and not include_structural:
                continue

            active.append((i, j, is_structural))

    total = len(active)
    fig, axs = plt.subplots(
        total,
        1,
        figsize=(10, max(3.0, 2.5 * total)),
        squeeze=False,
        sharex=True,
    )
    axs = axs.flatten()

    band_scale = None
    band_label = None
    if uncertainty == "sd":
        band_scale = 1.0
        band_label = "+/- 1 SD posterior"
    elif uncertainty == "95":
        band_scale = 1.96
        band_label = "95% posterior interval"

    for ax, (i, j, is_structural) in zip(axs, active):
        name = names[i][j]
        values = params[:, i, j]

        if true_funcs is not None:
            true_vals = [true_funcs[i][j](t) for t in times]
            ax.plot(times, true_vals, label="True")

        if is_structural:
            ax.plot(times, values, label="Structural zero")
        else:
            ax.plot(times, values, marker="o", label="Inferred")

        if band_scale is not None and not is_structural:
            sd = sigma[:, i, j]
            ax.fill_between(
                times,
                values - band_scale * sd,
                values + band_scale * sd,
                alpha=0.20,
                label=band_label,
            )

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        ax.legend()

    axs[-1].set_xlabel("Time (s)")
    fig.tight_layout()

    if show:
        plt.show()

    return fig, axs


def show_summary(
    result_or_params,
    true_funcs=None,
    time_centers=None,
    *,
    include_structural=False,
    print_table=True,
):
    """Summarize inferred parameters across all inference windows.

    For ``InferenceResult`` inputs the table includes posterior marginal
    uncertainty and the fraction of windows whose marginal posterior intervals
    exclude zero. Legacy array inputs remain supported, but posterior columns
    are unavailable.

    Returns
    -------
    pandas.DataFrame
        One row per displayed parameter.
    """
    data = _coerce_result(result_or_params, time_centers=time_centers)

    params = data["params"]
    times = data["time"]
    sigma = data["sigma"]
    structural_mask = data["structural_zero_mask"]

    n_osc, n_params = params.shape[1], params.shape[2]
    names = generate_param_names(n_osc)

    rows = []

    for i in range(n_osc):
        for j in range(n_params):
            is_structural = (
                structural_mask is not None
                and bool(structural_mask[i, j])
            )

            if is_structural and not include_structural:
                continue

            values = params[:, i, j]

            row = {
                "Oscillator": i + 1,
                "Parameter": names[i][j],
                "Status": "structural zero" if is_structural else "inferred",
                "Mean": np.mean(values),
                "Median": np.median(values),
                "Temporal SD": np.std(values, ddof=1) if len(values) > 1 else 0.0,
                "Q25": np.quantile(values, 0.25),
                "Q75": np.quantile(values, 0.75),
                "Min": np.min(values),
                "Max": np.max(values),
            }

            if true_funcs is not None:
                row["True value (last window)"] = true_funcs[i][j](times[-1])

            if sigma is not None:
                sd = sigma[:, i, j]

                if is_structural:
                    row["Posterior SD median"] = 0.0
                    row["|c| > 1 SD (%)"] = np.nan
                    row["95% interval excludes zero (%)"] = np.nan
                else:
                    row["Posterior SD median"] = np.median(sd)
                    row["|c| > 1 SD (%)"] = 100.0 * np.mean(
                        np.abs(values) > sd
                    )
                    row["95% interval excludes zero (%)"] = 100.0 * np.mean(
                        np.abs(values) > 1.96 * sd
                    )

            rows.append(row)

    df = pd.DataFrame(rows)

    if print_table:
        print(df.to_string(index=False))

    return df


def plot_noise(result, *, correlation=True, show=True):
    """Plot the inferred full noise covariance matrix through time.

    Parameters
    ----------
    result : InferenceResult
        Result returned by :func:`dynabayes.run_inference`.
    correlation : bool, optional
        Also plot normalized off-diagonal noise correlations
        ``rho_ij = E_ij / sqrt(E_ii E_jj)``.
    show : bool, optional
        Call ``plt.show()`` before returning.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes array.
    """
    if not isinstance(result, InferenceResult):
        raise TypeError("plot_noise requires an InferenceResult.")

    E = np.asarray(result.E, dtype=float)
    times = np.asarray(result.time, dtype=float)
    n_osc = E.shape[1]

    n_panels = 2 if n_osc > 1 else 1
    if correlation and n_osc > 1:
        n_panels += 1

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(10, 3.2 * n_panels),
        sharex=True,
        squeeze=False,
    )
    axes = axes.flatten()

    panel = 0

    for i in range(n_osc):
        axes[panel].plot(
            times,
            E[:, i, i],
            marker="o",
            label=f"E{i+1}{i+1}",
        )

    axes[panel].set_ylabel("Noise variance")
    axes[panel].set_title("Inferred diagonal noise terms")
    axes[panel].grid(alpha=0.3)
    axes[panel].legend()
    panel += 1

    if n_osc > 1:
        for i in range(n_osc):
            for j in range(i + 1, n_osc):
                axes[panel].plot(
                    times,
                    E[:, i, j],
                    marker="o",
                    label=f"E{i+1}{j+1}",
                )

        axes[panel].axhline(0.0, linestyle="--", linewidth=1)
        axes[panel].set_ylabel("Noise covariance")
        axes[panel].set_title("Inferred off-diagonal noise terms")
        axes[panel].grid(alpha=0.3)
        axes[panel].legend()
        panel += 1

    if correlation and n_osc > 1:
        for i in range(n_osc):
            for j in range(i + 1, n_osc):
                rho = E[:, i, j] / np.sqrt(E[:, i, i] * E[:, j, j])
                axes[panel].plot(
                    times,
                    rho,
                    marker="o",
                    label=f"rho{i+1}{j+1}",
                )

        axes[panel].axhline(0.0, linestyle="--", linewidth=1)
        axes[panel].set_ylabel("Noise correlation")
        axes[panel].set_title("Normalized off-diagonal noise correlations")
        axes[panel].grid(alpha=0.3)
        axes[panel].legend()

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes
