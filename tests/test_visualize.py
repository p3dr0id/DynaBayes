import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from dynabayes import plot_noise, plot_parameters, run_inference, show_summary


def _small_result():
    dt = 0.1
    t = np.arange(0.0, 80.0, dt)

    phi1 = 2 * np.pi * 1.2 * t + 0.15 * np.sin(2 * np.pi * 0.18 * t)
    phi2 = 2 * np.pi * 0.18 * t + 0.08 * np.sin(2 * np.pi * 0.04 * t)

    phi = np.vstack([phi1, phi2])

    return run_inference(
        phi,
        dt=dt,
        t=t,
        window_seconds=20.0,
        pr=0.20,
    )


def test_show_summary_returns_dataframe_with_posterior_columns():
    result = _small_result()

    df = show_summary(
        result,
        print_table=False,
    )

    assert "Parameter" in df.columns
    assert "Median" in df.columns
    assert "Posterior SD median" in df.columns
    assert "95% interval excludes zero (%)" in df.columns

    # N=2 has 10 historical parameters, but two B_ii terms are structural zeros
    # and are omitted by default.
    assert len(df) == 8


def test_show_summary_can_include_structural_zeros():
    result = _small_result()

    df = show_summary(
        result,
        include_structural=True,
        print_table=False,
    )

    structural = df[df["Status"] == "structural zero"]
    assert len(df) == 10
    assert len(structural) == 2
    assert np.all(structural["Median"].to_numpy() == 0.0)


def test_plot_parameters_supports_posterior_bands():
    result = _small_result()

    fig, axes = plot_parameters(
        result,
        uncertainty="95",
        show=False,
    )

    # Structural zeros are omitted by default.
    assert len(axes) == 8
    assert fig is not None

    plt.close(fig)


def test_plot_noise_returns_expected_panels_for_two_oscillators():
    result = _small_result()

    fig, axes = plot_noise(
        result,
        correlation=True,
        show=False,
    )

    # diagonal terms, off-diagonal covariance, normalized correlation
    assert len(axes) == 3
    assert fig is not None

    plt.close(fig)
