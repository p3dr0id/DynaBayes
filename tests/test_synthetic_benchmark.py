import numpy as np

from dynabayes import run_inference


DT = 0.1
WINDOW_SECONDS = 40.0
PR = 0.20
E_TRUE = np.array([[0.03, 0.0], [0.0, 0.01]])


def _simulate_known_system(duration=1200.0, seed=0, crossing_zero=False):
    """Simulate two coupled phase oscillators with known time-varying terms.

    The simulation is written independently from DynaBayes and follows the
    model exposed by the package:

        dphi_i/dt = omega_i
                    + sum_j A_ij sin(phi_j)
                    + sum_j B_ij sin(phi_j - phi_i)
                    + noise.

    Noise increments use covariance E_TRUE * dt, which is consistent with the
    E convention estimated by the inference routine.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, DT)
    n = t.size

    phi = np.zeros((2, n), dtype=float)
    phi[:, 0] = [0.2, 1.0]

    chol = np.linalg.cholesky(E_TRUE)

    for k in range(n - 1):
        tk = t[k]
        p1, p2 = phi[:, k]

        omega1 = 1.90 + 0.30 * np.sin(2 * np.pi * tk / 600.0)
        omega2 = 1.10

        A11 = 0.05
        A12 = 0.70 + 0.20 * np.sin(2 * np.pi * tk / 500.0 + 0.3)
        A21 = -0.08
        A22 = 0.03

        if crossing_zero:
            B12 = 0.25 * np.sin(2 * np.pi * tk / 500.0)
        else:
            B12 = -0.12

        B21 = 0.04

        drift = np.array([
            omega1
            + A11 * np.sin(p1)
            + A12 * np.sin(p2)
            + B12 * np.sin(p2 - p1),
            omega2
            + A21 * np.sin(p1)
            + A22 * np.sin(p2)
            + B21 * np.sin(p1 - p2),
        ])

        stochastic_increment = (
            chol @ rng.normal(size=2)
        ) * np.sqrt(DT)

        phi[:, k + 1] = (
            phi[:, k]
            + drift * DT
            + stochastic_increment
        )

    return t, phi


def _truth_at(times, crossing_zero=False):
    times = np.asarray(times)

    omega1 = 1.90 + 0.30 * np.sin(2 * np.pi * times / 600.0)
    A12 = 0.70 + 0.20 * np.sin(2 * np.pi * times / 500.0 + 0.3)

    if crossing_zero:
        B12 = 0.25 * np.sin(2 * np.pi * times / 500.0)
    else:
        B12 = np.full_like(times, -0.12, dtype=float)

    return omega1, A12, B12


def _corr(x, y):
    return np.corrcoef(np.asarray(x), np.asarray(y))[0, 1]


def _rmse(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sqrt(np.mean((x - y) ** 2))


def test_tracks_known_time_varying_frequency_and_coupling():
    t, phi = _simulate_known_system(seed=0, crossing_zero=False)

    result = run_inference(
        phi,
        dt=DT,
        t=t,
        window_seconds=WINDOW_SECONDS,
        pr=PR,
    )

    omega1_true, A12_true, _ = _truth_at(result.time)

    omega1_hat = result.params[:, 0, 0]
    A12_hat = result.params[:, 0, 2]

    assert _corr(omega1_hat, omega1_true) > 0.95
    assert _rmse(omega1_hat, omega1_true) < 0.10

    assert _corr(A12_hat, A12_true) > 0.90
    assert _rmse(A12_hat, A12_true) < 0.12

    # Noise covariance should recover the correct scale and near-zero cross term.
    E_median = np.median(result.E, axis=0)
    assert abs(E_median[0, 0] - E_TRUE[0, 0]) < 0.010
    assert abs(E_median[1, 1] - E_TRUE[1, 1]) < 0.005
    assert abs(E_median[0, 1]) < 0.005

    # The corrected inner loop should converge rapidly.
    assert np.mean(result.iterations) < 10
    assert np.max(result.iterations) < 25


def test_tracks_cross_coupling_through_zero():
    t, phi = _simulate_known_system(seed=1, crossing_zero=True)

    result = run_inference(
        phi,
        dt=DT,
        t=t,
        window_seconds=WINDOW_SECONDS,
        pr=PR,
    )

    _, _, B12_true = _truth_at(
        result.time,
        crossing_zero=True,
    )

    # For equation 1 in the historical N=2 layout:
    # [omega_1, A_11, A_12, B_11, B_12]
    B12_hat = result.params[:, 0, 4]

    assert _corr(B12_hat, B12_true) > 0.85
    assert _rmse(B12_hat, B12_true) < 0.10

    meaningful = np.abs(B12_true) > 0.05
    sign_accuracy = np.mean(
        np.sign(B12_hat[meaningful])
        == np.sign(B12_true[meaningful])
    )

    assert sign_accuracy > 0.85
