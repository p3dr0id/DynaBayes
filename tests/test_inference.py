import numpy as np
import pytest

from dynabayes.inference import InferenceResult, run_inference


def _synthetic_phases(dt=0.1, duration=120.0, seed=1234):
    """Small deterministic-noise phase dataset for API-level regression tests."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, dt)

    f_c = 1.20 + 0.04 * np.sin(2 * np.pi * t / 60.0)
    f_r = 0.18 + 0.015 * np.sin(2 * np.pi * t / 75.0 + 0.4)

    dphi_c = 2 * np.pi * f_c * dt + rng.normal(0.0, 0.015, size=t.size)
    dphi_r = 2 * np.pi * f_r * dt + rng.normal(0.0, 0.008, size=t.size)

    phi_c = np.cumsum(dphi_c)
    phi_r = np.cumsum(dphi_r)

    return t, np.vstack([phi_c, phi_r])


def test_inference_result_contract_and_shapes():
    dt = 0.1
    t, phi = _synthetic_phases(dt=dt)

    result = run_inference(
        phi,
        dt=dt,
        t=t,
        window_seconds=20.0,
        pr=0.20,
    )

    assert isinstance(result, InferenceResult)
    assert result.params.ndim == 3
    assert result.params.shape[1:] == (2, 5)
    assert result.sigma.shape == result.params.shape
    assert result.E.shape == (result.params.shape[0], 2, 2)
    assert result.time.shape == (result.params.shape[0],)
    assert result.iterations.shape == (result.params.shape[0],)
    assert result.criterion.shape == (result.params.shape[0],)

    assert np.all(np.isfinite(result.params))
    assert np.all(np.isfinite(result.sigma))
    assert np.all(np.isfinite(result.E))
    assert np.all(np.isfinite(result.covariance))


def test_structural_Bii_terms_are_exact_zero():
    dt = 0.1
    _, phi = _synthetic_phases(dt=dt)

    result = run_inference(
        phi,
        dt=dt,
        window_seconds=20.0,
        pr=0.20,
    )

    # Historical layout for N=2:
    # [omega, A_i1, A_i2, B_i1, B_i2]
    # Equation 0 -> B_11 at index 3
    # Equation 1 -> B_22 at index 4
    assert np.all(result.params[:, 0, 3] == 0.0)
    assert np.all(result.params[:, 1, 4] == 0.0)
    assert np.all(result.sigma[:, 0, 3] == 0.0)
    assert np.all(result.sigma[:, 1, 4] == 0.0)

    assert np.all(result.structural_zero_mask[0, 3])
    assert np.all(result.structural_zero_mask[1, 4])


def test_noise_matrix_is_symmetric_positive_definite():
    dt = 0.1
    _, phi = _synthetic_phases(dt=dt)

    result = run_inference(
        phi,
        dt=dt,
        window_seconds=20.0,
        pr=0.20,
    )

    for E in result.E:
        assert np.allclose(E, E.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(E)
        assert np.all(eigvals > 0.0)


def test_propagation_parameter_and_backward_compatibility_alias():
    dt = 0.1
    t, phi = _synthetic_phases(dt=dt)

    result = run_inference(
        phi,
        dt=dt,
        t=t,
        window_seconds=20.0,
        pr=0.20,
    )

    assert result.pr == pytest.approx(0.20)
    assert result.p_internal == pytest.approx(4.0)

    # Historical two-value unpacking remains available.
    params, centers = result
    assert params is result.params
    assert centers is result.time

    # Old pw argument is accepted temporarily, but must warn.
    with pytest.warns(DeprecationWarning):
        old_api_result = run_inference(
            phi,
            dt=dt,
            t=t,
            window_seconds=20.0,
            pw=0.20,
        )

    assert old_api_result.pr == pytest.approx(0.20)
    assert old_api_result.p_internal == pytest.approx(4.0)


def test_sequential_propagation_rejects_overlapping_windows():
    dt = 0.1
    _, phi = _synthetic_phases(dt=dt)

    window_size = int(20.0 / dt)

    with pytest.raises(ValueError, match="non-overlapping"):
        run_inference(
            phi,
            dt=dt,
            window_size=window_size,
            step_size=window_size // 2,
            pr=0.20,
        )
