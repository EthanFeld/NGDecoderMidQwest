import numpy as np
import pytest
from scipy.special import expit

from examples.distance_projection import binomial_fit, binomial_predict, power_fit, power_predict


def test_binomial_likelihood_recovers_count_trend_and_keeps_zero_error_points():
    d = np.arange(7, 16, 2)
    n = np.full(5, 100_000_000)
    expected = expit(-5 - 0.5 * (d - 7))
    k = np.rint(n * expected).astype(int)
    fit = binomial_fit(d, k, n)
    predicted, low, high = binomial_predict(fit, d)
    np.testing.assert_allclose(predicted, expected, rtol=1e-4)
    assert np.all(low < predicted) and np.all(predicted < high)
    sparse = binomial_fit(d, [4, 2, 1, 0, 0], [4096] * 5)
    assert sparse["errors"] == [4, 2, 1, 0, 0]
    assert binomial_predict(sparse, [31])[0][0] > 0


def test_ler_slope_is_not_forced_to_improve():
    fit = binomial_fit([11, 13, 15], [10, 20, 40], [4096] * 3)
    assert binomial_predict(fit, [31])[0][0] > binomial_predict(fit, [15])[0][0]


def test_count_rescaling_preserves_mle_and_scales_fit_uncertainty():
    d = [7, 9, 11, 13]
    k = np.array([100, 30, 9, 3])
    n = np.full(4, 100000)
    multiplier = 100000
    small, large = binomial_fit(d, k, n), binomial_fit(d, k * multiplier, n * multiplier)
    np.testing.assert_allclose(large["coefficients"], small["coefficients"], atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(large["covariance"]) * multiplier, small["covariance"], rtol=1e-6
    )


def test_unidentified_ler_slope_is_not_invented():
    with pytest.raises(ValueError, match="cannot identify"):
        binomial_fit([11, 13, 15], [0, 0, 0], [4096] * 3)
    with pytest.raises(ValueError, match="at least three"):
        binomial_fit([2], [100], [4096])


def test_power_law_projection_recovers_known_scaling():
    d = np.arange(7, 16, 2)
    fit = power_fit(d, 0.002 * d**3)
    predicted, low, high = power_predict(fit, [15, 31])
    np.testing.assert_allclose(predicted, 0.002 * np.array([15, 31]) ** 3, rtol=1e-12)
    assert np.all(low <= predicted) and np.all(high >= predicted)
