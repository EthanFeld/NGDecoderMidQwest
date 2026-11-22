"""Descriptive distance extrapolations with explicit fit assumptions.

These models do not certify large-distance decoder behavior or budget admission.
Binomial likelihood includes zero-error points without substituting a fake LER.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, xlogy
from scipy.stats import chi2, t


def binomial_fit(distances, errors, shots):
    d, k, n = (np.asarray(value, dtype=float) for value in (distances, errors, shots))
    if (
        d.ndim != 1
        or d.shape != k.shape
        or d.shape != n.shape
        or len(d) < 3
        or len(np.unique(d)) != len(d)
        or not np.all(np.isfinite([d, k, n]))
        or np.any(d <= 0)
        or np.any(n <= 0)
        or np.any(k < 0)
        or np.any(k > n)
        or np.any(k != np.floor(k))
        or np.any(n != np.floor(n))
    ):
        raise ValueError("need at least three distinct distances with valid binomial counts")
    if k.sum() == 0 or k.sum() == n.sum():
        raise ValueError("all-zero/all-failure data cannot identify a distance slope")
    center, scale = float(d.mean()), 10.0
    design = np.column_stack([np.ones(len(d)), (d - center) / scale])
    # A constant scale preserves the MLE while keeping high-shot likelihoods
    # and optimizer tolerances comparable to sparse-count fits. Inference below
    # still uses the unscaled Fisher information.
    normalizer = float(max(1, min(k.sum(), (n - k).sum())))

    def loss(beta):
        eta = design @ beta
        return float(
            np.sum(
                ((n - k) / normalizer) * np.logaddexp(0, eta)
                + (k / normalizer) * np.logaddexp(0, -eta)
            )
        )

    def gradient(beta):
        return design.T @ ((n * expit(design @ beta) - k) / normalizer)

    start = [np.log((k.sum() + 0.5) / (n.sum() - k.sum() + 0.5)), 0]
    result = minimize(loss, start, jac=gradient, method="BFGS", options={"gtol": 1e-10})
    if not result.success and np.max(np.abs(gradient(result.x))) > 1e-7:
        raise ValueError(f"binomial distance fit failed: {result.message}")
    fitted = expit(design @ result.x)
    information = design.T @ ((n * fitted * (1 - fitted))[:, None] * design)
    if np.linalg.cond(information) > 1e12:
        raise ValueError("distance slope is not identified by these counts")
    covariance = np.linalg.inv(information)
    deviance = float(
        2
        * np.sum(
            xlogy(k, np.divide(k, n * fitted)) + xlogy(n - k, np.divide(n - k, n * (1 - fitted)))
        )
    )
    return {
        "model": "logit(LER) = intercept + slope * (distance - center) / scale",
        "center": center,
        "scale": scale,
        "coefficients": result.x.tolist(),
        "covariance": covariance.tolist(),
        "training_distances": d.astype(int).tolist(),
        "errors": k.astype(int).tolist(),
        "shots": n.astype(int).tolist(),
        "deviance": max(0.0, deviance),
        "degrees_of_freedom": len(d) - 2,
        "deviance_p_value": float(chi2.sf(max(0.0, deviance), len(d) - 2)),
        "interval": "approximate 95% Wald fit interval conditional on logit-linear trend",
    }


def binomial_predict(fit, distances):
    d = np.asarray(distances, dtype=float)
    design = np.column_stack([np.ones(len(d)), (d - fit["center"]) / fit["scale"]])
    eta = design @ np.asarray(fit["coefficients"])
    variance = np.einsum("ij,jk,ik->i", design, np.asarray(fit["covariance"]), design)
    half = 1.959963984540054 * np.sqrt(np.maximum(variance, 0))
    return expit(eta), expit(eta - half), expit(eta + half)


def power_fit(distances, times):
    d, y = np.asarray(distances, dtype=float), np.asarray(times, dtype=float)
    if (
        d.ndim != 1
        or d.shape != y.shape
        or len(d) < 3
        or len(np.unique(d)) != len(d)
        or not np.all(np.isfinite([d, y]))
        or np.any(d <= 0)
        or np.any(y <= 0)
    ):
        raise ValueError("need at least three distinct positive distances and positive timings")
    design = np.column_stack([np.ones(len(d)), np.log(d / 15)])
    beta = np.linalg.lstsq(design, np.log(y), rcond=None)[0]
    residual = np.log(y) - design @ beta
    covariance = float(residual @ residual / (len(d) - 2)) * np.linalg.inv(design.T @ design)
    return {
        "model": "log(time) = intercept + exponent * log(distance / 15)",
        "coefficients": beta.tolist(),
        "covariance": covariance.tolist(),
        "training_distances": d.astype(int).tolist(),
        "times": y.tolist(),
        "degrees_of_freedom": len(d) - 2,
        "interval": "95% t interval for fitted log-mean conditional on power-law trend",
    }


def power_predict(fit, distances):
    d = np.asarray(distances, dtype=float)
    design = np.column_stack([np.ones(len(d)), np.log(d / 15)])
    mean = design @ np.asarray(fit["coefficients"])
    variance = np.einsum("ij,jk,ik->i", design, np.asarray(fit["covariance"]), design)
    half = t.ppf(0.975, fit["degrees_of_freedom"]) * np.sqrt(np.maximum(variance, 0))
    return np.exp(mean), np.exp(mean - half), np.exp(mean + half)
