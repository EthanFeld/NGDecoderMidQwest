import numpy as np

from examples.distance_projection import power_fit
from examples.plot_full_xyz_distance_sweep import build_models, ordered_time_curves


def test_sparse_tail_values_are_not_promoted_into_tail_projections():
    points = []
    errors = [100, 30, 10, 3, 1, 0, 0]
    for d, k in zip([3, 5, 7, 9, 11, 13, 15], errors, strict=True):
        base = {
            "shots": 10000,
            "errors": k,
            "quality_complete": True,
            "median_us": d**3,
            "p95_us": 2 * d**3,
            "p99_us": 3 * d**3,
        }
        points.append(
            {
                "distance": d,
                "noise": 0.003,
                "decoders": {
                    "relay_bp": {
                        **base,
                        "latency_samples": 64,
                        "median_resolved": True,
                        "p95_resolved": False,
                        "p99_resolved": False,
                    },
                    "pymatching": {
                        **base,
                        "latency_samples": 1024,
                        "median_resolved": True,
                        "p95_resolved": True,
                        "p99_resolved": True,
                    },
                },
            }
        )
    models, rows = build_models(points, [0.003])
    native = models["0.003"]["relay_bp"]
    assert "median_us" in native["time"]
    assert "p95_us" not in native["time"] and "p99_us" not in native["time"]
    assert "p95_us" in native["unavailable"] and "p99_us" in native["unavailable"]
    assert not any(r["decoder"] == "relay_bp" and r["metric"] == "p99_us" for r in rows)
    assert all(
        r["kind"] == "conditional_trend_projection" and not r["budget_admitted"] for r in rows
    )


def test_single_positive_distance_does_not_invent_identified_ler_slope():
    points = []
    for d, k in [(3, 10), (5, 0), (7, 0), (9, 0)]:
        m = {"shots": 10000, "errors": k, "quality_complete": True}
        points.append({"distance": d, "noise": 0.001, "decoders": {"relay_bp": m, "pymatching": m}})
    models, rows = build_models(points, [0.001])
    assert "ler" not in models["0.001"]["relay_bp"]
    assert not rows


def test_extrapolated_percentiles_stay_ordered_when_independent_fits_cross():
    d = np.array([7, 9, 11, 13, 15])
    entry = {
        "time": {
            "median_us": power_fit(d, d**3),
            "p95_us": power_fit(d, 16 * d**2),
            "p99_us": power_fit(d, 17 * d**2),
        }
    }
    curves = ordered_time_curves(entry, [31])
    assert curves["median_us"][0][0] <= curves["p95_us"][0][0] <= curves["p99_us"][0][0]
    for mean, low, high in curves.values():
        assert low[0] <= mean[0] <= high[0]
