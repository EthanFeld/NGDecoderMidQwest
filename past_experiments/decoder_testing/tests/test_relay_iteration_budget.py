"""Check limits and early stopping against the actual pinned native decoder."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from surface_decoders.cli import main as estimate_main
from surface_decoders.ibm_relay_bp import (
    PAPER_S1_CONFIG,
    PAPER_S5_CONFIG,
    TUNED_CONFIG,
    IBMRelayBPConfig,
)

relay_bp = pytest.importorskip("relay_bp")


@pytest.mark.parametrize("pre,sets,per_leg", [(5, 0, 7), (5, 1, 7), (5, 4, 7), (9, 2, 3)])
def test_unconvergeable_syndrome_executes_exact_total_iteration_limit(pre, sets, per_leg):
    # Identical checks cannot equal contradictory syndrome bits. No early success
    # can conceal a forwarding, off-by-one, or total-iteration accounting error.
    check = csr_matrix(np.array([[1, 1], [1, 1]], dtype=np.uint8))
    config = IBMRelayBPConfig(pre_iter=pre, num_sets=sets, set_max_iter=per_leg)
    native = relay_bp.RelayDecoderF32(
        check, error_priors=np.array([0.01, 0.01]), **config.native_kwargs()
    )
    result = native.decode_detailed(np.array([0, 1], dtype=np.uint8))
    assert not result.success
    assert result.iterations == config.max_iterations == pre + sets * per_leg
    assert config.total_legs == sets + 1
    assert result.max_iter == pre  # Returned leg cap is not the total search budget.


def test_solution_target_counts_repeat_valid_solutions_and_stops_early():
    config = IBMRelayBPConfig(pre_iter=80, num_sets=600, set_max_iter=60, stop_nconv=5)
    native = relay_bp.RelayDecoderF32(
        csr_matrix(np.array([[1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)),
        error_priors=np.array([0.01] * 3),
        **config.native_kwargs(),
    )
    result = native.decode_detailed(np.zeros(3, dtype=np.uint8))
    assert result.success and result.iterations == 5
    assert config.max_iterations == 36080
    # This graph has only one valid correction for this syndrome; it is found five times.
    np.testing.assert_array_equal(result.decoding, [0, 0, 0])


def test_all_stopping_mode_exhausts_legs_but_each_leg_stops_on_convergence():
    config = IBMRelayBPConfig(
        pre_iter=80, num_sets=4, set_max_iter=60, stop_nconv=1, stopping_criterion="all"
    )
    native = relay_bp.RelayDecoderF32(
        csr_matrix(np.array([[1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)),
        error_priors=np.array([0.01] * 3),
        **config.native_kwargs(),
    )
    result = native.decode_detailed(np.zeros(3, dtype=np.uint8))
    assert result.success and result.iterations == config.total_legs == 5


def test_paper_counts_and_latency_profile_are_explicitly_different():
    assert (TUNED_CONFIG.total_legs, TUNED_CONFIG.max_iterations) == (32, 1940)
    assert (PAPER_S1_CONFIG.total_legs, PAPER_S1_CONFIG.max_iterations) == (301, 18080)
    assert (PAPER_S5_CONFIG.total_legs, PAPER_S5_CONFIG.max_iterations) == (601, 36080)
    assert PAPER_S5_CONFIG.stop_nconv == 5
    assert PAPER_S5_CONFIG.detector_mode == "full"
    assert PAPER_S5_CONFIG.alpha is None
    assert replace(PAPER_S5_CONFIG, alpha=0).native_kwargs()["alpha"] == 0
    with pytest.raises(ValueError, match="stopping_criterion"):
        replace(TUNED_CONFIG, stopping_criterion="unknown")


def test_paper_resource_cli_reports_total_legs_and_iteration_limit(capsys):
    import json

    assert (
        estimate_main(
            ["estimate", "--distances", "3", "--decoders", "ibm_relay_bp", "--ibm-paper-s5"]
        )
        == 0
    )
    row = json.loads(capsys.readouterr().out)
    assert row["ibm_search_budget"] == {
        "total_legs": 601,
        "maximum_iterations": 36080,
        "stopping_criterion": "nconv",
    }
    assert row["ibm_relay_bp_config"]["detector_mode"] == "full"
