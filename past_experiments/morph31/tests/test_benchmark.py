import json

import numpy as np
import pytest

from morph31.benchmark import (
    main,
    paired_accuracy,
    paired_operation_reduction,
    run_benchmark,
    wilson,
)
from morph31.circuit import Experiment
from morph31.cost_model import CycleModel, Operations


def test_bounded_paired_benchmark(tmp_path):
    report = run_benchmark(Experiment(3, 3), tmp_path, shots=20, seed=31)
    assert report["completed_shots"] == 20
    conventional = report["decoders"]["conventional"]
    assert conventional["shots"] == report["reference"]["shots"] == 20
    assert conventional["peak_total_packed_bytes"] > conventional["peak_working_bytes"]
    assert conventional["cycles"]["mean_cycles_per_round"] > 0
    assert wilson(0, 100)[1] > 0
    assert "not measured" in conventional["cycles"]["status"]
    assert CycleModel().cycles(Operations(memory_reads=5)) == 10


def test_cli_config_and_graph_rom(tmp_path):
    config = tmp_path / "test.toml"
    config.write_text('distance=3\nrounds=2\nbasis="z"\n')
    output, rom = tmp_path / "report.json", tmp_path / "graph.rom"
    assert main(["graph", "--config", str(config), "--cache", str(tmp_path / "cache"),
                 "--output", str(output), "--rom", str(rom)]) == 0
    report = json.loads(output.read_text())
    assert report["equivalence_verified"]
    assert report["rom_roundtrip_verified"]
    assert report["experiment"]["basis"] == "z"
    assert rom.stat().st_size == report["graph"]["graph_rom_bytes"]


def test_paired_comparisons_do_not_claim_zero_risk():
    truth = np.zeros((32, 1), dtype=np.uint8)
    candidate, reference = truth.copy(), truth.copy()
    comparison = paired_accuracy(candidate, reference, truth)
    assert comparison["mcnemar_exact_two_sided_p"] == 1
    assert comparison["added_failure_rate_ci95_wilson"][1] > 0
    candidate[:6] = 1
    comparison = paired_accuracy(candidate, reference, truth)
    assert comparison["added_failures"] == 6 and comparison["removed_failures"] == 0
    assert comparison["mcnemar_exact_two_sided_p"] == pytest.approx(2 / 64)
    reduction = paired_operation_reduction([100, 200, 300], [75, 150, 225])
    assert reduction["operation_reduction_fraction"] == .25
    assert reduction["operation_reduction_ci95_paired_delta"] == [.25, .25]
