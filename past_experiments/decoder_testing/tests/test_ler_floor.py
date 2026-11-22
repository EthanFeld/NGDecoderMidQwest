import itertools
import math
from dataclasses import replace

import numpy as np
import pytest
import stim
from scipy.stats import beta

from qecbench import Experiment, SweepConfig
from qecbench.cli import main
from qecbench.floor_runner import binomial_upper, run_floor_sweep, summarize_floor
from qecbench.ler_floor import (
    Fault,
    FloorOptions,
    analyze_dem,
    complement_pair_risk,
    exact_joint_ler,
    genie_path_bound,
    read_faults,
)
from qecbench.storage import read_json


def exhaustive_risk(dem):
    """Independent truth: enumerate original events, then sum each syndrome coset."""
    mechanisms = []
    for instruction in dem.flattened():
        if instruction.type == "error":
            detectors = logical = 0
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    detectors ^= 1 << target.val
                elif target.is_logical_observable_id():
                    logical ^= 1 << target.val
            mechanisms.append((instruction.args_copy()[0], detectors, logical))
    joint = {}
    for errors in itertools.product((0, 1), repeat=len(mechanisms)):
        probability = 1.0
        syndrome = logical = 0
        for error, (p, detectors, observables) in zip(errors, mechanisms, strict=True):
            probability *= p if error else 1 - p
            if error:
                syndrome ^= detectors
                logical ^= observables
        joint.setdefault(syndrome, {}).setdefault(logical, 0.0)
        joint[syndrome][logical] += probability
    return math.fsum(math.fsum(sorted(classes.values())[:-1]) for classes in joint.values())


@pytest.mark.parametrize(
    "text",
    [
        "logical_observable L0",
        "error(0.2) L0",
        "error(0.8) L0",
        "error(1) D0 L0\nerror(0.1) D0\nerror(0.2) D0 L0",
        "error(0.3) D0\nerror(0.2) D0 L0\nerror(0.1) L0",
        "error(0.2) D0 L0 ^ D0 L1\nerror(0.4) L0 L1\nerror(0.1) L1",
        "error(0.2) D0\nerror(0.3) D0\nerror(0.4) D0 L0\nerror(0.5) D0 L1",
        "repeat 2 {\nerror(0.2) D0 L0\nshift_detectors 1\n}\nerror(0.1) D0 D1",
        "error(0.1) D19 L0\nerror(0.2) D19\ndetector D90",
    ],
)
def test_exact_joint_matches_original_event_enumeration(text):
    dem = stim.DetectorErrorModel(text)
    result = analyze_dem(dem)
    assert result["method"] == "exact_joint"
    assert result["exact_ler"] == pytest.approx(exhaustive_risk(dem), abs=1e-14)


def test_separator_components_remain_one_correlated_event():
    dem = stim.DetectorErrorModel("error(0.2) D0 L0 ^ D1 L0\nerror(0.1) D0 D1 L0")
    faults = read_faults(dem, 256)
    assert faults[0].detectors == (0, 1)
    assert faults[0].logical == 0
    assert len(faults) == 2


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.1],
        [0.5],
        [0.5, 0.5, 0.5],
        [0.01] * 7,
        [0.02, 0.2, 0.07, 0.4, 0.01],
        [0.9, 0.8, 0.99],
    ],
)
def test_quantized_pair_bounds_enclose_exhaustive_overlap(probabilities):
    exact = 0
    for errors in itertools.product((0, 1), repeat=len(probabilities)):
        a = math.prod(p if e else 1 - p for p, e in zip(probabilities, errors, strict=True))
        b = math.prod(1 - p if e else p for p, e in zip(probabilities, errors, strict=True))
        exact += 0.5 * min(a, b)
    low, high = complement_pair_risk(np.array(probabilities), 16384)
    assert low <= exact + 1e-15
    assert exact <= high + 1e-15
    assert high - low < max(1e-12, 0.02 * exact)


def test_rare_chain_avoids_subtraction_cancellation():
    faults = tuple(
        Fault(0.001, (i - 1, i) if 0 < i < 16 else ((0,) if i == 0 else (15,)), int(i == 16))
        for i in range(17)
    )
    result = genie_path_bound(faults, 16, 1, FloorOptions(paths=1))
    assert 0 < result["lower"] < 1e-20
    assert result["paths"][0]["length"] == 17


@pytest.mark.parametrize("seed", range(20))
def test_genie_lower_bound_never_beats_full_optimal_risk(seed):
    rng = np.random.default_rng(seed)
    lines = ["logical_observable L0", "logical_observable L1"]
    for _ in range(8):
        targets = [f"D{i}" for i in range(4) if rng.random() < 0.55]
        targets += [f"L{i}" for i in range(2) if rng.random() < 0.4]
        if not targets:
            targets = ["L0"]
        lines.append(f"error({rng.uniform(0.01, 0.95)}) " + " ".join(targets))
    dem = stim.DetectorErrorModel("\n".join(lines))
    result = analyze_dem(dem, FloorOptions(exact_bits=0, paths=8, chain_bins=4096))
    assert result["minimum_ler_lower"] <= exhaustive_risk(dem) + 1e-12


def test_independent_pair_packing_can_share_detector_nodes():
    faults = (Fault(0.1, (0,), 0), Fault(0.2, (0,), 1), Fault(0.15, (0,), 0), Fault(0.3, (0,), 1))
    result = genie_path_bound(faults, 1, 1, FloorOptions(paths=4))
    assert len(result["paths"]) == 2
    supports = [set(p["mechanisms"]) for p in result["paths"]]
    assert not supports[0].intersection(supports[1])
    individual = [p["pair_ler_low"] for p in result["paths"]]
    assert result["lower"] == pytest.approx((1 - math.prod(1 - 2 * p for p in individual)) / 2)


def test_hypergraph_witness_and_trivial_fallback():
    dem = stim.DetectorErrorModel("error(0.1) D0 D1 D2\nerror(0.2) D0 D1 D2 L0")
    result = analyze_dem(dem, FloorOptions(exact_bits=0))
    assert result["logical_paths"] == 1
    assert result["minimum_ler_lower"] == pytest.approx(exhaustive_risk(dem), rel=0.005)
    no_ambiguity = analyze_dem(
        stim.DetectorErrorModel("error(0.1) D0 L0"), FloorOptions(exact_bits=0)
    )
    assert no_ambiguity["method"] == "trivial"
    assert no_ambiguity["minimum_ler_lower"] == 0


def test_sparse_detector_ids_do_not_allocate_by_largest_label():
    dem = stim.DetectorErrorModel("error(0.1) D1000000\nerror(0.2) D1000000 L0")
    assert analyze_dem(dem)["exact_ler"] == pytest.approx(0.1)
    bounded = analyze_dem(dem, FloorOptions(exact_bits=0))
    assert bounded["minimum_ler_lower"] == pytest.approx(0.1, rel=0.005)


def test_native_zero_weight_logical_edge():
    bounded = analyze_dem(stim.DetectorErrorModel("error(0.5) L0"), FloorOptions(exact_bits=0))
    assert bounded["minimum_ler_lower"] == 0.5


def test_work_and_memory_caps_skip_dense_enumeration():
    faults = tuple(Fault(0.1, (i,), int(i == 0)) for i in range(12))
    assert exact_joint_ler(faults, 1, FloorOptions(max_exact_work=1))[1] == "work_limit"
    tiny = FloorOptions(memory_mib=0.005, chain_bins=16)
    assert exact_joint_ler(faults, 1, tiny)[1] == "memory_limit"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"paths": 0},
        {"chain_bins": 1},
        {"exact_bits": 31},
        {"exact_bits": True},
        {"memory_mib": 0},
        {"memory_mib": math.inf},
        {"max_exact_work": 0},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        FloorOptions(**kwargs)


@pytest.mark.parametrize("errors,shots", [(0, 100), (2, 100), (50, 100), (99, 100), (100, 100)])
def test_one_sided_exact_binomial_bound(errors, shots):
    expected = beta.ppf(0.95, errors + 1, shots - errors) if errors < shots else 1
    assert binomial_upper(errors, shots, 0.05) == pytest.approx(expected)


def test_floor_cli_reports_and_resume(tmp_path):
    output = tmp_path / "floor"
    args = [
        "floor",
        "--distances",
        "3",
        "5",
        "--rounds",
        "1",
        "--noise",
        "0",
        "--shots",
        "17",
        "--decoders",
        "zero",
        "pymatching",
        "--output",
        str(output),
    ]
    assert main(args) == 0
    rows = read_json(output / "ler_floor.json")
    assert len(rows) == 2
    assert all(row["minimum_ler_lower"] == row["minimum_ler_upper"] == 0 for row in rows)
    assert all(row["status"] == "exact" for row in rows)
    assert (output / "ler_floor.csv").read_text().startswith("experiment_id,")
    original_csv = (output / "ler_floor.csv").read_text()
    manifest = read_json(output / "ler_floor_manifest.json")
    assert manifest["simultaneous_comparisons"] == 4
    assert manifest["alpha_per_comparison"] == pytest.approx(0.05 / 4)
    assert main([*args, "--resume"]) == 0
    assert read_json(output / "ler_floor.json") == rows
    assert (output / "ler_floor.csv").read_text() == original_csv
    figure = tmp_path / "floor.png"
    assert main(["plot-floor", str(output), "--output", str(figure)]) == 0
    assert figure.read_bytes().startswith(b"\x89PNG")


def test_fixed_shots_required_and_no_output_on_invalid_request(tmp_path):
    config = SweepConfig(
        distances=[3], noise=[0.01], shots=10, max_errors=1, output=str(tmp_path / "bad")
    )
    with pytest.raises(ValueError, match="fixed shots"):
        run_floor_sweep(config)
    with pytest.raises(ValueError, match="confidence"):
        run_floor_sweep(replace(config, max_errors=0), confidence=1)
    assert not config.output_path.exists()


def test_numerical_lower_is_not_clipped_to_a_conflicting_measured_upper():
    analysis = analyze_dem(stim.DetectorErrorModel("error(0.5) L0"))
    row = {
        "experiment_id": "test",
        **Experiment(3, 3, 0.1).__dict__,
        "decoder": "bad",
        "errors": 0,
        "shots": 1000,
        "logical_error_rate": 0.0,
        "complete": True,
    }
    summary = summarize_floor(analysis, [row], 0.05)
    assert summary["status"] == "confidence_conflict"
    assert summary["minimum_ler_lower"] == 0.5
    assert summary["minimum_ler_upper"] == 0.5
