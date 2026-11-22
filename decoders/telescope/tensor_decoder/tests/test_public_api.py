from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tensor_decoder import DecoderConfig, DecoderContext, Experiment, build_decoder
from tensor_decoder.baseline import CorrelatedMatching


def test_one_public_factory_and_default_configuration():
    cfg = DecoderConfig()
    assert (cfg.path_factor, cfg.min_gain, cfg.ambiguous_gap, cfg.min_margin) == (3, 8, 8, 3)
    assert cfg.iterations == 2 and cfg.reverse_seed
    with pytest.raises(ValueError):
        replace(cfg, iterations=0)


def test_large_model_cap_uses_correlated_matching():
    e = Experiment(17, 17, 0.005)
    c = e.circuit()
    ctx = DecoderContext(e, c, c.detector_error_model(decompose_errors=True))
    decoder = build_decoder(ctx)
    assert decoder.search is None
    events = c.compile_detector_sampler(seed=1527).sample(8, bit_packed=True)
    events.flags.writeable = False
    np.testing.assert_array_equal(
        decoder.decode_batch(events), CorrelatedMatching(ctx).decode_batch(events)
    )


def test_stored_d15_hard_cases_preserve_validated_labels_and_scores():
    import json

    root = Path(__file__).parents[1] / "results/validation"
    cases = []
    for lane in ("a", "b"):
        data = json.loads((root / f"validation_{lane}.json").read_text())
        cases.extend(data["cases"][::100])
    e = Experiment(15, 15, 0.005)
    c = e.circuit()
    decoder = build_decoder(DecoderContext(e, c, c.detector_error_model(decompose_errors=True)))
    events = np.array([case["events"] for case in cases], np.uint8)
    events.flags.writeable = False
    actual, details = decoder.decode_with_diagnostics(events)
    np.testing.assert_array_equal(actual[:, 0], [case["candidate"] for case in cases])
    expected = [case["margin"] if case["margin"] is not None else np.nan for case in cases]
    np.testing.assert_array_equal(details["competition_margin"], expected)
