"""Replay all saved hard cases against the validated predictions and scores."""

import json
from pathlib import Path

import numpy as np

from tensor_decoder import (
    DecoderContext,
    Experiment,
    build_decoder,
    qecbench_implementation_identity,
)
from tensor_decoder.matching import audit_vendor


def main():
    root = Path(__file__).resolve().parents[1]
    records = []
    for lane in ("a", "b"):
        data = json.loads((root / f"results/validation/validation_{lane}.json").read_text())
        records.extend(data["cases"])
    e = Experiment(15, 15, 0.005)
    c = e.circuit()
    decoder = build_decoder(DecoderContext(e, c, c.detector_error_model(decompose_errors=True)))
    events = np.asarray([record["events"] for record in records], np.uint8)
    events.flags.writeable = False
    actual, details = decoder.decode_with_diagnostics(events)
    np.testing.assert_array_equal(actual[:, 0], [r["candidate"] for r in records])
    np.testing.assert_array_equal(details["competition_states"], [r["states"] for r in records])
    margins = [r["margin"] if r["margin"] is not None else np.nan for r in records]
    np.testing.assert_array_equal(details["competition_margin"], margins)
    assert audit_vendor() == 37
    print(f"Verified {len(records)} hard-case predictions, margins and state counts.")
    print(json.dumps(qecbench_implementation_identity(), indent=2))


if __name__ == "__main__":
    main()
