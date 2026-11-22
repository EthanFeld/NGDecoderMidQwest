"""Bounded runtime pilot for the frozen full XYZ accuracy profile.

Incomplete pilots are not quality/quantile estimates. A dedicated child is stopped
if a single decode exceeds the pilot limit; no completed-shot subset is scored.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from pathlib import Path

from qecbench import DecoderContext, Experiment
from qecbench.storage import write_json
from surface_decoders.ibm_relay_bp import ACCURACY_CONFIG, IBMRelayBPDecoder


def worker(noise, channel):
    experiment = Experiment(15, 15, noise)
    circuit = experiment.circuit()
    context = DecoderContext(
        experiment, circuit, circuit.detector_error_model(decompose_errors=True)
    )
    decoder = IBMRelayBPDecoder(context, ACCURACY_CONFIG)
    events = circuit.compile_detector_sampler(seed=1337000 + int(noise * 1e6)).sample(
        shots=4, bit_packed=True
    )
    events.flags.writeable = False
    channel.put({"event": "ready"})
    for i in range(4):
        channel.put({"event": "start", "shot": i})
        start = time.perf_counter()
        decoder.decode_batch(events[i : i + 1])
        channel.put({"event": "done", "shot": i, "seconds": time.perf_counter() - start})
    channel.put({"event": "complete"})


def main():
    context = mp.get_context("spawn")
    rows = []
    for p in [0.003, 0.007, 0.01]:
        channel = context.Queue()
        process = context.Process(target=worker, args=(p, channel))
        process.start()
        row = {"distance": 15, "noise": p, "pilot": True, "samples_seconds": [], "complete": False}
        try:
            while True:
                event = channel.get(timeout=45 if not row.get("ready") else 20)
                if event["event"] == "ready":
                    row["ready"] = True
                    print("READY", p, flush=True)
                elif event["event"] == "done":
                    row["samples_seconds"].append(event["seconds"])
                    print("SHOT", p, event["shot"], round(event["seconds"], 3), "s", flush=True)
                elif event["event"] == "complete":
                    row["complete"] = True
                    break
        except queue.Empty:
            row["censored_single_call_lower_bound_seconds"] = 20
            print("PILOT CALL LIMIT", p, "20 s; no LER or quantile estimate", flush=True)
        finally:
            if process.is_alive():
                process.terminate()
            process.join()
        rows.append(row)
        output = Path("past_experiments/decoder_testing/results/full-xyz-distance-noise-20261121")
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "runtime_pilot.json", rows)


if __name__ == "__main__":
    main()
