"""Bounded-memory paired sampling; independent spawned workers per parameter point."""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Experiment, SweepConfig
from .decoders import DecoderContext, load_factory, validate_predictions
from .metrics import count_failures, summarize
from .storage import (
    OutputLock,
    canonical,
    digest,
    environment_identity,
    implementation_identity,
    read_json,
    write_json,
    write_reports,
)

SCHEMA_VERSION = 1


def experiment_id(experiment: Experiment) -> str:
    return digest(asdict(experiment))[:24]


def batch_seed(seed: int, experiment: Experiment, batch_index: int) -> int:
    return int(digest([seed, asdict(experiment), batch_index])[:16], 16) % (2**63)


def effective_batch_size(config: SweepConfig, num_detectors: int, num_observables: int) -> int:
    # Includes packed input, output, comparison scratch, and an unpacked adapter's
    # input + a second detector-sized buffer. Decoder-internal memory is separate.
    per_shot = max(
        1, 3 * ((num_detectors + 7) // 8) + 4 * ((num_observables + 7) // 8) + 2 * num_detectors
    )
    capacity = int(config.memory_mib * 1024**2) // per_shot
    if capacity < 1:
        raise ValueError("memory_mib is too small for even one shot at this distance")
    return min(config.batch_size, capacity)


def _finished(state: dict[str, Any], config: SweepConfig) -> bool:
    return state["shots"] >= config.shots or (
        config.max_errors > 0
        and state["shots"] >= config.min_shots
        and all(s["errors"] >= config.max_errors for s in state["stats"].values())
    )


def _checkpoint_path(output: Path, experiment: Experiment) -> Path:
    return output / "checkpoints" / f"{experiment_id(experiment)}.json"


def _read_checkpoint(
    path: Path, experiment: Experiment, config: SweepConfig, identity: str
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    state = read_json(path)
    if (
        state.get("identity") != identity
        or state.get("experiment") != asdict(experiment)
        or set(state.get("stats", {})) != set(config.decoders)
    ):
        raise ValueError(f"checkpoint identity mismatch: {path}")
    shots = state.get("shots")
    batches = state.get("batches")
    if (
        type(shots) is not int
        or not 0 < shots <= config.shots
        or type(batches) is not int
        or batches < 1
        or shots != min(batches * state["effective_batch_size"], config.shots)
        or state.get("complete") != _finished(state, config)
        or any(
            type(s.get("errors")) is not int or not 0 <= s["errors"] <= shots
            for s in state["stats"].values()
        )
    ):
        raise ValueError(f"invalid checkpoint counters: {path}")
    return state


def _rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for decoder, stats in state["stats"].items():
        row = {
            "experiment_id": state["experiment_id"],
            **state["experiment"],
            "decoder": decoder,
            "num_detectors": state["num_detectors"],
            "num_observables": state["num_observables"],
            "num_qubits": state["num_qubits"],
            "shots": state["shots"],
            "errors": stats["errors"],
            **summarize(stats["errors"], state["shots"], stats["decode_seconds"]),
            "compile_seconds": stats["compile_seconds"],
            "decode_seconds": stats["decode_seconds"],
            "sample_seconds": state["sample_seconds"],
            "batches": state["batches"],
            "effective_batch_size": state["effective_batch_size"],
            "wall_seconds": state["wall_seconds"],
            "complete": state["complete"],
            "stop_reason": state["stop_reason"],
        }
        rows.append(row)
    return rows


def _run_experiment(
    experiment: Experiment, config: SweepConfig, identity: str
) -> list[dict[str, Any]]:
    path = _checkpoint_path(config.output_path, experiment)
    state = _read_checkpoint(path, experiment, config, identity)
    if state and state["complete"]:
        return _rows(state)

    started = time.perf_counter()
    circuit = experiment.circuit()
    context = DecoderContext(
        experiment, circuit, circuit.detector_error_model(decompose_errors=True)
    )
    batch_size = effective_batch_size(config, context.num_detectors, context.num_observables)
    if state is None:
        state = {
            "identity": identity,
            "experiment_id": experiment_id(experiment),
            "experiment": asdict(experiment),
            "num_detectors": context.num_detectors,
            "num_observables": context.num_observables,
            "num_qubits": circuit.num_qubits,
            "effective_batch_size": batch_size,
            "shots": 0,
            "batches": 0,
            "sample_seconds": 0.0,
            "wall_seconds": 0.0,
            "complete": False,
            "stop_reason": "running",
            "stats": {
                name: {"errors": 0, "decode_seconds": 0.0, "compile_seconds": 0.0}
                for name in config.decoders
            },
        }
    previous_wall = state["wall_seconds"]
    decoders = {}
    for name in config.decoders:
        factory = load_factory(name)
        before = time.perf_counter()
        decoder = factory(context)
        state["stats"][name]["compile_seconds"] += time.perf_counter() - before
        if not callable(getattr(decoder, "decode_batch", None)):
            raise ValueError(f"decoder {name!r} did not return an object with decode_batch")
        decoders[name] = decoder

    if config.warmup_shots:
        warmup = min(config.warmup_shots, batch_size)
        events = circuit.compile_detector_sampler(
            seed=batch_seed(config.seed, experiment, -1)
        ).sample(shots=warmup, bit_packed=True)
        events.flags.writeable = False
        for name, decoder in decoders.items():
            try:
                validate_predictions(decoder.decode_batch(events), warmup, context.num_observables)
            except Exception as error:
                raise RuntimeError(
                    f"decoder {name!r} failed during warmup at {experiment}"
                ) from error

    while not _finished(state, config):
        batch_index = state["batches"]
        shots = min(batch_size, config.shots - state["shots"])
        before = time.perf_counter()
        sampler = circuit.compile_detector_sampler(
            seed=batch_seed(config.seed, experiment, batch_index)
        )
        events, actual = sampler.sample(shots=shots, separate_observables=True, bit_packed=True)
        state["sample_seconds"] += time.perf_counter() - before
        events.flags.writeable = False
        # All algorithms see the same shots; rotating order reduces fixed-order cache bias.
        names = config.decoders
        offset = batch_index % len(names)
        for name in names[offset:] + names[:offset]:
            try:
                before = time.perf_counter()
                predictions = decoders[name].decode_batch(events)
                elapsed = time.perf_counter() - before
                validate_predictions(predictions, shots, context.num_observables)
                state["stats"][name]["errors"] += count_failures(predictions, actual)
                state["stats"][name]["decode_seconds"] += elapsed
            except Exception as error:
                raise RuntimeError(
                    f"decoder {name!r} failed at {experiment}, batch {batch_index}"
                ) from error
        state["shots"] += shots
        state["batches"] += 1
        state["complete"] = _finished(state, config)
        state["stop_reason"] = (
            "shots"
            if state["shots"] == config.shots
            else "max_errors"
            if state["complete"]
            else "running"
        )
        state["wall_seconds"] = previous_wall + time.perf_counter() - started
        write_json(path, state)

    return _rows(state)


def _collect_rows(config: SweepConfig, identity: str) -> list[dict[str, Any]]:
    rows = []
    for experiment in config.experiments():
        state = _read_checkpoint(
            _checkpoint_path(config.output_path, experiment), experiment, config, identity
        )
        if state:
            rows.extend(_rows(state))
    return rows


def run_sweep(
    config: SweepConfig,
    *,
    resume: bool = False,
    progress: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run a sweep and return report rows. Use a __main__ guard with multiple workers."""
    # Validate plugin importability before creating the output directory.
    for name in config.decoders:
        load_factory(name)
    identity_data = {
        "schema_version": SCHEMA_VERSION,
        "config": config.identity(),
        "environment": environment_identity(),
        "implementation": implementation_identity(config.decoders),
    }
    identity = digest(identity_data)
    output = config.output_path
    output.mkdir(parents=True, exist_ok=True)
    with OutputLock(output):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if not resume:
                raise ValueError(
                    f"output already contains a run; use --resume or another path: {output}"
                )
            manifest = read_json(manifest_path)
            if manifest.get("identity") != identity or canonical(
                manifest.get("identity_data")
            ) != canonical(identity_data):
                raise ValueError(
                    "resume rejected: config, environment, or source changed; "
                    "use a new output directory"
                )
        else:
            if resume:
                raise ValueError(f"no manifest to resume in {output}")
            if any(path.name != ".run.lock" for path in output.iterdir()):
                raise ValueError(f"new run requires an empty output directory: {output}")
            write_json(
                manifest_path,
                {
                    "identity": identity,
                    "identity_data": identity_data,
                    "created_utc": datetime.now(UTC).isoformat(),
                    "command_config": asdict(config),
                    "sampling": "Stim circuit; independently seeded batches; paired decoders",
                    "timing": "decode calls only; amortized batch time is not single-shot latency",
                },
            )
        experiments = config.experiments()
        pending = []
        for experiment in experiments:
            state = _read_checkpoint(
                _checkpoint_path(output, experiment), experiment, config, identity
            )
            if not state or not state["complete"]:
                pending.append(experiment)
        try:
            if config.workers == 1 or len(pending) <= 1:
                for experiment in pending:
                    rows = _run_experiment(experiment, config, identity)
                    if progress:
                        progress(rows)
            else:
                # Spawn avoids fork-specific behavior and works on Windows, Linux, and macOS.
                with ProcessPoolExecutor(
                    max_workers=min(config.workers, len(pending)),
                    mp_context=multiprocessing.get_context("spawn"),
                ) as executor:
                    futures = {
                        executor.submit(_run_experiment, exp, config, identity): exp
                        for exp in pending
                    }
                    for future in as_completed(futures):
                        rows = future.result()
                        if progress:
                            progress(rows)
        finally:
            # Recover committed partial work even if a decoder fails or the user interrupts.
            write_reports(output, _collect_rows(config, identity))
        return _collect_rows(config, identity)
