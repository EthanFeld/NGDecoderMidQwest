import json
from dataclasses import replace

import pytest

from qecbench import SweepConfig, run_sweep, runner
from qecbench.storage import OutputLock, read_json


def config_at(path, **kwargs):
    values = dict(
        distances=[3, 5],
        noise=[0.005],
        bases=["x", "z"],
        decoders=["pymatching", "zero"],
        shots=257,
        batch_size=100,
        warmup_shots=8,
        seed=9,
        output=str(path),
    )
    values.update(kwargs)
    return SweepConfig(**values)


def counters(rows):
    return {(r["experiment_id"], r["decoder"]): (r["shots"], r["errors"]) for r in rows}


def test_exact_caps_paired_decoders_outputs_and_resume(tmp_path):
    config = config_at(tmp_path / "run")
    rows = run_sweep(config)
    assert len(rows) == 8
    assert all(r["shots"] == 257 and r["batches"] == 3 for r in rows)
    assert all(r["complete"] and r["stop_reason"] == "shots" for r in rows)
    assert all(r["compile_seconds"] > 0 and r["decode_seconds"] > 0 for r in rows)
    assert all(0 <= r["ci95_low"] <= r["ci95_high"] <= 1 for r in rows)
    assert (config.output_path / "results.csv").read_text().startswith("experiment_id,")
    assert len(read_json(config.output_path / "results.json")) == 8
    assert run_sweep(replace(config, workers=2), resume=True) == rows
    with pytest.raises(ValueError, match="already contains"):
        run_sweep(config)
    with pytest.raises(ValueError, match="resume rejected"):
        run_sweep(replace(config, seed=10), resume=True)


def test_spawn_workers_and_native_unpacked_plugins_agree(tmp_path):
    single = run_sweep(config_at(tmp_path / "single"))
    parallel = run_sweep(config_at(tmp_path / "parallel", workers=2))
    assert counters(single) == counters(parallel)
    plugins = run_sweep(
        config_at(
            tmp_path / "plugins",
            workers=2,
            decoders=[
                "pymatching",
                "examples.custom_decoder:build_decoder",
                "examples.packed_decoder:build_decoder",
            ],
        )
    )
    for experiment_id in {r["experiment_id"] for r in plugins}:
        results = [r["errors"] for r in plugins if r["experiment_id"] == experiment_id]
        assert len(set(results)) == 1


def test_resume_replays_only_uncommitted_batch(tmp_path, monkeypatch):
    config = config_at(tmp_path / "interrupted", distances=[3], bases=["z"], shots=400)
    original = runner.write_json

    def interrupt_before_second_commit(path, state):
        if "batches" in state and state["batches"] == 2:
            raise OSError("simulated interruption")
        original(path, state)

    with monkeypatch.context() as patch:
        patch.setattr(runner, "write_json", interrupt_before_second_commit)
        with pytest.raises(OSError, match="simulated interruption"):
            run_sweep(config)
    partial = read_json(config.output_path / "results.json")
    assert all(r["shots"] == 100 and not r["complete"] for r in partial)
    resumed = run_sweep(config, resume=True)
    fresh = run_sweep(replace(config, output=str(tmp_path / "fresh")))
    assert counters(resumed) == counters(fresh)
    assert all(r["batches"] == 4 for r in resumed)


def test_seed_stream_matches_manual_stim_sampling(tmp_path):
    config = config_at(tmp_path / "manual", distances=[3], bases=["z"], decoders=["zero"])
    (row,) = run_sweep(config)
    (experiment,) = config.experiments()
    circuit = experiment.circuit()
    errors = 0
    for batch in range(3):
        _, actual = circuit.compile_detector_sampler(
            seed=runner.batch_seed(config.seed, experiment, batch)
        ).sample(shots=min(100, 257 - batch * 100), bit_packed=True, separate_observables=True)
        errors += int((actual[:, 0] != 0).sum())
    assert row["errors"] == errors


def test_stopping_requires_all_decoders(tmp_path):
    rows = run_sweep(
        config_at(
            tmp_path / "stop",
            distances=[3],
            bases=["z"],
            noise=[0.1],
            shots=2000,
            max_errors=1,
            min_shots=150,
        )
    )
    assert all(r["shots"] == 200 and r["stop_reason"] == "max_errors" for r in rows)
    # Noise-free decoding never reaches the error target; exact shot cap still applies.
    rows = run_sweep(
        config_at(tmp_path / "no-errors", distances=[3], bases=["z"], noise=[0], max_errors=1)
    )
    assert all(r["shots"] == 257 and r["errors"] == 0 for r in rows)


@pytest.mark.parametrize("decoder", ["fixture_decoders:mutating", "fixture_decoders:wrong_shape"])
def test_invalid_decoder_fails_visibly(tmp_path, decoder):
    config = config_at(
        tmp_path / decoder.split(":")[-1], distances=[3], bases=["z"], decoders=[decoder]
    )
    with pytest.raises(RuntimeError, match="failed during warmup"):
        run_sweep(config)
    assert read_json(config.output_path / "results.json") == []


def test_lock_and_missing_resume(tmp_path):
    output = tmp_path / "lock"
    output.mkdir()
    with OutputLock(output):
        with pytest.raises(ValueError, match="another run owns"):
            with OutputLock(output):
                pass
    with OutputLock(output):
        pass
    with pytest.raises(ValueError, match="no manifest"):
        run_sweep(config_at(tmp_path / "missing"), resume=True)


def test_checkpoint_corruption_rejected(tmp_path):
    config = config_at(tmp_path / "corrupt", distances=[3], bases=["z"])
    run_sweep(config)
    (path,) = (config.output_path / "checkpoints").glob("*.json")
    data = read_json(path)
    data["shots"] = 999999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="invalid checkpoint"):
        run_sweep(config, resume=True)
