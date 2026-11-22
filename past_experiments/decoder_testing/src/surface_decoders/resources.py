"""Preflight byte/work models; supplied platform rate converts work to estimated latency."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from qecbench import DecoderContext

from .model import ModelStats, model_stats


@dataclass(frozen=True)
class ResourceLimits:
    max_memory_mib: float = 256
    latency_budget_us: float | None = None
    operations_per_second: float | None = None
    max_build_operations: int = 200_000_000
    max_model_mechanisms: int = 1_000_000
    max_detectors: int = 100_000

    def __post_init__(self):
        for name in ("max_memory_mib", "latency_budget_us", "operations_per_second"):
            value = getattr(self, name)
            if value is None and name != "max_memory_mib":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_build_operations", "max_model_mechanisms", "max_detectors"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.latency_budget_us is not None and self.operations_per_second is None:
            raise ValueError(
                "latency_budget_us requires operations_per_second for the target platform"
            )


@dataclass(frozen=True)
class RelayBPConfig:
    legs: int = 8
    iterations: int = 30
    solutions: int = 1
    first_memory: float = 0.35
    memory_min: float = -0.254
    memory_max: float = 0.985
    llr_clip: float = 50
    seed: int = 0
    chunk_shots: int = 32
    failure_policy: str = "repair"

    def __post_init__(self):
        for name in ("legs", "iterations", "solutions", "chunk_shots"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.solutions > self.legs:
            raise ValueError("solutions cannot exceed legs")
        for name in ("first_memory", "memory_min", "memory_max", "llr_clip"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.memory_min > self.memory_max or self.llr_clip <= 0:
            raise ValueError("memory_min must be <= memory_max; llr_clip must be positive")
        if self.failure_policy not in ("repair", "raise"):
            raise ValueError("failure_policy must be 'repair' or 'raise'")


@dataclass(frozen=True)
class LUTConfig:
    chunk_shots: int = 4096

    def __post_init__(self):
        if type(self.chunk_shots) is not int or self.chunk_shots < 1:
            raise ValueError("chunk_shots must be a positive integer")


@dataclass(frozen=True)
class ResourceEstimate:
    decoder: str
    stats: ModelStats
    persistent_bytes: int | None
    peak_working_bytes: int | None
    build_operations: int | None
    decode_operations_per_shot: int | None
    estimated_latency_us: float | None
    joint_state_bits: int | None
    accepted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


class ResourceLimitError(ValueError):
    def __init__(self, estimate: ResourceEstimate):
        self.estimate = estimate
        super().__init__(f"{estimate.decoder} resource cap: {'; '.join(estimate.reasons)}")


def estimate_resources(
    context: DecoderContext,
    decoder: str,
    config: RelayBPConfig | LUTConfig | None = None,
    limits: ResourceLimits | None = None,
) -> ResourceEstimate:
    """Scan without expanding a graph/table. Caps are per decoder, excluding harness data.

    Counts are exact after a successful scan; memory/work are conservative models.
    Peak includes compile and configured chunk scratch, not caller-owned batch arrays.
    """
    limits = limits or ResourceLimits()
    if decoder not in ("relay_bp", "lut"):
        raise ValueError("decoder must be 'relay_bp' or 'lut'")
    config = config or (RelayBPConfig() if decoder == "relay_bp" else LUTConfig())
    expected = RelayBPConfig if decoder == "relay_bp" else LUTConfig
    if not isinstance(config, expected):
        raise ValueError(f"{decoder} requires {expected.__name__}")
    reasons = []
    raw_errors = context.dem.num_errors
    if context.num_detectors > limits.max_detectors:
        reasons.append(f"{context.num_detectors} detectors exceed scan cap {limits.max_detectors}")
    if raw_errors > limits.max_model_mechanisms:
        reasons.append(f"{raw_errors} DEM mechanisms exceed scan cap {limits.max_model_mechanisms}")
    if reasons:
        return ResourceEstimate(
            decoder,
            ModelStats(context.num_detectors, context.num_observables, raw_errors, 0, 0),
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            tuple(reasons),
        )
    stats = model_stats(context)
    m, k, n, e, a = (
        stats.detectors,
        stats.observables,
        stats.mechanisms,
        stats.edges,
        stats.observable_edges,
    )
    common = 4096 + 256 * (n + k) + 48 * (e + a) + 16 * m
    state_bits = None
    if decoder == "relay_bp":
        basis = m * (128 + (m + 7) // 8 + (k + 7) // 8) if config.failure_policy == "repair" else 0
        graph = 8 * (12 * e + 8 * n + 8 * m + 4 * a)
        persistent = common + graph + basis + 8 * config.legs * n
        scratch = config.chunk_shots * (128 * e + 64 * n + 16 * m + 16 * k + 256)
        peak = 2 * common + persistent + scratch
        repair_work = m * (3 * m + 3 * k) if basis else 0
        build = n * m * max(1, (m + k + 63) // 64) if basis else n + e
        build += n * config.legs + e + a
        decode = config.legs * config.iterations * (40 * e + 15 * n + 8 * m + 4 * a)
        decode += repair_work + 4 * m + k + 16
    else:
        state_bits = m + k
        if state_bits > 62:
            reasons.append(f"joint state space 2**{state_bits} exceeds 62-bit address cap")
            return ResourceEstimate(
                decoder, stats, None, None, None, None, None, state_bits, False, tuple(reasons)
            )
        states, syndromes = 1 << state_bits, 1 << m
        persistent = syndromes * ((k + 7) // 8 + 1) + 16 * (m + k) + 4096
        build_peak = common + 40 * states + syndromes * (24 + 16 * k)
        scratch = config.chunk_shots * (8 * m + 4 * k + 64)
        peak = max(build_peak, persistent + scratch)
        build = max(1, 4 * n) * states
        decode = 4 * m + (m + 7) // 8 + (k + 7) // 8 + 16
    latency = None
    if limits.operations_per_second is not None:
        latency = decode / limits.operations_per_second * 1e6
        if not math.isfinite(latency):
            reasons.append("estimated latency overflows operation-rate model")
            latency = None
        elif limits.latency_budget_us is not None and latency > limits.latency_budget_us:
            reasons.append(
                f"estimated latency {latency:.6g} us exceeds {limits.latency_budget_us:g} us"
            )
    if peak > limits.max_memory_mib * 1024**2:
        reasons.append(f"peak estimate {peak} bytes exceeds {limits.max_memory_mib:g} MiB")
    if build > limits.max_build_operations:
        reasons.append(f"build estimate {build} operations exceeds {limits.max_build_operations}")
    return ResourceEstimate(
        decoder,
        stats,
        persistent,
        peak,
        build,
        decode,
        latency,
        state_bits,
        not reasons,
        tuple(reasons),
    )


def enforce_resources(context, decoder, config, limits) -> ResourceEstimate:
    estimate = estimate_resources(context, decoder, config, limits)
    if not estimate.accepted:
        raise ResourceLimitError(estimate)
    return estimate
