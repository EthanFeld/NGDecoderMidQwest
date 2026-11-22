"""Packed adapter for the IBM authors' unmodified native Relay-BP implementation.

Source: https://github.com/trmue/relay, pinned in the optional dependency.
DEM extraction preserves correlated mechanisms. No matching/GF(2) repair is added.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from qecbench import DecoderContext

from .model import FaultModel, ModelStats, model_stats, validate_input
from .resources import ResourceEstimate, ResourceLimitError, ResourceLimits

UPSTREAM_COMMIT = "72341ef85cd45779e6483c338aef7ba85094eac1"


def qecbench_implementation_identity() -> dict:
    """Include external Python and native code in the harness resume identity."""
    import relay_bp

    root = Path(relay_bp.__file__).parent
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in (".py", ".pyd", ".so", ".dll")
    ]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "relay_bp_version": importlib.metadata.version("relay-bp"),
        "upstream_python_native_sha256": digest.hexdigest(),
    }


@dataclass(frozen=True)
class IBMRelayBPConfig:
    """Native defaults, plus bounded adapter chunks and optional strict convergence."""

    pre_iter: int = 80
    num_sets: int = 300
    set_max_iter: int = 60
    gamma0: float = 0.1
    gamma_dist_interval: tuple[float, float] = (-0.24, 0.66)
    stop_nconv: int = 1
    stopping_criterion: str = "nconv"
    seed: int = 0
    precision: str = "f32"
    alpha: float | None = None
    alpha_iteration_scaling_factor: float = 1.0
    min_log_prior_ratio: float | None = None
    chunk_shots: int = 32
    failure_policy: str = "return"
    merge_equivalent_faults: bool = False
    detector_mode: str = "full"

    def __post_init__(self):
        for name in ("pre_iter", "set_max_iter", "stop_nconv", "chunk_shots"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.num_sets) is not int or self.num_sets < 0:
            raise ValueError("num_sets must be a nonnegative integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an integer in [0, 2**64)")
        if self.precision not in ("f32", "f64"):
            raise ValueError("precision must be 'f32' or 'f64'")
        if self.failure_policy not in ("return", "raise"):
            raise ValueError("failure_policy must be 'return' or 'raise'")
        if self.stopping_criterion not in ("nconv", "all", "pre_iter"):
            raise ValueError("stopping_criterion must be 'nconv', 'all', or 'pre_iter'")
        if type(self.merge_equivalent_faults) is not bool:
            raise ValueError("merge_equivalent_faults must be boolean")
        if self.detector_mode not in ("full", "memory_basis"):
            raise ValueError("detector_mode must be 'full' or 'memory_basis'")
        if len(self.gamma_dist_interval) != 2:
            raise ValueError("gamma_dist_interval must contain two finite ordered bounds")
        for value in (self.gamma0, *self.gamma_dist_interval, self.alpha_iteration_scaling_factor):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("memory strengths and alpha scaling must be finite numbers")
        if self.gamma_dist_interval[0] >= self.gamma_dist_interval[1]:
            raise ValueError("gamma_dist_interval lower bound must be less than upper bound")
        for name in ("alpha", "min_log_prior_ratio"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (value < 0 if name == "alpha" else value <= 0)
            ):
                raise ValueError(f"{name} must be finite and valid when specified")

    @property
    def total_legs(self) -> int:
        """The first ordered leg plus num_sets disordered relay legs."""
        return 1 + self.num_sets

    @property
    def max_iterations(self) -> int:
        """Worst case; each successful leg and the solution target can stop early."""
        return self.pre_iter + self.num_sets * self.set_max_iter

    def native_kwargs(self) -> dict:
        result = asdict(self)
        for name in (
            "precision",
            "chunk_shots",
            "failure_policy",
            "merge_equivalent_faults",
            "detector_mode",
        ):
            result.pop(name)
        return result


# Explicit bounded-work comparison profile; separate from upstream defaults above.
FAST_CONFIG = IBMRelayBPConfig(
    pre_iter=30, num_sets=7, set_max_iter=30, gamma0=0.35, gamma_dist_interval=(-0.254, 0.985)
)

# Frozen after independent calibration at (d, p) = (7, .003), (9, .003), (7, .007).
# Discarding the other check basis is an explicit speed/conditioning tradeoff.
TUNED_CONFIG = IBMRelayBPConfig(
    pre_iter=80,
    num_sets=31,
    set_max_iter=60,
    gamma0=0.35,
    gamma_dist_interval=(-0.25, 0.66),
    stop_nconv=3,
    alpha=0.9,
    merge_equivalent_faults=True,
    detector_mode="memory_basis",
)

# Paper search settings, full XYZ graph. These are accuracy comparisons, not
# configurations certified to fit the user's median latency or memory budget.
PAPER_S1_CONFIG = IBMRelayBPConfig(
    pre_iter=80,
    num_sets=300,
    set_max_iter=60,
    gamma0=0.35,
    gamma_dist_interval=(-0.254, 0.985),
    stop_nconv=1,
    merge_equivalent_faults=True,
)
PAPER_S5_CONFIG = IBMRelayBPConfig(
    pre_iter=80,
    num_sets=600,
    set_max_iter=60,
    gamma0=0.35,
    gamma_dist_interval=(-0.254, 0.985),
    stop_nconv=5,
    merge_equivalent_faults=True,
)

# Frozen candidates selected by the independent iteration audit. Keeping these
# separate preserves the earlier latency-profile results and their provenance.
ITERATION_CANDIDATE_CONFIG = replace(TUNED_CONFIG, set_max_iter=120)
LONG_SEARCH_CONFIG = replace(TUNED_CONFIG, num_sets=600, stop_nconv=5)
ACCURACY_CONFIG = PAPER_S5_CONFIG


def estimate_ibm_resources(
    context: DecoderContext,
    config: IBMRelayBPConfig | None = None,
    limits: ResourceLimits | None = None,
) -> ResourceEstimate:
    """Conservative native sparse-graph model; serial native batches, no repair basis.

    Memory excludes interpreter, installed library, caller-owned circuit/DEM and
    full input/output batches. Native allocation constants are planning estimates.
    """
    config, limits = config or IBMRelayBPConfig(), limits or ResourceLimits()
    reasons = []
    if context.num_detectors > limits.max_detectors:
        reasons.append("detector scan cap exceeded")
    if context.dem.num_errors > limits.max_model_mechanisms:
        reasons.append("DEM mechanism scan cap exceeded")
    if reasons:
        stats = ModelStats(
            context.num_detectors, context.num_observables, context.dem.num_errors, 0, 0
        )
        return ResourceEstimate(
            "ibm_relay_bp", stats, None, None, None, None, None, None, False, tuple(reasons)
        )
    stats = model_stats(context)
    m, k, n, e, a = (
        stats.detectors,
        stats.observables,
        stats.mechanisms,
        stats.edges,
        stats.observable_edges,
    )
    model = 4096 + 256 * (n + k) + 48 * (e + a) + 16 * m
    native = 192 * e + 128 * n + 128 * m + 64 * a + 4096
    sparse = 16 * (e + a) + 16 * (m + k + n + 3)
    persistent = native + sparse + 8 * n + m + k + 4096
    scratch = config.chunk_shots * (32 * n + 16 * m + 16 * k + 256)
    peak = max(2 * model + 3 * native + 2 * sparse, persistent + scratch)
    build = 16 * (e + a + n + m + k + 1)
    iterations = config.max_iterations
    decode = iterations * (40 * e + 15 * n + 8 * m + 4 * a) + 4 * m + k + 16
    if peak > limits.max_memory_mib * 1024**2:
        reasons.append(f"peak estimate {peak} bytes exceeds {limits.max_memory_mib:g} MiB")
    if build > limits.max_build_operations:
        reasons.append(f"build estimate {build} operations exceeds {limits.max_build_operations}")
    latency = None
    if limits.operations_per_second is not None:
        latency = decode / limits.operations_per_second * 1e6
        if limits.latency_budget_us is not None and latency > limits.latency_budget_us:
            reasons.append(
                f"estimated latency {latency:.6g} us exceeds {limits.latency_budget_us:g} us"
            )
    return ResourceEstimate(
        "ibm_relay_bp",
        stats,
        persistent,
        peak,
        build,
        decode,
        latency,
        None,
        not reasons,
        tuple(reasons),
    )


def _column_matrix(columns, rows):
    from scipy.sparse import csr_matrix

    row_indices = np.asarray([row for column in columns for row in column], dtype=np.intp)
    column_indices = np.repeat(np.arange(len(columns)), [len(column) for column in columns])
    return csr_matrix(
        (np.ones(len(row_indices), dtype=np.uint8), (row_indices, column_indices)),
        shape=(rows, len(columns)),
        dtype=np.uint8,
    )


def memory_detector_indices(context: DecoderContext) -> np.ndarray:
    """Identify the logical-memory check basis in standard generated Stim circuits.

    Initialization/final readout expose the memory basis; interior checks at those
    same spatial sites measure that basis. Other-basis observations are discarded.
    This opt-in strategy is restricted to the contract's generated memory circuits.
    """
    expected = context.experiment.circuit().without_noise()
    if context.num_observables != 1 or context.circuit.without_noise() != expected:
        raise ValueError("memory_basis mode requires a standard generated surface-memory circuit")
    coordinates = context.circuit.get_detector_coordinates()
    if not coordinates or any(len(value) != 3 for value in coordinates.values()):
        raise ValueError("memory_basis mode requires three-dimensional detector coordinates")
    first = min(value[2] for value in coordinates.values())
    last = max(value[2] for value in coordinates.values())
    sites = {tuple(value[:2]) for value in coordinates.values() if value[2] == first}
    final_sites = {tuple(value[:2]) for value in coordinates.values() if value[2] == last}
    if first != 0 or last != context.experiment.rounds or sites != final_sites:
        raise ValueError("surface-memory boundary coordinates do not identify a unique check basis")
    return np.asarray(
        [i for i, value in coordinates.items() if tuple(value[:2]) in sites], dtype=np.intp
    )


class IBMRelayBPDecoder:
    def __init__(
        self,
        context: DecoderContext,
        config: IBMRelayBPConfig | None = None,
        limits: ResourceLimits | None = None,
    ):
        self.config = config or IBMRelayBPConfig()
        self.resource_estimate = estimate_ibm_resources(context, self.config, limits)
        if not self.resource_estimate.accepted:
            raise ResourceLimitError(self.resource_estimate)
        try:
            import relay_bp
        except ImportError as error:
            raise ImportError(
                "IBM Relay-BP requires the pinned optional dependency: "
                "python -m pip install -e './past_experiments/decoder_testing[ibm-relay]'"
            ) from error
        self.num_detectors = context.num_detectors
        self.num_observables = context.num_observables
        model = FaultModel.from_context(context)
        self.detector_indices = np.arange(self.num_detectors, dtype=np.intp)
        if self.config.detector_mode == "memory_basis":
            self.detector_indices = memory_detector_indices(context)
            model = model.project_detectors(self.detector_indices)
        if self.config.merge_equivalent_faults:
            model = model.merge_equivalent_faults()
        self.detector_offset, self.observable_offset = (
            model.detector_offset,
            model.observable_offset,
        )
        self.error_priors = model.probabilities
        self.check_matrix = _column_matrix(model.detectors, len(self.detector_indices))
        self.observable_matrix = _column_matrix(model.observables, self.num_observables)
        self.runner = None
        if len(self.error_priors):
            cls = (
                relay_bp.RelayDecoderF32
                if self.config.precision == "f32"
                else relay_bp.RelayDecoderF64
            )
            native = cls(
                self.check_matrix, error_priors=self.error_priors, **self.config.native_kwargs()
            )
            self.runner = relay_bp.ObservableDecoderRunner(
                native, self.observable_matrix, include_decode_result=False
            )

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        validate_input(detection_events, self.num_detectors)
        output = np.empty((len(detection_events), (self.num_observables + 7) // 8), dtype=np.uint8)
        for start in range(0, len(detection_events), self.config.chunk_shots):
            stop = start + self.config.chunk_shots
            syndrome = np.unpackbits(
                detection_events[start:stop], axis=1, count=self.num_detectors, bitorder="little"
            )
            if self.config.detector_mode == "memory_basis":
                syndrome = syndrome[:, self.detector_indices]
            syndrome ^= self.detector_offset
            syndrome.flags.writeable = False
            if self.runner is None:
                if np.any(syndrome):
                    raise ValueError("syndrome is outside the detector error model's support")
                predictions = np.zeros((len(syndrome), self.num_observables), dtype=np.uint8)
            elif self.config.failure_policy == "raise":
                errors = self.runner.decode_batch(syndrome, parallel=False, progress_bar=False)
                residual = (self.check_matrix @ errors.T).T % 2
                if np.any(residual != syndrome):
                    raise RuntimeError(
                        "IBM Relay-BP did not converge to a syndrome-consistent result"
                    )
                predictions = np.asarray((self.observable_matrix @ errors.T).T % 2, dtype=np.uint8)
            else:
                predictions = self.runner.decode_observables_batch(
                    syndrome, parallel=False, progress_bar=False
                )
            predictions ^= self.observable_offset
            output[start:stop] = np.packbits(predictions, axis=1, bitorder="little")
        return output


def build_decoder(context: DecoderContext) -> IBMRelayBPDecoder:
    """Use upstream native default search parameters."""
    return IBMRelayBPDecoder(context)


def build_fast_decoder(context: DecoderContext) -> IBMRelayBPDecoder:
    """Use an explicit eight-leg, 30-iteration profile for latency comparisons."""
    return IBMRelayBPDecoder(context, FAST_CONFIG)


def build_tuned_decoder(context: DecoderContext) -> IBMRelayBPDecoder:
    """Use the frozen memory-basis profile; admit using measured median latency."""
    return IBMRelayBPDecoder(context, TUNED_CONFIG)


def build_paper_s1_decoder(context: DecoderContext) -> IBMRelayBPDecoder:
    """Full XYZ graph; 301 total legs, up to 18,080 iterations, one solution."""
    return IBMRelayBPDecoder(context, PAPER_S1_CONFIG)


def build_paper_s5_decoder(context: DecoderContext) -> IBMRelayBPDecoder:
    """Full XYZ graph; 601 total legs, up to 36,080 iterations, five solutions."""
    return IBMRelayBPDecoder(context, PAPER_S5_CONFIG)


def build_iteration_candidate_decoder(context: DecoderContext) -> IBMRelayBPDecoder:
    """Calibrated 120-iteration-leg candidate, retaining the memory-basis graph."""
    return IBMRelayBPDecoder(context, ITERATION_CANDIDATE_CONFIG)


def build_accuracy_decoder(context: DecoderContext) -> IBMRelayBPDecoder:
    """Full XYZ accuracy profile; median latency must be checked separately."""
    return IBMRelayBPDecoder(context, ACCURACY_CONFIG)
