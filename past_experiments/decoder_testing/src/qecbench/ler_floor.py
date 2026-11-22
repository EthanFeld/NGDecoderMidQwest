"""Model-based bounds on the Bayes-optimal logical failure probability.

Exact joint probabilities are used when affordable. Otherwise a genie reveals
all faults except independent logical-complement pairs. More information can
only help decoding, so the remaining ambiguity is a lower bound for every decoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import stim
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


@dataclass(frozen=True)
class FloorOptions:
    exact_bits: int = 18
    max_exact_work: int = 100_000_000
    paths: int = 32
    chain_bins: int = 16384
    memory_mib: float = 256

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("exact_bits", 0, 30),
            ("max_exact_work", 1, 2**63 - 1),
            ("paths", 1, 4096),
            ("chain_bins", 16, 1_000_000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        if (
            isinstance(self.memory_mib, bool)
            or not isinstance(self.memory_mib, (float, int))
            or not math.isfinite(self.memory_mib)
            or self.memory_mib <= 0
        ):
            raise ValueError("memory_mib must be finite and positive")
        if self.chain_bins * 64 > self.memory_mib * 1024**2:
            raise ValueError("memory budget is too small for chain_bins; reduce --chain-bins")


@dataclass(frozen=True)
class Fault:
    probability: float
    detectors: tuple[int, ...]
    logical: int


def read_faults(dem: stim.DetectorErrorModel, memory_mib: float) -> tuple[Fault, ...]:
    """Preserve each entire correlated mechanism, then merge identical actions.

    Deterministic affine offsets and replacing p>1/2 by 1-p only relabel observed
    syndromes/logical classes, leaving the optimal failure probability unchanged.
    """
    # Conservative preflight for Python graph/model working storage. Circuit and
    # DEM objects, decoder internals, and the interpreter are additional memory.
    if dem.num_errors * 1536 > memory_mib * 1024**2:
        raise ValueError("DEM exceeds floor working-memory budget; raise --memory-mib")
    actions: dict[tuple[tuple[int, ...], int], float] = {}
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        probability = float(instruction.args_copy()[0])
        probability = min(probability, 1 - probability)
        if probability == 0:
            continue
        detectors: set[int] = set()
        logical = 0
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors.symmetric_difference_update((target.val,))
            elif target.is_logical_observable_id():
                logical ^= 1 << target.val
            # Separators describe components of the SAME Bernoulli event.
        if not detectors and not logical:
            continue
        key = (tuple(sorted(detectors)), logical)
        log_bias = -math.inf if probability == 0.5 else math.log1p(-2 * probability)
        actions[key] = actions.get(key, 0.0) + log_bias
    return tuple(
        Fault(-0.5 * math.expm1(log_bias), detectors, logical)
        for (detectors, logical), log_bias in actions.items()
    )


def _mask(detectors: tuple[int, ...], positions: dict[int, int]) -> int:
    return sum(1 << positions.setdefault(detector, len(positions)) for detector in detectors)


def exact_joint_ler(
    faults: tuple[Fault, ...], observables: int, options: FloorOptions
) -> tuple[float | None, str]:
    """Sum all logical cosets, compressing redundant syndrome bits over GF(2)."""
    if not faults or observables == 0 or not any(fault.logical for fault in faults):
        return 0.0, "exact"
    if options.exact_bits == 0:
        return None, "disabled"
    basis: dict[int, tuple[int, int]] = {}
    positions: dict[int, int] = {}
    coordinates = []
    budget = int(options.memory_mib * 1024**2)
    for fault in faults:
        residual = _mask(fault.detectors, positions)
        address = 0
        while residual:
            pivot = residual.bit_length() - 1
            if pivot in basis:
                vector, coordinate = basis[pivot]
                residual ^= vector
                address ^= coordinate
            else:
                coordinate = 1 << len(basis)
                basis[pivot] = (residual, coordinate)
                address ^= coordinate
                break
        bits = len(basis) + observables
        if bits > options.exact_bits:
            return None, "state_bit_limit"
        if (1 << bits) * 40 > budget:
            return None, "memory_limit"
        coordinates.append(address)
    rank = len(basis)
    states = 1 << (rank + observables)
    if states * len(faults) > options.max_exact_work:
        return None, "work_limit"
    addresses = np.arange(states, dtype=np.int64)
    mass = np.zeros(states, dtype=np.float64)
    scratch = np.empty_like(mass)
    mass[0] = 1
    for fault, syndrome in zip(faults, coordinates, strict=True):
        action = syndrome | (fault.logical << rank)
        np.take(mass, addresses ^ action, out=scratch)
        scratch *= fault.probability
        mass *= 1 - fault.probability
        mass += scratch
    joint = mass.reshape(1 << observables, 1 << rank)
    # Sum losing classes directly: subtracting total-success loses tiny LERs.
    winners = np.argmax(joint, axis=0)
    joint[winners, np.arange(1 << rank)] = 0
    return float(joint.sum()), "exact"


def complement_pair_risk(probabilities: np.ndarray, bins: int) -> tuple[float, float]:
    """Enclose 1/2 sum_e min(P(e), P(complement(e))) without 2**n enumeration.

    With w_i=log((1-p_i)/p_i), B=product(2 sqrt(p_i(1-p_i))), this is
    B/2 * E_uniform[exp(-abs(sum_i sign_i*w_i)/2)]. Quantizing weights gives
    multiplicative bounds from the triangle inequality, not an empirical fit.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or not len(probabilities):
        raise ValueError("a nonempty vector of probabilities is required")
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities <= 0)
        or np.any(probabilities >= 1)
    ):
        raise ValueError("pair probabilities must lie strictly between zero and one")
    if type(bins) is not int or bins < 16:
        raise ValueError("bins must be an integer >= 16")
    probabilities = np.minimum(probabilities, 1 - probabilities)
    weights = np.log1p(-probabilities) - np.log(probabilities)
    total = float(weights.sum())
    if total == 0:
        return 0.5, 0.5
    step = total / (bins - 1)
    integers = np.floor(weights / step).astype(np.int64)
    quantization_error = float(np.abs(weights - step * integers).sum())
    distribution = np.ones(1, dtype=np.float64)
    for weight in integers:
        weight = int(weight)
        if weight == 0:
            continue
        updated = np.zeros(len(distribution) + weight, dtype=np.float64)
        updated[: len(distribution)] += 0.5 * distribution
        updated[weight:] += 0.5 * distribution
        distribution = updated
    signed = 2 * np.arange(len(distribution)) - int(integers.sum())
    expectation = float(distribution @ np.exp(-0.5 * step * np.abs(signed)))
    # Conservative float64 allowance for convolution, sum, and elementary math.
    # These are numerical analytic bounds, not formal interval-arithmetic proofs.
    guard = 64 * np.finfo(float).eps * (len(probabilities) + len(distribution) + 1)
    log_scale = float(
        np.sum(math.log(2) + 0.5 * (np.log(probabilities) + np.log1p(-probabilities)))
    ) - math.log(2)
    low_expectation = max(0.0, expectation - guard)
    low = (
        math.exp(log_scale + math.log(low_expectation) - quantization_error / 2)
        if low_expectation > 0
        else 0.0
    )
    high = math.exp(log_scale + math.log(min(1.0, expectation + guard)) + quantization_error / 2)
    return max(0.0, low), min(0.5, high)


class _LogicalPathSearch:
    """Native sparse Dijkstra on the two logical sheets; no Python heap hot loop."""

    def __init__(self, adjacency, boundary, bit):
        self.start = 2 * boundary
        self.target = self.start + 1
        nodes = 2 * (boundary + 1)
        columns, weights, edge_ids = [], [], []
        starts = [0]
        for state in range(nodes):
            node, parity = divmod(state, 2)
            for neighbor, logical, edge, weight in adjacency.get(node, ()):
                columns.append(2 * neighbor + (parity ^ ((logical >> bit) & 1)))
                weights.append(weight)
                edge_ids.append(edge)
            starts.append(len(columns))
        self.edge_ids = np.asarray(edge_ids, dtype=np.int32)
        # Direct CSR construction intentionally preserves parallel edges; adding
        # duplicate adjacency weights would incorrectly alter shortest paths.
        self.graph = csr_matrix(
            (
                np.asarray(weights, dtype=float),
                np.asarray(columns, dtype=np.int32),
                np.asarray(starts, dtype=np.int32),
            ),
            shape=(nodes, nodes),
        )
        self.order = np.argsort(self.edge_ids, kind="stable")
        self.sorted_edges = self.edge_ids[self.order]
        self.blocked: set[int] = set()

    def find(self, blocked: set[int]) -> tuple[int, ...] | None:
        for edge in blocked - self.blocked:
            left, right = np.searchsorted(self.sorted_edges, (edge, edge + 1))
            self.graph.data[self.order[left:right]] = math.inf
        self.blocked.update(blocked)
        distances, parents = dijkstra(
            self.graph, directed=True, indices=self.start, return_predecessors=True
        )
        if not math.isfinite(distances[self.target]):
            return None
        support: set[int] = set()
        state = self.target
        while state != self.start:
            parent = int(parents[state])
            left, right = self.graph.indptr[parent : parent + 2]
            positions = np.flatnonzero(self.graph.indices[left:right] == state) + left
            position = positions[np.argmin(self.graph.data[positions])]
            support.symmetric_difference_update((int(self.edge_ids[position]),))
            state = parent
        return tuple(sorted(support))


def _algebraic_witness(faults: tuple[Fault, ...], blocked: set[int], bit: int):
    remaining = [i for i in range(len(faults)) if i not in blocked]
    # General hypergraph fallback is bounded; the graph search scales much farther.
    if len(remaining) > 512:
        return None
    remaining.sort(key=lambda i: -faults[i].probability)
    basis: dict[int, tuple[int, int, int]] = {}
    positions: dict[int, int] = {}
    for position, index in enumerate(remaining):
        vector = _mask(faults[index].detectors, positions)
        logical = (faults[index].logical >> bit) & 1
        support = 1 << position
        while vector:
            pivot = vector.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (vector, logical, support)
                break
            previous, previous_logical, previous_support = basis[pivot]
            vector ^= previous
            logical ^= previous_logical
            support ^= previous_support
        if vector == 0 and logical:
            return tuple(remaining[j] for j in range(len(remaining)) if support & (1 << j))
    return None


def genie_path_bound(
    faults: tuple[Fault, ...], detectors: int, observables: int, options: FloorOptions
) -> dict:
    adjacency: dict[int, list[tuple[int, int, int, float]]] = {}
    active = {
        detector for fault in faults if len(fault.detectors) <= 2 for detector in fault.detectors
    }
    mapping = {detector: i for i, detector in enumerate(sorted(active))}
    boundary = len(mapping)
    for index, fault in enumerate(faults):
        if len(fault.detectors) > 2:
            continue
        nodes = (*(mapping[d] for d in fault.detectors), boundary, boundary)[:2]
        a, b = nodes
        weight = -math.log(2) - 0.5 * (math.log(fault.probability) + math.log1p(-fault.probability))
        weight = max(0.0, weight)
        adjacency.setdefault(a, []).append((b, fault.logical, index, weight))
        adjacency.setdefault(b, []).append((a, fault.logical, index, weight))
    candidates = []
    for bit in range(observables):
        blocked: set[int] = set()
        search = _LogicalPathSearch(adjacency, boundary, bit)
        log_lower_bias = log_upper_bias = 0.0
        paths = []
        for _ in range(options.paths):
            support = search.find(blocked)
            if support is None:
                support = _algebraic_witness(faults, blocked, bit)
            if support is None:
                break
            detector_parity: set[int] = set()
            logical_parity = 0
            for index in support:
                detector_parity.symmetric_difference_update(faults[index].detectors)
                logical_parity ^= faults[index].logical
            if (
                detector_parity
                or not ((logical_parity >> bit) & 1)
                or blocked.intersection(support)
            ):
                raise RuntimeError("invalid or overlapping logical witness")
            probabilities = np.array([faults[i].probability for i in support])
            low, high = complement_pair_risk(probabilities, options.chain_bins)
            # Store biases in log form below; direct 1-product cancellation erases rare LER.
            log_lower_bias += math.log1p(-2 * low) if low < 0.5 else -math.inf
            log_upper_bias += math.log1p(-2 * high) if high < 0.5 else -math.inf
            blocked.update(support)
            paths.append(
                {
                    "mechanisms": list(support),
                    "length": len(support),
                    "pair_ler_low": low,
                    "pair_ler_high": high,
                }
            )
        lower = -0.5 * math.expm1(log_lower_bias)
        upper = -0.5 * math.expm1(log_upper_bias)
        candidates.append({"observable": bit, "lower": lower, "genie_upper": upper, "paths": paths})
    best = max(
        candidates,
        key=lambda c: c["lower"],
        default={
            "observable": None,
            "lower": 0.0,
            "genie_upper": 0.0,
            "paths": [],
        },
    )
    return best


def analyze_dem(dem: stim.DetectorErrorModel, options: FloorOptions | None = None) -> dict:
    """Return an exact floor or a decoder-independent analytic lower bound.

    Genie upper endpoints bound ONLY the easier genie problem, never the full
    code. A full-code upper bound must come from an achievable decoder separately.
    """
    options = options or FloorOptions()
    if dem.num_observables > 8:
        raise ValueError("floor analysis supports at most eight logical observables")
    faults = read_faults(dem, options.memory_mib)
    exact, reason = exact_joint_ler(faults, dem.num_observables, options)
    base = {
        "num_detectors": dem.num_detectors,
        "num_observables": dem.num_observables,
        "mechanisms": len(faults),
        "exact_skip_reason": None if exact is not None else reason,
    }
    if exact is not None:
        return {
            **base,
            "method": "exact_joint",
            "exact_ler": exact,
            "minimum_ler_lower": exact,
            "minimum_ler_upper_model": exact,
            "logical_paths": 0,
            "witnesses": [],
        }
    genie = genie_path_bound(faults, dem.num_detectors, dem.num_observables, options)
    return {
        **base,
        "method": "genie_pairs" if genie["paths"] else "trivial",
        "exact_ler": None,
        "minimum_ler_lower": genie["lower"],
        "minimum_ler_upper_model": 1 - 2.0 ** (-dem.num_observables),
        "logical_paths": len(genie["paths"]),
        "witnesses": genie["paths"],
        "genie_observable": genie["observable"],
        "genie_problem_ler_upper": genie["genie_upper"],
    }
