"""Static CUDA graph data for the PyMatching 2.4.0 two-pass decoder.

Weight construction follows UserGraph in the pinned Apache-2.0 PyMatching source.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pymatching
import stim

_MAX_HALF_WEIGHT = (1 << 24) - 1
_BOUNDARY = -1
_Edge = tuple[int, int]


@dataclass(frozen=True)
class CorrelatedGraph:
    """CSR adjacency, signed integer edge weights, and sparse reweight rules.

    Each directed adjacency slot refers to one undirected edge. The two rows of
    an ordinary edge share its weight and rule IDs. A boundary uses endpoint -1
    and comes first in its detector's adjacency row. Divide the solver's total
    integer weight by `scale` to recover the cost in the original units.
    """

    offsets: np.ndarray
    neighbors: np.ndarray
    edge_ids: np.ndarray
    endpoints: np.ndarray
    weights: np.ndarray
    rule_offsets: np.ndarray
    rule_targets: np.ndarray
    rule_weights: np.ndarray
    logical: np.ndarray
    scale: float

    @property
    def num_detectors(self) -> int:
        return len(self.offsets) - 1


def _bernoulli_xor(p: float, q: float) -> float:
    return p * (1 - q) + q * (1 - p)


def _component(detectors: list[int]) -> _Edge:
    if len(detectors) == 1:
        return detectors[0], _BOUNDARY
    if len(detectors) == 2:
        return min(detectors), max(detectors)
    raise ValueError("correlated matching requires one or two detectors per component")


def _joint_probabilities(dem: stim.DetectorErrorModel) -> dict[_Edge, dict[_Edge, float]]:
    # Match UserGraph's XOR accumulation in flattened DEM instruction order.
    joint = defaultdict(dict)
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        probability = instruction.args_copy()[0]
        if probability == 0:
            continue
        if not 0 < probability <= 0.5:
            raise ValueError("correlated matching requires error probabilities in [0, 0.5]")
        components, detectors = [], []
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors.append(target.val)
            elif target.is_separator():
                components.append(_component(detectors))
                detectors.clear()
        if not detectors and not components:
            continue
        components.append(_component(detectors))

        for left, right in combinations(components, 2):
            joint[left][right] = _bernoulli_xor(joint[left].get(right, 0.0), probability)
            joint[right][left] = _bernoulli_xor(joint[right].get(left, 0.0), probability)
        for edge in components:
            joint[edge][edge] = _bernoulli_xor(joint[edge].get(edge, 0.0), probability)
    return joint


def _quantize(weights: np.ndarray, scale: float) -> np.ndarray:
    # std::round rounds ties away from zero. Double all weights so collisions
    # on the sparse-blossom timeline occur at integer times.
    scaled = weights * scale
    absolute = np.abs(scaled)
    integral = np.floor(absolute)
    rounded = integral + (absolute - integral >= 0.5)
    return (2 * np.copysign(rounded, scaled)).astype(np.int32)


def correlated_graph(
    dem: stim.DetectorErrorModel,
    matching: pymatching.Matching | None = None,
) -> CorrelatedGraph:
    """Preserve UserGraph's edge insertion order and both passes' integer weights."""
    if pymatching.__version__ != "2.4.0":
        raise RuntimeError("the correlated CUDA graph requires PyMatching 2.4.0")
    if matching is None:
        matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    if matching.num_nodes != dem.num_detectors or matching.boundary or dem.num_observables > 1:
        raise ValueError("CUDA correlations require a virtual boundary and at most one observable")

    joint = _joint_probabilities(dem)
    rows = [[] for _ in range(dem.num_detectors)]
    edge_index, endpoint_rows, original, logical = {}, [], [], []
    for edge, (left, right, data) in enumerate(matching.edges()):
        key = (left, _BOUNDARY) if right is None else (min(left, right), max(left, right))
        if left == right:
            raise ValueError("the correlated CUDA graph requires distinct endpoints")
        edge_index[key] = edge
        endpoint_rows.append((left, _BOUNDARY if right is None else right))
        original.append(data["weight"])
        logical.append(int(0 in data["fault_ids"]))
        if right is None:
            rows[left].insert(0, (_BOUNDARY, edge))
        else:
            rows[left].append((right, edge))
            rows[right].append((left, edge))
    if edge_index.keys() != joint.keys():
        raise ValueError("matching topology differs from the correlated detector error model")

    rule_offsets, rule_targets, implied = [0], [], []
    for cause in edge_index:
        chances = joint[cause]
        marginal = chances[cause]
        if marginal:
            # UserGraph orders target edge pairs lexicographically, with a
            # boundary represented by SIZE_MAX rather than a detector number.
            for affected in sorted(chances, key=lambda key: (key[0], key[1] < 0, key[1])):
                if affected == cause:
                    continue
                conditional = min(0.5, chances[affected] / marginal)
                if conditional == 0:
                    raise ValueError("correlated matching received a zero conditional probability")
                target = edge_index[affected]
                weight = math.log((1 - conditional) / conditional)
                if original[target] * weight < 0:
                    raise ValueError("a correlation cannot change an edge weight's sign")
                rule_targets.append(target)
                implied.append(weight)
        rule_offsets.append(len(rule_targets))

    all_weights = np.asarray(original + implied, dtype=np.float64)
    maximum = np.max(np.abs(all_weights), initial=0)
    if not np.isfinite(maximum) or maximum > _MAX_HALF_WEIGHT:
        raise ValueError("edge exceeds the PyMatching weight limit")
    scale = 1.0 if np.all(all_weights == np.trunc(all_weights)) else _MAX_HALF_WEIGHT / maximum
    weights = _quantize(all_weights, scale)
    slots = np.asarray([slot for row in rows for slot in row], dtype=np.int32).reshape(-1, 2)
    return CorrelatedGraph(
        offsets=np.asarray([0, *np.cumsum([len(row) for row in rows])], dtype=np.int32),
        neighbors=np.ascontiguousarray(slots[:, 0]),
        edge_ids=np.ascontiguousarray(slots[:, 1]),
        endpoints=np.asarray(endpoint_rows, dtype=np.int32).reshape(-1, 2),
        weights=weights[: len(original)],
        rule_offsets=np.asarray(rule_offsets, dtype=np.int32),
        rule_targets=np.asarray(rule_targets, dtype=np.int32),
        rule_weights=np.abs(weights[len(original) :]),
        logical=np.asarray(logical, dtype=np.uint8),
        scale=float(2 * scale),
    )
