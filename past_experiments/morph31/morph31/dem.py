"""Parse separator components while retaining the complete correlated DEM on disk.

Projection assumption: components separated by ^ are treated as independent.
Only identical (endpoints, logical mask) edges are combined with odd-parity
probability. Logical-distinct parallel edges and their multiplicities survive.
"""

import math
from collections import Counter, defaultdict

import numpy as np

from .graph import BOUNDARY, Edge, ExplicitGraph


def parse_dem(dem):
    groups = defaultdict(Counter)
    errors = correlated = zero_probability = identities = components = 0
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        errors += 1
        probability, = instruction.args_copy()
        if not 0 <= probability <= 1 or not math.isfinite(probability):
            raise ValueError("invalid DEM probability")
        if probability == 0:
            zero_probability += 1
            continue
        parts = [[]]
        for target in instruction.targets_copy():
            if target.is_separator():
                parts.append([])
            else:
                parts[-1].append(target)
        correlated += len(parts) > 1
        # Identical components in the SAME mechanism cancel, not act independently.
        mechanism = Counter()
        for part in parts:
            detectors, mask = set(), 0
            for target in part:
                if target.is_relative_detector_id():
                    d = target.val
                    detectors.symmetric_difference_update((d,))
                elif target.is_logical_observable_id():
                    mask ^= 1 << target.val
                else:
                    raise ValueError(f"unsupported DEM target: {target}")
            if len(detectors) > 2:
                raise ValueError("non-graphlike component; request Stim decompose_errors=True")
            nodes = sorted(detectors)
            if not nodes and not mask:
                identities += 1
                continue
            if len(nodes) == 2:
                u, v = nodes
            elif nodes:
                u, v = BOUNDARY, nodes[0]
            else:
                u = v = BOUNDARY  # Undetectable logical mechanism; retained explicitly.
            mechanism[u, v, mask] += 1
        for key, count in mechanism.items():
            if count % 2:
                groups[key][probability] += 1
                components += 1
    edges = []
    for (u, v, mask), probabilities in sorted(groups.items()):
        parity_bias = math.prod((1 - 2 * p)**n for p, n in sorted(probabilities.items()))
        p = (1 - parity_bias) / 2
        edges.append(Edge(u, v, p, mask, sum(probabilities.values())))
    coordinates = dem.get_detector_coordinates()
    if any(len(coordinates.get(i, ())) != 3 for i in range(dem.num_detectors)):
        raise ValueError("the graph compiler requires all 3D detector coordinates")
    return ExplicitGraph(
        dem.num_detectors, dem.num_observables,
        np.array([coordinates[i] for i in range(dem.num_detectors)]), edges,
        {
            "projection": "independent separator components; identical labels parity-merged",
            "correlations_preserved": False,
            "dem_error_instructions": errors,
            "correlated_error_instructions": correlated,
            "projected_components": components,
            "zero_probability_instructions": zero_probability,
            "identity_components": identities,
            "undetectable_logical_edges": sum(e.u == e.v == BOUNDARY for e in edges),
        },
    )
