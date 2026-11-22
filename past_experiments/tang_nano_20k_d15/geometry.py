"""Generate and audit the finite detector graph for the shipped circuit."""

import json
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def circuit(p):
    import stim

    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=15, rounds=15,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        after_reset_flip_probability=p, before_measure_flip_probability=p,
    )


def make_geometry():
    c = circuit(.005)
    coords = c.get_detector_coordinates()
    positions = sorted((int(y), int(x)) for x, y, t in coords.values() if t == 0)
    lookup = {(int(t), int(y), int(x)): key for key, (x, y, t) in coords.items()}
    planes = [[lookup[t, y, x] for y, x in positions] for t in range(16)]
    order = [key for plane in planes for key in plane]
    index = {key: i for i, key in enumerate(order)}
    graph = {}
    for inst in c.detector_error_model(flatten_loops=True):
        if inst.type != "error":
            continue
        support = []
        flip = 0
        for target in inst.targets_copy():
            if target.is_relative_detector_id() and target.val in index:
                support.append(index[target.val])
            elif target.is_logical_observable_id():
                flip ^= 1
        if not support:
            if flip:
                raise ValueError("Found a detector-silent single-fault logical mechanism.")
            continue
        if len(support) > 2:
            raise ValueError("Projected single faults must have one or two detector endpoints.")
        key = tuple(sorted(support))
        old_flip, old_p = graph.get(key, (flip, 0))
        if old_flip != flip:
            raise ValueError("One projected footprint spans two logical classes.")
        p = inst.args_copy()[0]
        graph[key] = flip, old_p + p - 2 * old_p * p
    edges, singles = [], []
    for support, (flip, p) in sorted(graph.items()):
        row = [*support, flip, round(1000 * p / .005)]
        (edges if len(support) == 2 else singles).append(row)
    if (c.num_detectors, len(planes), len(positions), len(edges), len(singles)) != (3360, 16, 112, 9195, 256):
        raise ValueError("Unexpected Stim circuit shape or detector graph.")
    return {"detector_count": c.num_detectors, "planes": planes, "positions": positions,
            "edges": edges, "singles": singles}


if __name__ == "__main__":
    data = make_geometry()
    (ROOT / "data/geometry.json").write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print("16 planes, 112 checks per plane, 9,451 distinct projected single-fault supports")
