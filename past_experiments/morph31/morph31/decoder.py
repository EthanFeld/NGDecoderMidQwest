"""Unweighted sparse collision clustering, spanning-forest recovery, and guards.

Only odd, unanchored clusters grow. Even clusters retain their frontiers and
resume if another odd cluster joins them. Each union contributes one forest
edge. A boundary anchors a tree; GF(2) leaf peeling produces a valid correction.
This is a conventional greedy collision decoder, not weighted MWPM.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from .cost_model import Operations
from .graph import BOUNDARY, correction_syndrome
from .sparse_memory import Limits, MemoryTracker


class DecodeFailure(RuntimeError):
    pass


@dataclass
class DecodeResult:
    logical_mask: int
    correction: tuple
    operations: Operations
    memory: dict
    ml_accepted: int = 0


def binary_syndrome(syndrome, count):
    a = np.asarray(syndrome)
    if a.shape != (count,) or np.any((a != 0) & (a != 1)):
        raise ValueError(f"expected a binary syndrome of shape ({count},)")
    return a.astype(np.uint8, copy=True)


class SparseDecoder:
    def __init__(self, graph, limits=None, policy=None):
        self.graph = graph
        self.limits = limits or Limits()
        self.policy = policy

    def decode(self, syndrome):
        start = time.monotonic()
        original = binary_syndrome(syndrome, self.graph.num_detectors)
        ops = Operations()
        ops.memory_reads += (self.graph.num_detectors + 7) // 8
        memory = MemoryTracker(self.graph.num_detectors, self.limits)
        original_count = int(np.count_nonzero(original))
        memory.observe(active=original_count, clusters=0, vertices=0, frontier=0, forest=0)
        pre_correction, accepted = (), 0
        residual = original.copy()
        if self.policy is not None:
            residual, pre_correction, accepted = self.policy.predecode(self.graph, residual, ops)
            if ops.neighbor_requests > self.limits.max_expansions:
                raise DecodeFailure("motif neighbor request limit exceeded")
            if time.monotonic() - start > self.limits.max_seconds:
                raise DecodeFailure("motif time limit exceeded")
            memory.observe(active=original_count, clusters=0, vertices=0, frontier=0,
                           forest=len(pre_correction))
        active = list(map(int, np.flatnonzero(residual)))
        owner = {v: i for i, v in enumerate(active)}
        parents = list(range(len(active)))
        sizes = [1] * len(active)
        parities = [1] * len(active)
        anchors = [False] * len(active)
        frontiers = [deque([(v, 0)]) for v in active]
        schedule = deque(range(len(active)))
        scheduled = set(range(len(active)))
        forest = []
        frontier_count = len(active)
        ops.memory_writes += len(active) * 6
        ops.queue_operations += len(active)

        def observe(recovery=0):
            memory.observe(active=len(active), clusters=len(parents), vertices=len(owner),
                           frontier=frontier_count, forest=len(forest) + len(pre_correction),
                           recovery=recovery)

        def find(root):
            while parents[root] != root:
                ops.find_steps += 1
                ops.memory_reads += 2
                root = parents[root]
            return root

        def odd(root):
            return parities[root] and not anchors[root]

        def enqueue(root):
            if odd(root) and root not in scheduled:
                if not frontiers[root]:
                    raise DecodeFailure("odd component exhausted without a boundary or partner")
                schedule.append(root)
                scheduled.add(root)
                ops.queue_operations += 1

        def merge(a, b):
            if sizes[a] < sizes[b]:
                a, b = b, a
            parents[b] = a
            sizes[a] += sizes[b]
            parities[a] ^= parities[b]
            anchors[a] |= anchors[b]
            # Account for the actual queue moves in this Python reference.
            ops.queue_operations += len(frontiers[b])
            frontiers[a].extend(frontiers[b])
            frontiers[b].clear()
            ops.cluster_merges += 1
            ops.memory_reads += 6
            ops.memory_writes += 5
            return a

        observe()
        while schedule:
            scheduled_root = schedule.popleft()
            scheduled.discard(scheduled_root)
            ops.queue_operations += 1
            root = find(scheduled_root)
            if root != scheduled_root or not odd(root):
                continue
            if not frontiers[root]:
                raise DecodeFailure("odd component has no remaining frontier")
            vertex, cursor = frontiers[root].popleft()
            frontier_count -= 1
            ops.queue_operations += 1
            neighbors, generation_cost = self.graph.neighbors_with_cost(vertex)
            # Adjacent seed defects collide before one-sided boundary growth.
            # This prevents a single pair fault from becoming two boundary paths.
            # The order depends only on the fixed residual input, so saved cursors
            # remain valid when a paused cluster resumes.
            neighbors = sorted(neighbors, key=lambda n: (
                0 if n.vertex != BOUNDARY and residual[n.vertex]
                else 1 if n.vertex == BOUNDARY else 2))
            ops.neighbor_requests += 1
            ops.neighbor_generation += generation_cost + len(neighbors)
            ops.memory_reads += 5 + 2 * len(neighbors)
            if ops.neighbor_requests > self.limits.max_expansions:
                raise DecodeFailure("neighbor request limit exceeded")
            if ops.neighbor_requests % 256 == 0 and time.monotonic() - start > (
                self.limits.max_seconds
            ):
                raise DecodeFailure("per-shot time limit exceeded")
            while cursor < len(neighbors) and odd(root):
                neighbor = neighbors[cursor]
                cursor += 1
                other = neighbor.vertex
                ops.edge_examinations += 1
                ops.memory_reads += 1
                if other == BOUNDARY:
                    anchors[root] = True
                    forest.append(neighbor.edge)
                    ops.memory_writes += 2
                elif other not in owner:
                    owner[other] = root
                    frontiers[root].append((other, 0))
                    frontier_count += 1
                    forest.append(neighbor.edge)
                    ops.memory_writes += 3
                    ops.queue_operations += 1
                else:
                    peer = find(owner[other])
                    if root != peer:
                        forest.append(neighbor.edge)
                        root = merge(root, peer)
            if cursor < len(neighbors):
                frontiers[root].appendleft((vertex, cursor))
                frontier_count += 1
                ops.queue_operations += 1
            enqueue(root)
            observe()

        if any(odd(i) for i in range(len(parents)) if find(i) == i):
            raise DecodeFailure("unsatisfied odd cluster")

        # Growth maps/queues are dead. Release them before recovery allocation.
        recovery_vertices = len(owner)
        owner.clear()
        parents.clear()
        sizes.clear()
        parities.clear()
        anchors.clear()
        frontiers.clear()
        scheduled.clear()
        active.clear()
        memory.observe(active=0, clusters=0, vertices=0, frontier=0,
                       forest=len(forest) + len(pre_correction), recovery=recovery_vertices)
        # XOR of incident edge indices plus degree is enough to peel a forest;
        # no adjacency lists or removed-edge set are needed.
        incidence_xor = defaultdict(int)
        degrees = defaultdict(int)
        for index, edge in enumerate(forest):
            for v in (edge.u, edge.v):
                if v != BOUNDARY:
                    incidence_xor[v] ^= index
                    degrees[v] += 1
                    ops.recovery_operations += 1
        leaves = deque(v for v in degrees if degrees[v] == 1)
        correction = list(pre_correction)
        ops.memory_reads += len(pre_correction)
        while leaves:
            vertex = leaves.popleft()
            ops.queue_operations += 1
            if degrees[vertex] != 1:
                continue
            index = incidence_xor[vertex]
            edge = forest[index]
            other = edge.v if vertex == edge.u else edge.u
            if residual[vertex]:
                correction.append(edge)
                residual[vertex] ^= 1
                if other != BOUNDARY:
                    residual[other] ^= 1
                ops.memory_writes += 2
            degrees[vertex] -= 1
            if other != BOUNDARY:
                degrees[other] -= 1
                incidence_xor[other] ^= index
                if degrees[other] == 1:
                    leaves.append(other)
                    ops.queue_operations += 1
            ops.recovery_operations += 1
            ops.memory_reads += 3
        if np.any(residual):
            raise DecodeFailure("forest recovery left a nonzero syndrome")
        actual, mask = correction_syndrome(correction, self.graph.num_detectors)
        ops.validation_operations += len(correction) * 2 + (self.graph.num_detectors + 7) // 8
        if not np.array_equal(actual, original):
            raise DecodeFailure("boundary(correction) != input syndrome")
        return DecodeResult(mask, tuple(correction), ops, memory.report(), accepted)

    def decode_batch(self, syndromes):
        syndromes = np.asarray(syndromes)
        if syndromes.ndim != 2 or syndromes.shape[1] != self.graph.num_detectors:
            raise ValueError("expected a shots by detectors matrix")
        predictions = np.zeros((len(syndromes), self.graph.num_observables), dtype=np.uint8)
        for i, syndrome in enumerate(syndromes):
            mask = self.decode(syndrome).logical_mask
            predictions[i] = [(mask >> bit) & 1 for bit in range(self.graph.num_observables)]
        return predictions
