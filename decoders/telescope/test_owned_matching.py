"""Independent exhaustive oracles for the original in-repository T-join solver."""

import itertools

import numpy as np
import pytest
from owned_backend import OwnedGraph


def parity(n, u, v, chosen):
    result = np.zeros(n, dtype=np.uint8)
    for a, b, used in zip(u, v, chosen, strict=True):
        if used:
            result[a] ^= 1
            if b >= 0:
                result[b] ^= 1
    return result


def enumerate_optima(n, u, v, weights):
    """Enumerate every physical edge subset, with no matching implementation."""
    optima = {}
    feasible = {}
    masks = [(1 << int(a)) ^ ((1 << int(b)) if b >= 0 else 0) for a, b in zip(u, v, strict=True)]
    for bits in itertools.product((0, 1), repeat=len(weights)):
        syndrome, cost = 0, 0.0
        for used, mask, weight in zip(bits, masks, weights, strict=True):
            if used:
                syndrome ^= mask
                cost += float(weight)
        feasible.setdefault(syndrome, []).append((cost, bits))
        if syndrome not in optima or cost < optima[syndrome][0]:
            optima[syndrome] = cost, np.asarray(bits, dtype=np.uint8)
    return optima, feasible


def syndrome_bits(n, mask):
    return np.array([(mask >> i) & 1 for i in range(n)], dtype=np.uint8)


def assert_solution(graph, n, u, v, weights, syndrome, optimum, seed=None):
    selected, cost = graph.solve(syndrome, weights, seed=seed)
    assert np.asarray(selected).shape == (len(weights),)
    assert np.isin(selected, (0, 1)).all()
    np.testing.assert_array_equal(parity(n, u, v, selected), syndrome)
    assert cost == pytest.approx(float(np.dot(selected, weights)), abs=1e-9)
    assert cost == pytest.approx(optimum, abs=1e-9)
    return selected, cost


@pytest.mark.parametrize("seed", [519803, 726115])
def test_random_graphs_all_edge_subsets_and_every_syndrome(seed):
    rng = np.random.default_rng(seed)
    for n in range(2, 8):
        all_edges = [(a, b) for a in range(n) for b in range(a + 1, n)]
        all_edges += [(a, -1) for a in range(n)]
        for trial in range(3):
            count = min(len(all_edges), 3 + 3 * trial)
            edges = [all_edges[i] for i in rng.choice(len(all_edges), count, replace=False)]
            u = np.array([a for a, _ in edges], dtype=np.int32)
            v = np.array([b for _, b in edges], dtype=np.int32)
            weights = rng.choice([-7.0, -2.5, -1.0, 0.0, 0.0, 0.25, 1.0, 4.0], count)
            optima, _ = enumerate_optima(n, u, v, weights)
            graph = OwnedGraph(n, u, v, cache=False)
            for mask in range(1 << n):
                syndrome = syndrome_bits(n, mask)
                if mask in optima:
                    assert_solution(graph, n, u, v, weights, syndrome, optima[mask][0])
                else:
                    with pytest.raises((ValueError, RuntimeError)):
                        graph.solve(syndrome, weights)


@pytest.mark.parametrize("weights", [
    [-1.0, -1.0, -1.0, 5.0],
    [0.0, 0.0, 0.0, 0.0],
    [-4.0, 0.0, 1.0, -2.0],
    [1.25, 1.5, 3.0, 0.5],
])
def test_odd_cycles_negative_edges_and_boundary(weights):
    n, u, v = 3, np.array([0, 1, 2, 0]), np.array([1, 2, 0, -1])
    weights = np.asarray(weights)
    optima, _ = enumerate_optima(n, u, v, weights)
    graph = OwnedGraph(n, u, v, cache=False)
    for mask, (cost, _) in optima.items():
        assert_solution(graph, n, u, v, weights, syndrome_bits(n, mask), cost)


def test_boundary_can_absorb_more_than_one_defect():
    n, u, v = 3, np.arange(3), np.full(3, -1)
    graph = OwnedGraph(n, u, v, cache=False)
    assert_solution(graph, n, u, v, np.array([2.0, 4.0, 7.0]), np.ones(3, np.uint8), 13.0)


def test_parallel_edges_and_negative_cycles():
    n, u, v = 2, np.array([0, 0, 0, 1]), np.array([1, 1, -1, -1])
    weights = np.array([-3.0, -2.0, 4.0, 1.0])
    optima, _ = enumerate_optima(n, u, v, weights)
    graph = OwnedGraph(n, u, v, cache=False)
    for mask, (cost, _) in optima.items():
        assert_solution(graph, n, u, v, weights, syndrome_bits(n, mask), cost)


def test_seeds_and_fpga_hints_never_change_the_optimal_cost():
    n = 4
    u = np.array([0, 1, 2, 3, 0, 0, 1, 2, 3])
    v = np.array([1, 2, 3, 0, 2, -1, -1, -1, -1])
    weights = np.array([1.0, 1.5, 0.0, -0.25, 2.0, 4.0, 3.0, 2.0, 5.0])
    optima, feasible = enumerate_optima(n, u, v, weights)
    graph = OwnedGraph(n, u, v, cache=False)
    for mask, (cost, best) in optima.items():
        syndrome = syndrome_bits(n, mask)
        # A deliberately expensive feasible seed must not become the answer by fiat.
        worst = max(feasible[mask], key=lambda item: item[0])[1]
        for seed in (best, np.array(worst, dtype=np.uint8), np.zeros(len(weights), np.uint8)):
            assert_solution(graph, n, u, v, weights, syndrome, cost, seed=seed)
        for removed in (np.zeros(n, np.uint8), syndrome, syndrome & np.array([1, 0, 1, 0])):
            hint = graph.hint(syndrome, removed, weights)
            np.testing.assert_array_equal(parity(n, u, v, hint), syndrome)
            assert_solution(graph, n, u, v, weights, syndrome, cost, seed=hint)


@pytest.mark.parametrize("base,delta", [
    (1.0, 5e-10), (1.0, 5e-12), (1.0, 1e-15), (1e12, 1.0), (1e-12, 1e-15),
])
def test_near_optimal_seed_is_not_accepted_by_a_loose_cost_tolerance(base, delta):
    # The supplied two-boundary seed is strictly worse than the direct edge.
    graph = OwnedGraph(2, np.array([0, 0, 1]), np.array([1, -1, -1]), cache=False)
    weights = np.array([2 * base, base, base + delta])
    selected, cost = graph.solve(
        np.ones(2, np.uint8), weights, seed=np.array([0, 1, 1], dtype=np.uint8),
    )
    np.testing.assert_array_equal(selected, [1, 0, 0])
    assert cost == 2 * base


def test_disconnected_odd_components_fail_explicitly():
    graph = OwnedGraph(4, np.array([0, 2]), np.array([1, 3]), cache=False)
    weights = np.array([1.0, 2.0])
    for syndrome in (np.array([1, 0, 0, 0]), np.array([1, 0, 1, 0])):
        with pytest.raises((ValueError, RuntimeError)):
            graph.solve(syndrome, weights)
        with pytest.raises((ValueError, RuntimeError)):
            graph.hint(syndrome, syndrome, weights)


def test_route_cache_is_reused_and_invalidated_by_changed_weights():
    # Two odd triangles force a cross-triangle edge, defeating nearest-radius certification.
    n = 6
    edges = [(a, b) for a in range(n) for b in range(a + 1, n)]
    u = np.array([a for a, _ in edges])
    v = np.array([b for _, b in edges])
    weights = np.array([1.0 if a // 3 == b // 3 else 10.0 for a, b in edges])
    graph = OwnedGraph(n, u, v, cache=True)
    syndrome = np.ones(n, np.uint8)
    assert_solution(graph, n, u, v, weights, syndrome, 12.0)
    first = graph.stats()
    assert_solution(graph, n, u, v, weights, syndrome, 12.0)
    second = graph.stats()
    assert second["cache_hits"] > first["cache_hits"]
    assert second["dijkstra_searches"] == first["dijkstra_searches"]
    changed = weights.copy()
    changed[(u == 0) & (v == 3)] = 9.5
    assert_solution(graph, n, u, v, changed, syndrome, 11.5)
    assert graph.stats()["dijkstra_searches"] > second["dijkstra_searches"]


@pytest.mark.parametrize("weights", [np.array([np.nan]), np.array([np.inf]), np.array([-np.inf])])
def test_nonfinite_weights_fail(weights):
    graph = OwnedGraph(1, np.array([0]), np.array([-1]), cache=False)
    with pytest.raises((ValueError, RuntimeError)):
        graph.solve(np.array([1]), weights)


def test_malformed_arrays_are_rejected_before_native_calls():
    graph = OwnedGraph(2, np.array([0, 1]), np.array([-1, -1]), cache=False)
    good_syndrome, good_weights = np.array([1, 0]), np.array([1.0, 2.0])
    for syndrome in (np.array([1]), np.array([1, 0, 1]), np.array([2, 0]), np.array([0.5, 0])):
        with pytest.raises((ValueError, RuntimeError)):
            graph.solve(syndrome, good_weights)
    for weights in (np.array([1.0]), np.array([1.0, 2.0, 3.0])):
        with pytest.raises((ValueError, RuntimeError)):
            graph.solve(good_syndrome, weights)
    for seed in (np.array([1]), np.array([2, 0]), np.array([0.5, 0])):
        with pytest.raises((ValueError, RuntimeError)):
            graph.solve(good_syndrome, good_weights, seed=seed)
    for removed in (np.array([1]), np.array([2, 0]), np.array([0.5, 0])):
        with pytest.raises((ValueError, RuntimeError)):
            graph.hint(good_syndrome, removed, good_weights)


@pytest.mark.parametrize("n,u,v", [
    (2, [0], [2]), (2, [-1], [0]), (2, [0], [0]),
    (2, [0, 1], [-1]), (2, [0], [-2]),
])
def test_malformed_graph_endpoints_fail(n, u, v):
    with pytest.raises((ValueError, RuntimeError)):
        OwnedGraph(n, np.array(u), np.array(v), cache=False)


def two_triangle_reference_graph():
    # Each triangle has odd syndrome. Any completion must cross between them,
    # so the cheap nearest-radius certificate cannot finish these shots.
    u = np.array([0, 1, 0, 3, 4, 3, 0, 1, 2])
    v = np.array([1, 2, 2, 4, 5, 5, 3, 4, 5])
    weights = np.array([1.0] * 6 + [10.0] * 3)
    return 6, u, v, weights


@pytest.mark.parametrize("negative", [False, True])
def test_decreased_weights_reuse_prior_routes_and_match_full_oracle(negative):
    n, u, v, original = two_triangle_reference_graph()
    prior = OwnedGraph(n, u, v, cache=True)
    syndrome = np.ones(n, np.uint8)
    assert_solution(prior, n, u, v, original, syndrome, 12.0)
    changed = np.array([0.75] * 6 + [8.5] * 3)
    if negative:
        changed[0] *= -1
    optima, _ = enumerate_optima(n, u, v, changed)
    reused, fresh = OwnedGraph(n, u, v, cache=False), OwnedGraph(n, u, v, cache=False)
    reused.set_reference(prior)
    try:
        for mask, (cost, _) in optima.items():
            row = syndrome_bits(n, mask)
            assert_solution(reused, n, u, v, changed, row, cost)
            assert_solution(fresh, n, u, v, changed, row, cost)
        assert reused.stats()["incremental_searches"] > 0
        assert fresh.stats()["incremental_searches"] == 0
    finally:
        reused.close()
        fresh.close()
        prior.close()


def test_any_absolute_weight_increase_disables_reference_reuse():
    n, u, v, original = two_triangle_reference_graph()
    prior = OwnedGraph(n, u, v, cache=True)
    syndrome = np.ones(n, np.uint8)
    prior.solve(syndrome, original)
    changed = original.copy()
    changed[-3:] -= 0.5
    changed[0] += 0.5
    optima, _ = enumerate_optima(n, u, v, changed)
    reused = OwnedGraph(n, u, v, cache=False)
    reused.set_reference(prior)
    try:
        assert_solution(reused, n, u, v, changed, syndrome, optima[(1 << n) - 1][0])
        assert reused.stats()["dijkstra_searches"] > 0
        assert reused.stats()["incremental_searches"] == 0
    finally:
        reused.close()
        prior.close()


@pytest.mark.parametrize("operation", ["solve", "hint"])
def test_closed_reference_is_detached_before_native_use(operation):
    n, u, v, original = two_triangle_reference_graph()
    prior = OwnedGraph(n, u, v, cache=True)
    syndrome = np.ones(n, np.uint8)
    prior.solve(syndrome, original)
    reused = OwnedGraph(n, u, v, cache=False)
    reused.set_reference(prior)
    prior.close()
    changed = np.array([0.75] * 6 + [8.5] * 3)
    try:
        if operation == "solve":
            assert_solution(reused, n, u, v, changed, syndrome, 10.0)
        else:
            hint = reused.hint(syndrome, syndrome, changed)
            np.testing.assert_array_equal(parity(n, u, v, hint), syndrome)
        assert reused._reference is None
        assert reused.stats()["incremental_searches"] == 0
    finally:
        reused.close()


def test_reference_rejects_self_closed_and_different_topologies():
    n, u, v, _ = two_triangle_reference_graph()
    graph = OwnedGraph(n, u, v, cache=False)
    reordered = OwnedGraph(n, u[::-1], v[::-1], cache=True)
    different_size = OwnedGraph(n + 1, u, v, cache=True)
    closed = OwnedGraph(n, u, v, cache=True)
    closed.close()
    try:
        for reference in (graph, reordered, different_size):
            with pytest.raises((ValueError, RuntimeError)):
                graph.set_reference(reference)
        with pytest.raises(RuntimeError):
            graph.set_reference(closed)
        assert graph._reference is None
    finally:
        graph.close()
        reordered.close()
        different_size.close()


def test_closed_graph_operations_raise_without_entering_native_code():
    graph = OwnedGraph(1, np.array([0]), np.array([-1]), cache=False)
    graph.close()
    graph.close()
    for operation in (
        lambda: graph.solve(np.ones(1, np.uint8), np.ones(1)),
        lambda: graph.hint(np.ones(1, np.uint8), np.ones(1, np.uint8), np.ones(1)),
        graph.stats,
        lambda: graph.set_reference(None),
    ):
        with pytest.raises(RuntimeError):
            operation()
