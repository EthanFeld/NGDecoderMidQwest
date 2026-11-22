// Copyright 2022 PyMatching Contributors
// Licensed under Apache-2.0. See vendor/pymatching_2_4_0/LICENSE.
// Device graph and two-pass edge extraction adapted from PyMatching 2.4.0.

pm::MatchingGraph::MatchingGraph() = default;
pm::MatchingGraph::MatchingGraph(MatchingGraph&&) noexcept = default;
pm::SearchGraph::SearchGraph() = default;
pm::SearchGraph::SearchGraph(SearchGraph&&) noexcept = default;

namespace {

// Each node's adjacency uses boundary slot zero followed by ordinary edges
// in UserGraph insertion order.
struct GraphView {
    int nodes, edges;
    const int* offsets;
    const int* neighbors;
    const int* edge_ids;
    const int* endpoints;
    const int* signed_weights;
    const int* rule_offsets;
    const int* rule_targets;
    const int* rule_weights;
    const uint8_t* logical;
    const int* edge_slots;
};

struct PathEdge {
    int id, from, to;
};

struct Shot {
    const GraphView model;
    pm::Mwpm matcher;
    gpu::vector<pm::DetectorNode*> matching_neighbors;
    gpu::vector<pm::SearchDetectorNode*> search_neighbors;
    gpu::vector<pm::obs_int> observables;
    gpu::vector<pm::weight_int> weights;
    gpu::vector<uint8_t> picked, negative;
    gpu::vector<int> negative_edges, reweighted;
    gpu::vector<size_t> detections;
    gpu::vector<PathEdge> path;

    Shot(const GraphView& graph)
        : model(graph), matching_neighbors(graph.offsets[graph.nodes]),
          search_neighbors(graph.offsets[graph.nodes]),
          observables(graph.offsets[graph.nodes]), weights(graph.offsets[graph.nodes]),
          picked(graph.edges), negative(graph.nodes) {
        auto& flood = matcher.flooder.graph;
        auto& search = matcher.search_flooder.graph;
        flood.num_nodes = search.num_nodes = graph.nodes;
        flood.num_observables = 1;
        flood.nodes.resize(graph.nodes);
        search.nodes.resize(graph.nodes);

        for (int edge = 0; edge < graph.edges; ++edge) {
            if (graph.signed_weights[edge] < 0) {
                negative_edges.push_back(edge);
                int a = graph.endpoints[2 * edge], b = graph.endpoints[2 * edge + 1];
                negative[a] ^= 1;
                if (b >= 0) negative[b] ^= 1;
            }
        }
        for (int node = 0; node < graph.nodes; ++node) {
            int start = graph.offsets[node], end = graph.offsets[node + 1];
            auto& match_node = flood.nodes[node];
            auto& search_node = search.nodes[node];
            for (int k = start; k < end; ++k) {
                int neighbor = graph.neighbors[k], edge = graph.edge_ids[k];
                matching_neighbors[k] = neighbor < 0 ? nullptr : &flood.nodes[neighbor];
                search_neighbors[k] = neighbor < 0 ? nullptr : &search.nodes[neighbor];
                observables[k] = graph.logical[edge];
                weights[k] = gpu::abs(graph.signed_weights[edge]);
            }
            size_t degree = end - start;
            match_node.neighbors.bind(matching_neighbors.data() + start, degree);
            match_node.neighbor_weights.bind(weights.data() + start, degree);
            match_node.neighbor_observables.bind(observables.data() + start, degree);
            search_node.neighbors.bind(search_neighbors.data() + start, degree);
            search_node.neighbor_weights.bind(weights.data() + start, degree);
        }
    }

    void run_pass(const uint8_t* syndrome) {
        auto& flooder = matcher.flooder;
        auto& nodes = flooder.graph.nodes;
        flooder.queue.clear();
        flooder.queue.cur_time = 0;
        detections.clear();
        for (auto& node : nodes) node.reset();
        for (int node = 0; node < model.nodes; ++node) {
            if (syndrome[node]) {
                detections.push_back(node);
                if (!negative[node]) matcher.create_detection_event(&nodes[node]);
            }
        }
        for (int node = 0; node < model.nodes; ++node) {
            if (negative[node] && !syndrome[node]) matcher.create_detection_event(&nodes[node]);
        }
        while (true) {
            auto event = flooder.run_until_next_mwpm_notification();
            if (event.event_type == pm::NO_EVENT) break;
            matcher.process_event(event);
        }
        if (matcher.node_arena.allocated.size() != matcher.node_arena.available.size()) {
            gpu::fail();
        }
    }

    void append_edge(int edge, int from, int to) {
        picked[edge] ^= 1;
        path.push_back({edge, from, to});
    }

    void extract_path() {
        auto& flooder = matcher.flooder;
        auto& search = matcher.search_flooder;
        auto& nodes = flooder.graph.nodes;
        flooder.match_edges.clear();
        auto shatter = [&](size_t i) {
            if (nodes[i].region_that_arrived) {
                matcher.shatter_blossom_and_extract_match_edges(
                    nodes[i].region_that_arrived_top, flooder.match_edges);
            }
        };
        for (auto i : detections) shatter(i);
        for (int i = 0; i < model.nodes; ++i) {
            if (negative[i]) shatter(i);
        }
        if (flooder.region_arena.allocated.size() != flooder.region_arena.available.size()) {
            gpu::fail();
        }
        path.clear();
        for (int edge : negative_edges) {
            append_edge(edge, model.endpoints[2 * edge], model.endpoints[2 * edge + 1]);
        }
        for (const auto& match : flooder.match_edges) {
            size_t from = match.loc_from - nodes.data();
            size_t to = match.loc_to ? match.loc_to - nodes.data() : SIZE_MAX;
            auto* first = search.graph.nodes.data();
            auto collision = search.run_until_collision(first + from, to == SIZE_MAX ? nullptr : first + to);
            if (!collision.detector_node) gpu::fail();
            search.iter_edges_tracing_back_from_collision_edge(
                collision, [&](const pm::SearchGraphEdge& e) {
                    int a = e.detector_node - first;
                    int slot = model.offsets[a] + e.neighbor_index;
                    append_edge(model.edge_ids[slot], a, model.neighbors[slot]);
                });
            search.reset();
        }
        // Match the upstream swap-with-last cancellation and endpoint order.
        for (size_t i = 0; i < path.size();) {
            int edge = path[i].id;
            if (picked[edge]) {
                picked[edge] = 0;
                ++i;
            } else {
                path[i] = path.back();
                path.pop_back();
            }
        }
    }

    void set_weight(int edge, pm::weight_int value) {
        int first = model.edge_slots[2 * edge], second = model.edge_slots[2 * edge + 1];
        weights[first] = value;
        if (second >= 0) weights[second] = value;
    }

    void reweight() {
        for (const auto& chosen : path) {
            int start = model.rule_offsets[chosen.id], end = model.rule_offsets[chosen.id + 1];
            for (int k = start; k < end; ++k) {
                int edge = model.rule_targets[k];
                auto weight = static_cast<pm::weight_int>(model.rule_weights[k]);
                auto old = weights[model.edge_slots[2 * edge]];
                if (weight < old) {
                    if (old == gpu::abs(model.signed_weights[edge])) reweighted.push_back(edge);
                    set_weight(edge, weight);
                }
            }
        }
    }

    void restore_weights() {
        for (int edge : reweighted) {
            set_weight(edge, gpu::abs(model.signed_weights[edge]));
        }
        reweighted.clear();
    }
};

}  // namespace

extern "C" __global__ void correlated_matching(
    GraphView model, const uint8_t* syndromes, int count, uint8_t* scratch,
    size_t scratch_stride, Shot* volatile* workspaces, volatile int* sizes, int* edges, int limit
) {
    unsigned int lane = blockDim.x * blockIdx.x + threadIdx.x;
    if (lane >= count) return;
    sizes[lane] = -1;
    Shot* workspace = workspaces[lane];
    workspaces[lane] = nullptr;  // Recreate the workspace after an early exit.
    if (!workspace) {
        gpu::set_workspace(scratch + lane * scratch_stride, scratch_stride);
        workspace = new (gpu::allocate(sizeof(Shot))) Shot(model);
    }
    // The graph and arenas are reused. Only ephemeral state and the changed
    // weights are reset, keeping consecutive shots independent.
    Shot& shot = *workspace;
    auto* syndrome = syndromes + lane * model.nodes;
    shot.run_pass(syndrome);
    shot.extract_path();
    shot.reweight();
    shot.run_pass(syndrome);
    shot.extract_path();
    shot.restore_weights();
    workspaces[lane] = workspace;
    if (shot.path.size() > limit) return;
    int* out = edges + 2 * lane * limit;
    for (const auto& edge : shot.path) {
        *out++ = edge.from;
        *out++ = edge.to;
    }
    sizes[lane] = shot.path.size();
}
