#include <algorithm>
#include <cstdint>
#include <deque>
#include <vector>

namespace {
struct Edge {
    int left, right, label, cost;
};

struct Site {
    int parent, size, parity, boundary;
    int rank = 0;
    bool active() const { return parity != 0 && boundary == 0; }
    bool logical() const { return parity != 0 && boundary == 2; }
};

struct Result {
    int label, status, clocks;
};

class Schedule {
    std::deque<int> waiting;
    int members = 0;
public:
    const int budget;
    const bool short_updates;
    int elapsed = 0;
    Schedule(const int limit, bool allow_short) : budget(limit), short_updates(allow_short) {}

    bool step(const int clocks = 1) {
        for (int i = 0; i < clocks; ++i) {
            if (++elapsed >= budget) return false;
            if (members) --members;
            else if (!waiting.empty()) {
                const int job = waiting.front();
                members = short_updates && job <= 2 ? 0 : job;
                waiting.pop_front();
            }
        }
        return true;
    }

    bool push(const int job) {
        while (job && waiting.size() == 4)
            if (!step()) return false;
        if (!step()) return false;
        if (job) waiting.push_back(job);
        return true;
    }

    bool epoch() {
        while (members || !waiting.empty())
            if (!step()) return false;
        return step();
    }
};

int root(const std::vector<Site>& nodes, int index, int& distance) {
    while (nodes[index].parent != index) {
        index = nodes[index].parent;
        ++distance;
    }
    return index;
}

Result decode(int sites, const std::vector<std::vector<Edge>>& by_slice,
              int max_age, int initial_age, const uint8_t* syndrome, int cap, bool pipelined_picker, bool dual_port,
              bool short_updates) {
    const Result timeout{0, 1, cap};
    std::vector<Site> region;
    region.reserve(sites);
    for (int i = 0; i < sites; ++i) region.push_back({i, 1, syndrome[i], 0});
    std::vector<int> age(sites), before(sites);
    for (int j=0; j<sites; ++j) age[j] = syndrome[j] * initial_age;
    Schedule clock(cap, short_updates);
    bool label = false;
    for (;;) {
        if (!clock.epoch()) return timeout;
        before = age;
        bool change = false;
        for (int j = 0; j < sites; ++j) {
            int depth = 0;
            if (!region[root(region, j, depth)].active()) continue;
            change = true;
            age[j] = std::min(max_age, age[j] + 1);
        }
        if (!change) return {int(label), 0, clock.elapsed};

        for (const auto& plane : by_slice) {
            if (!clock.step()) return timeout;  // Rows with no hits stream at one per cycle.
            bool emitted = false;
            for (const Edge& edge : plane) {
                const bool boundary = edge.right < 0;
                const int prev = before[edge.left] + (boundary ? 0 : before[edge.right]);
                const int next = age[edge.left] + (boundary ? 0 : age[edge.right]);
                if (prev >= edge.cost || next < edge.cost) continue;

                if (!emitted && pipelined_picker && !clock.step()) return timeout;
                emitted = true;
                int depth_a = 0, depth_b = 0;
                int a = root(region, edge.left, depth_a);
                int b = boundary ? -1 : root(region, edge.right, depth_b);
                if (!clock.step(1 + std::max(depth_a, depth_b))) return timeout;
                if (a == b) {
                    if (!clock.step()) return timeout;
                    continue;
                }

                const Site r = region[a];
                const Site s = boundary ? Site{-1, 0, 0, 1 << edge.label} : region[b];
                const bool swap = !boundary && (dual_port ? s.rank > r.rank : s.size > r.size);
                const int parent = swap ? b : a;
                const int child = swap ? a : b;
                const int rank = std::max(r.rank, s.rank) + (!boundary && r.rank == s.rank);
                const Site joined{parent, r.size+s.size, r.parity^s.parity, r.boundary|s.boundary, rank};
                int activity_writes = (r.active() != joined.active()) ? r.size : 0;
                if (!boundary && s.active() != joined.active()) activity_writes += s.size;
                if (!clock.push(activity_writes)) return timeout;
                if (joined.boundary == 3) {
                    if (!clock.step()) return timeout;
                    return {0, 2, clock.elapsed};
                }
                label ^= r.logical() ^ s.logical() ^ joined.logical();
                region[parent] = joined;
                if (!boundary) {
                    if (!dual_port && !clock.step()) return timeout;
                    region[child].parent = parent;
                }
            }
            if (emitted && !dual_port && !clock.step()) return timeout;
        }
    }
}
}

extern "C" void simulate_growth_from_age(int vertices, int width, int max_age, int size,
                                         const int32_t* edge_data, int shots, const uint8_t* words,
                                         int cap, int dual_port, int short_updates,
                                         int initial_age, int32_t* output) {
    std::vector<std::vector<Edge>> graph(vertices/width);
    for (int i = 0; i < size; ++i) {
        const int32_t* e = edge_data+4*i;
        graph[std::max(e[0],e[1])/width].push_back({e[0],e[1],e[2],e[3]});
    }
    for (int i = 0; i < shots; ++i) {
        auto out = decode(vertices, graph, max_age, initial_age, words+i*vertices, cap, width <= 32,
                          dual_port != 0, short_updates != 0);
        output[3*i] = out.label;
        output[3*i+1] = out.status;
        output[3*i+2] = out.clocks;
    }
}
