#pragma once

// Original, standard-library-only rejection check for nearest-radius proofs.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace owned_matching {

// Vertices 0..n-1 are detectors and vertex n is the virtual boundary.
// A zero-cost component containing at least three defects gives every one of
// those defects a zero nearest-neighbor radius. If its defect count is odd,
// every perfect matching must pair one of them outside the component. That
// crossing has strictly positive cost (or is impossible), while the endpoint
// radii contribute at most half that cost. The nearest-radius/2 lower bound
// therefore cannot be attained. Rejecting only this proof changes no solver
// weights, path state, candidate ordering, or returned correction.
inline bool zero_component_obstructs_certificate(
    int n,
    const std::vector<int>& starts,
    const std::vector<int>& targets,
    const std::vector<double>& adjacent_weights,
    const std::vector<int>& defects) {
    if (n < 0 || defects.size() < 3 ||
        starts.size() != static_cast<std::size_t>(n) + 2 ||
        targets.size() != adjacent_weights.size()) return false;

    const auto vertices = static_cast<std::size_t>(n) + 1;
    std::vector<std::uint8_t> is_defect(vertices, 0), seen(vertices, 0);
    for (int vertex : defects) {
        if (vertex < 0 || vertex > n || is_defect[vertex]) return false;
        is_defect[vertex] = 1;
    }
    std::vector<int> queue;
    queue.reserve(defects.size());
    for (int root : defects) {
        if (seen[root]) continue;
        queue.clear();
        queue.push_back(root);
        seen[root] = 1;
        int count = 0;
        for (std::size_t head = 0; head < queue.size(); ++head) {
            const int vertex = queue[head];
            count += is_defect[vertex];
            const int begin = starts[vertex], end = starts[vertex + 1];
            if (begin < 0 || end < begin ||
                static_cast<std::size_t>(end) > targets.size()) return false;
            for (int index = begin; index < end; ++index) {
                if (adjacent_weights[index] != 0) continue;
                const int other = targets[index];
                if (other < 0 || other > n) return false;
                if (!seen[other]) {
                    seen[other] = 1;
                    queue.push_back(other);
                }
            }
        }
        // Counting the entire component is essential: a fourth defect can
        // make an apparent three-defect obstruction disappear.
        if (count >= 3 && (count & 1)) return true;
    }
    return false;
}

}  // namespace owned_matching
