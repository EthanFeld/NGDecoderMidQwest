#pragma once

// In-repository implementation of Edmonds' weighted matching algorithm.
// No third-party matching implementation is included or called here.
//
// A primal-dual alternating forest grows over tight edges. Odd cycles are
// contracted into blossoms; zero-dual inner blossoms are expanded. Each
// augmentation increases cardinality by one. Weights are negated costs.
// Vertex duals are unrestricted, as required by perfect matching's equality
// constraints; this forces maximum cardinality even when an augmentation
// temporarily lowers total weight.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace owned_matching {

struct PricingPhase {
    // Vertices index the metric matrix. Colors identify immediate children in
    // the positive-dual laminar tree; unequal colors have this phase as LCA.
    std::vector<int> vertices;
    std::vector<long double> offsets;
    std::vector<int> colors;
};

struct LazyMatchingOptions {
    // The caller has supplied canonical exact distances for a sparse graph.
    // Include every known edge instead of selecting a fixed nearest count.
    bool provided_graph = false;
    std::function<void(int)> grow_nearest;
    // True certifies/resolves every omitted geometric constraint. Returning
    // false conservatively invokes the original vertex-radius oracle.
    std::function<bool(const std::vector<PricingPhase>&)> price_geometry;
};

struct MatchingStats {
    int initial_pairs = 0;
    int hinted_pairs = 0;
    int augmentations = 0;
    int contractions = 0;
    int expansions = 0;
    int sparse_rounds = 0;
    int priced_edges = 0;
    int sparse_edges = 0;
    int dense_fallbacks = 0;
    std::int64_t prepare_ns = 0;
    std::int64_t build_ns = 0;
    std::int64_t solve_ns = 0;
    std::int64_t pricing_ns = 0;
    std::int64_t oracle_ns = 0;
    int radius_queries = 0;
    int full_queries = 0;
    int geometric_queries = 0;
    int geometric_fallbacks = 0;
    int nearest_queries = 0;
};

namespace detail {

using MatchingClock = std::chrono::steady_clock;
inline std::int64_t matching_elapsed(MatchingClock::time_point start,
                                     MatchingClock::time_point finish) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
}

class WeightedBlossom {
    using Real = long double;
    struct Edge {
        std::uint16_t from = 0, to = 0;
    };

    int n_, capacity_, extent_, stamp_ = 0;
    const std::vector<std::vector<double>>& costs_;
    std::vector<Edge> edges_;
    std::vector<Real> dual_;
    std::vector<double> fast_dual_;
    std::vector<int> mate_, top_, parent_, side_, slack_, visited_;
    std::vector<std::vector<int>> petals_, member_;
    std::vector<std::vector<int>> neighbors_;
    const std::vector<std::vector<int>>* borrowed_neighbors_;
    std::vector<int> queue_;
    std::size_t head_ = 0;
    Real tolerance_;
    double fast_tolerance_;
    MatchingStats* stats_;

    Edge& edge(int a, int b) { return edges_[a * capacity_ + b]; }
    const Edge& edge(int a, int b) const { return edges_[a * capacity_ + b]; }
    const std::vector<int>& neighbor_row(int vertex) const {
        return borrowed_neighbors_ ? (*borrowed_neighbors_)[vertex - 1] : neighbors_[vertex];
    }
    Real reduced(const Edge& e) const {
        return dual_[e.from] + dual_[e.to] + 2 * static_cast<Real>(costs_[e.from - 1][e.to - 1]);
    }
    double fast_reduced(const Edge& e) const {
        return fast_dual_[e.from] + fast_dual_[e.to] + 2 * costs_[e.from - 1][e.to - 1];
    }
    bool tight(const Edge& e) const {
        if (fast_reduced(e) > fast_tolerance_) return false;
        return reduced(e) <= tolerance_;
    }
    bool less_reduced(const Edge& a, const Edge& b) const {
        const double difference = fast_reduced(a) - fast_reduced(b);
        if (std::abs(difference) > 2 * fast_tolerance_) return difference < 0;
        return reduced(a) < reduced(b);
    }

    void enqueue(int b) {
        if (b <= n_) queue_.push_back(b);
        else for (int p : petals_[b]) enqueue(p);
    }
    void assign_top(int b, int representative) {
        top_[b] = representative;
        if (b > n_) for (int p : petals_[b]) assign_top(p, representative);
    }
    void offer_slack(int u, int b) {
        if (!slack_[b] || less_reduced(edge(u, b), edge(slack_[b], b)))
            slack_[b] = u;
    }
    void recompute_slack(int b) {
        slack_[b] = 0;
        for (int u = 1; u <= n_; ++u)
            if (edge(u, b).from && top_[u] != b && side_[top_[u]] == 0)
                offer_slack(u, b);
    }

    // Orient a blossom so that the path from its base to p has even length.
    int orient(int b, int p) {
        auto& cycle = petals_[b];
        auto found = std::find(cycle.begin(), cycle.end(), p);
        if (found == cycle.end()) throw std::logic_error("invalid blossom member");
        int position = static_cast<int>(found - cycle.begin());
        if (position & 1) {
            std::reverse(cycle.begin() + 1, cycle.end());
            position = static_cast<int>(cycle.size()) - position;
        }
        return position;
    }
    void install_match(int b, int other) {
        const Edge e = edge(b, other);
        mate_[b] = e.to;
        if (b <= n_) return;
        const int p = member_[b][e.from];
        const int position = orient(b, p);
        auto& cycle = petals_[b];
        for (int i = 0; i < position; ++i)
            install_match(cycle[i], cycle[i ^ 1]);
        install_match(p, other);
        std::rotate(cycle.begin(), cycle.begin() + position, cycle.end());
    }
    void augment(int b, int other) {
        for (;;) {
            const int previous = top_[mate_[b]];
            install_match(b, other);
            if (!previous) return;
            const int next = top_[parent_[previous]];
            install_match(previous, next);
            b = next;
            other = previous;
        }
    }
    int common_base(int a, int b) {
        ++stamp_;
        while (a || b) {
            if (a) {
                if (visited_[a] == stamp_) return a;
                visited_[a] = stamp_;
                a = top_[mate_[a]];
                if (a) a = top_[parent_[a]];
            }
            std::swap(a, b);
        }
        return 0;
    }
    void contract(int a, int base, int b) {
        if (stats_) ++stats_->contractions;
        int blossom = n_ + 1;
        while (blossom <= extent_ && top_[blossom]) ++blossom;
        if (blossom > extent_) ++extent_;
        if (blossom >= capacity_) throw std::logic_error("too many blossoms");
        dual_[blossom] = 0;
        side_[blossom] = 0;
        mate_[blossom] = mate_[base];
        auto& cycle = petals_[blossom];
        cycle.clear();
        cycle.push_back(base);
        for (int x = a; x != base;) {
            cycle.push_back(x);
            int y = top_[mate_[x]];
            cycle.push_back(y);
            enqueue(y);
            x = top_[parent_[y]];
        }
        std::reverse(cycle.begin() + 1, cycle.end());
        for (int x = b; x != base;) {
            cycle.push_back(x);
            int y = top_[mate_[x]];
            cycle.push_back(y);
            enqueue(y);
            x = top_[parent_[y]];
        }
        assign_top(blossom, blossom);
        for (int x = 1; x <= extent_; ++x) {
            edge(blossom, x) = Edge{};
            edge(x, blossom) = Edge{};
        }
        // Original vertices have implicit singleton membership. Allocate a
        // dense membership row only when an actual blossom is constructed.
        member_[blossom].assign(n_ + 1, 0);
        for (int p : cycle) {
            for (int x = 1; x <= extent_; ++x) {
                if (edge(p, x).from &&
                    (!edge(blossom, x).from ||
                     less_reduced(edge(p, x), edge(blossom, x)))) {
                    edge(blossom, x) = edge(p, x);
                    edge(x, blossom) = edge(x, p);
                }
            }
            if (p <= n_) member_[blossom][p] = p;
            else for (int v = 1; v <= n_; ++v)
                if (member_[p][v]) member_[blossom][v] = p;
        }
        recompute_slack(blossom);
    }
    void expand(int blossom) {
        if (stats_) ++stats_->expansions;
        auto& cycle = petals_[blossom];
        for (int p : cycle) assign_top(p, p);
        const int entry = member_[blossom][edge(blossom, parent_[blossom]).from];
        const int position = orient(blossom, entry);
        for (int i = 0; i < position; i += 2) {
            const int inner = cycle[i], outer = cycle[i + 1];
            parent_[inner] = edge(outer, inner).from;
            side_[inner] = 1;
            side_[outer] = 0;
            slack_[inner] = 0;
            recompute_slack(outer);
            enqueue(outer);
        }
        side_[entry] = 1;
        parent_[entry] = parent_[blossom];
        for (std::size_t i = position + 1; i < cycle.size(); ++i) {
            side_[cycle[i]] = -1;
            recompute_slack(cycle[i]);
        }
        top_[blossom] = 0;
    }
    bool encounter(const Edge& e) {
        const int a = top_[e.from], b = top_[e.to];
        if (a == b) return false;
        if (side_[b] == -1) {
            parent_[b] = e.from;
            side_[b] = 1;
            const int next = top_[mate_[b]];
            if (!next) throw std::logic_error("unmatched vertex outside matching forest");
            slack_[b] = slack_[next] = 0;
            side_[next] = 0;
            enqueue(next);
        } else if (side_[b] == 0) {
            const int base = common_base(a, b);
            if (!base) {
                augment(a, b);
                augment(b, a);
                return true;
            }
            contract(a, base, b);
        }
        return false;
    }
    bool augment_once() {
        std::fill(side_.begin(), side_.end(), -1);
        std::fill(slack_.begin(), slack_.end(), 0);
        queue_.clear();
        head_ = 0;
        for (int b = 1; b <= extent_; ++b)
            if (top_[b] == b && !mate_[b]) {
                parent_[b] = 0;
                side_[b] = 0;
                enqueue(b);
            }
        if (queue_.empty()) return false;
        for (;;) {
            while (head_ < queue_.size()) {
                const int u = queue_[head_++];
                if (side_[top_[u]] == 1) continue;
                // Both endpoints in this scan are original vertices. Read the
                // caller's contiguous cost row directly instead of loading an
                // aggregate edge record and then looking its weight up again.
                const double* row = costs_[u - 1].data();
                const double source_dual = fast_dual_[u];
                for (int v : neighbor_row(u)) {
                    if (top_[u] == top_[v]) continue;
                    const double approximate = source_dual + fast_dual_[v] + 2 * row[v - 1];
                    if (!(approximate > fast_tolerance_) &&
                        dual_[u] + dual_[v] + 2 * static_cast<Real>(row[v - 1]) <= tolerance_) {
                        const Edge e{static_cast<std::uint16_t>(u), static_cast<std::uint16_t>(v)};
                        if (encounter(e)) return true;
                    } else {
                        offer_slack(u, top_[v]);
                    }
                }
            }
            Real step = std::numeric_limits<Real>::infinity();
            for (int b = n_ + 1; b <= extent_; ++b)
                if (top_[b] == b && side_[b] == 1)
                    step = std::min(step, dual_[b] / 2);
            for (int b = 1; b <= extent_; ++b)
                if (top_[b] == b && slack_[b]) {
                    const Real gap = reduced(edge(slack_[b], b));
                    if (side_[b] == -1) step = std::min(step, gap);
                    else if (side_[b] == 0) step = std::min(step, gap / 2);
                }
            if (!std::isfinite(step)) return false;
            step = std::max<Real>(0, step);
            for (int v = 1; v <= n_; ++v) {
                if (side_[top_[v]] == 0) {
                    dual_[v] -= step;
                } else if (side_[top_[v]] == 1) dual_[v] += step;
                fast_dual_[v] = static_cast<double>(dual_[v]);
            }
            for (int b = n_ + 1; b <= extent_; ++b)
                if (top_[b] == b) {
                    if (side_[b] == 0) dual_[b] += 2 * step;
                    else if (side_[b] == 1) dual_[b] -= 2 * step;
                }
            queue_.clear();
            head_ = 0;
            for (int b = 1; b <= extent_; ++b)
                if (top_[b] == b && slack_[b] && top_[slack_[b]] != b &&
                    tight(edge(slack_[b], b)))
                    if (encounter(edge(slack_[b], b))) return true;
            for (int b = n_ + 1; b <= extent_; ++b)
                if (top_[b] == b && side_[b] == 1 && dual_[b] <= tolerance_)
                    expand(b);
        }
    }

public:
    explicit WeightedBlossom(const std::vector<std::vector<double>>& costs,
                             const std::vector<int>* preferred,
                             MatchingStats* stats,
                             const std::vector<std::vector<int>>* sparse_neighbors = nullptr)
        : n_(static_cast<int>(costs.size())), capacity_(2 * n_ + 1), extent_(n_), costs_(costs),
          edges_(static_cast<std::size_t>(capacity_) * capacity_),
          dual_(capacity_), fast_dual_(capacity_), mate_(capacity_), top_(capacity_), parent_(capacity_),
          side_(capacity_, -1), slack_(capacity_), visited_(capacity_),
          petals_(capacity_), member_(capacity_),
          neighbors_(sparse_neighbors ? 0 : n_ + 1),
          borrowed_neighbors_(sparse_neighbors), stats_(stats) {
        if (stats_) *stats_ = MatchingStats{};
        if (n_ & 1) throw std::invalid_argument("perfect matching needs an even vertex count");
        Real maximum = 0;
        for (int a = 0; a < n_; ++a) {
            if (costs[a].size() != static_cast<std::size_t>(n_))
                throw std::invalid_argument("matching cost matrix is not square");
            for (int b = 0; b < a; ++b) {
                double x = costs[a][b], y = costs[b][a];
                if (std::isnan(x) || std::isnan(y) || x < 0 || y < 0 ||
                    (std::isfinite(x) != std::isfinite(y)) ||
                    (std::isfinite(x) && std::abs(x - y) >
                        1e-12 * std::max({1.0, std::abs(x), std::abs(y)})))
                    throw std::invalid_argument("matching costs must be symmetric and nonnegative");
                if (std::isfinite(x)) maximum = std::max(maximum, static_cast<Real>(x));
            }
        }
        tolerance_ = 16 * std::numeric_limits<Real>::epsilon() * std::max<Real>(1, maximum) * (n_ + 1);
        fast_tolerance_ = static_cast<double>(16 * std::numeric_limits<double>::epsilon() *
            std::max<Real>(1, maximum) * (n_ + 1));
        for (int a = 1; a <= n_; ++a) {
            top_[a] = a;
            Real largest = -std::numeric_limits<Real>::infinity();
            if (!sparse_neighbors) for (int b = 1; b <= n_; ++b)
                if (a != b && std::isfinite(costs[a - 1][b - 1])) neighbors_[a].push_back(b);
            for (int b : neighbor_row(a)) {
                    const Real weight = -costs[a - 1][b - 1];
                    edge(a, b) = Edge{static_cast<std::uint16_t>(a), static_cast<std::uint16_t>(b)};
                    largest = std::max(largest, weight);
            }
            dual_[a] = std::isfinite(largest) ? largest : 0;
            fast_dual_[a] = static_cast<double>(dual_[a]);
        }
        // A matching consisting only of tight edges is complementary to the
        // feasible initial duals. Prefer FPGA-suggested pairs only when that
        // condition certifies that they cannot invalidate optimality.
        if (preferred && preferred->size() != static_cast<std::size_t>(n_))
            throw std::invalid_argument("preferred matching has incorrect size");
        if (preferred) for (int a = 1; a <= n_; ++a) {
            const int b = (*preferred)[a - 1] + 1;
            if (b > 0 && b <= n_ && a != b && !mate_[a] && !mate_[b] &&
                edge(a, b).from && tight(edge(a, b))) {
                mate_[a] = b;
                mate_[b] = a;
                if (stats_) ++stats_->hinted_pairs;
            }
        }
        for (int a = 1; a <= n_; ++a) if (!mate_[a])
            for (int b = a + 1; b <= n_; ++b)
                if (!mate_[b] && edge(a, b).from && tight(edge(a, b))) {
                    mate_[a] = b;
                    mate_[b] = a;
                    break;
                }
        if (stats_) for (int a = 1; a <= n_; ++a)
            if (mate_[a] > a) ++stats_->initial_pairs;
        queue_.reserve(static_cast<std::size_t>(n_) * 4);
    }

    // An omitted edge is safe only when it satisfies the *full* matching
    // dual, including every positive-dual odd set containing both endpoints.
    // Endpoint labels alone are insufficient after a blossom contracts.
    std::vector<double> vertex_radii() const {
        // Let r_i=max(0,-lab_i/2). An unknown metric edge with distance
        // d_ij>r_i+r_j has positive reduced cost because common odd-set duals
        // are nonnegative. Inflate the bound to include the full numerical
        // uncertainty band used by price_omitted, then round outwards.
        const Real uncertainty = 512 * std::numeric_limits<Real>::epsilon() * (n_ + 1);
        for (int b = n_ + 1; b <= extent_; ++b)
            if (top_[b] && dual_[b] < -4 * tolerance_) return {};
        std::vector<double> result(n_);
        for (int a = 1; a <= n_; ++a) {
            const Real exact = (std::max<Real>(0, -dual_[a] / 2) +
                uncertainty * (Real(.5) + std::abs(dual_[a])) / 2) / (1 - uncertainty);
            double rounded = static_cast<double>(exact);
            if (static_cast<Real>(rounded) < exact)
                rounded = std::nextafter(rounded, std::numeric_limits<double>::infinity());
            result[a - 1] = std::nextafter(rounded, std::numeric_limits<double>::infinity());
        }
        return result;
    }

    bool pricing_phases(std::vector<PricingPhase>& phases) const {
        phases.clear();
        struct PositiveSet {
            int id, parent = -1;
            Real prefix = 0;
            std::vector<int> vertices;
        };
        std::vector<PositiveSet> sets;
        for (int b = n_ + 1; b <= extent_; ++b) {
            if (!top_[b]) continue;
            if (dual_[b] < -4 * tolerance_) return false;
            if (dual_[b] <= 0) continue;
            PositiveSet current{b, -1, 0, {}};
            for (int v = 1; v <= n_; ++v)
                if (member_[b][v]) current.vertices.push_back(v - 1);
            if (!(current.vertices.size() & 1)) return false;
            sets.push_back(std::move(current));
        }
        // Active odd sets are laminar. The smallest larger set containing one
        // member is therefore the unique immediate positive-dual ancestor.
        for (std::size_t i = 0; i < sets.size(); ++i) {
            const int sample = sets[i].vertices.front() + 1;
            for (std::size_t j = 0; j < sets.size(); ++j)
                if (sets[j].vertices.size() > sets[i].vertices.size() &&
                    member_[sets[j].id][sample] &&
                    (sets[i].parent < 0 || sets[j].vertices.size() <
                     sets[sets[i].parent].vertices.size())) sets[i].parent = static_cast<int>(j);
        }
        for (auto& set : sets) {
            set.prefix = dual_[set.id];
            for (int ancestor = set.parent; ancestor >= 0; ancestor = sets[ancestor].parent)
                set.prefix += dual_[sets[ancestor].id];
        }
        const Real error_fraction = 512 * std::numeric_limits<Real>::epsilon() * (n_ + 1);
        auto emit = [&](const std::vector<int>& vertices, int set_index, Real prefix) {
            PricingPhase phase;
            phase.vertices = vertices;
            phase.colors.reserve(vertices.size());
            for (int v : vertices) phase.colors.push_back(v + 1);
            for (const auto& child : sets) if (child.parent == set_index)
                for (std::size_t i = 0; i < vertices.size(); ++i)
                    if (member_[child.id][vertices[i] + 1]) phase.colors[i] = child.id;
            bool omitted = false;
            for (std::size_t i = 0; i < vertices.size() && !omitted; ++i)
                for (std::size_t j = i + 1; j < vertices.size(); ++j)
                    if (phase.colors[i] != phase.colors[j] &&
                        !std::isfinite(costs_[vertices[i]][vertices[j]])) {
                        omitted = true;
                        break;
                    }
            if (!omitted) return;
            phase.offsets.reserve(vertices.size());
            for (int v : vertices) {
                const Real label = dual_[v + 1];
                const Real exact = label / 2 + prefix / 4;
                // Algebraically eliminate the distance term in the original
                // all-edge pricing error band. Every possibly nonpositive or
                // uncertain full slack has d_ij+offset_i+offset_j <= 0.
                const Real guard = error_fraction * (Real(.5) + std::abs(label) + prefix / 2) / 2;
                const Real lower = (exact - guard) / (1 - error_fraction);
                phase.offsets.push_back(std::nextafter(lower, -std::numeric_limits<Real>::infinity()));
            }
            phases.push_back(std::move(phase));
        };
        std::vector<int> universe(n_);
        for (int v = 0; v < n_; ++v) universe[v] = v;
        emit(universe, -1, 0);
        for (std::size_t i = 0; i < sets.size(); ++i)
            emit(sets[i].vertices, static_cast<int>(i), sets[i].prefix);
        return true;
    }

    bool price_omitted(std::vector<std::pair<int, int>>& additions,
                       int* priced = nullptr) const {
        additions.clear();
        if (priced) *priced = 0;
        std::vector<Real> common(static_cast<std::size_t>(n_) * n_, 0);
        Real dual_objective = 0;
        for (int a = 1; a <= n_; ++a) dual_objective += dual_[a] / 2;
        for (int b = n_ + 1; b <= extent_; ++b) {
            if (top_[b] && dual_[b] < -4 * tolerance_) return false;
            if (!top_[b] || dual_[b] <= 0) continue;
            std::vector<int> vertices;
            for (int a = 1; a <= n_; ++a) if (member_[b][a]) vertices.push_back(a - 1);
            if (!(vertices.size() & 1)) return false;
            int internal_pairs = 0;
            for (int a : vertices) if (mate_[a + 1] > a + 1 &&
                                       member_[b][mate_[a + 1]]) ++internal_pairs;
            if (internal_pairs != static_cast<int>(vertices.size() / 2)) return false;
            dual_objective += static_cast<Real>(vertices.size() - 1) * dual_[b] / 4;
            for (std::size_t i = 0; i < vertices.size(); ++i)
                for (std::size_t j = i + 1; j < vertices.size(); ++j)
                    common[static_cast<std::size_t>(vertices[i]) * n_ + vertices[j]] += dual_[b];
        }
        Real primal_objective = 0, objective_scale = 1;
        for (int a = 0; a < n_; ++a)
            if (mate_[a + 1] > a + 1) {
                const Real value = costs_[a][mate_[a + 1] - 1];
                primal_objective -= value;
                objective_scale += std::abs(value);
            }
        const Real objective_error = 512 * std::numeric_limits<Real>::epsilon() *
            (n_ + 1) * (objective_scale + std::abs(dual_objective));
        if (std::abs(primal_objective - dual_objective) > objective_error) return false;
        for (int a = 0; a < n_; ++a) for (int b = a + 1; b < n_; ++b) {
            if (!std::isfinite(costs_[a][b])) continue;
            if (priced) ++*priced;
            const Real z = common[static_cast<std::size_t>(a) * n_ + b];
            const Real slack = dual_[a + 1] + dual_[b + 1] + 2 * static_cast<Real>(costs_[a][b]) + z;
            const Real error = 128 * std::numeric_limits<Real>::epsilon() * (n_ + 1) *
                (1 + std::abs(dual_[a + 1]) + std::abs(dual_[b + 1]) +
                 2 * std::abs(static_cast<Real>(costs_[a][b])) + std::abs(z));
            if (edge(a + 1, b + 1).from) {
                if (slack < -error) return false;
            } else if (slack <= error) additions.emplace_back(a, b);
        }
        return true;
    }
    std::vector<int> solve() {
        int pairs = 0;
        for (int v = 1; v <= n_; ++v) if (mate_[v] > v) ++pairs;
        while (augment_once()) {
            ++pairs;
            if (stats_) ++stats_->augmentations;
        }
        if (pairs * 2 != n_) throw std::runtime_error("graph has no perfect matching");
        std::vector<int> result(n_);
        for (int v = 1; v <= n_; ++v) {
            if (mate_[v] <= 0 || mate_[v] > n_ || mate_[mate_[v]] != v)
                throw std::logic_error("invalid blossom matching result");
            result[v - 1] = mate_[v] - 1;
        }
        return result;
    }
};

}  // namespace detail

inline std::vector<int> minimum_weight_perfect_matching(
    const std::vector<std::vector<double>>& costs,
    const std::vector<int>* preferred = nullptr,
    MatchingStats* stats = nullptr) {
    if (costs.size() >= std::numeric_limits<std::uint16_t>::max())
        throw std::length_error("dense matching supports fewer than 65535 vertices");
    const auto before = detail::MatchingClock::now();
    detail::WeightedBlossom solver(costs, preferred, stats);
    const auto built = detail::MatchingClock::now();
    auto result = solver.solve();
    if (stats) {
        stats->build_ns = detail::matching_elapsed(before, built);
        stats->solve_ns = detail::matching_elapsed(built, detail::MatchingClock::now());
    }
    return result;
}

// Sparse candidate edges accelerate the alternating forest, but are never an
// approximation: the returned solution must pass pricing against *every*
// finite edge of the original graph and a primal/dual objective check. Any
// uncertain omitted constraint is inserted and the optimization is repeated.
// Failure to establish the certificate invokes the full dense solver.
namespace detail {

inline std::vector<int> sparse_matching_impl(
    const std::vector<std::vector<double>>& costs,
    const std::vector<int>* preferred,
    MatchingStats* stats,
    int nearest_count,
    const std::function<void(const std::vector<double>&)>* ensure_edges,
    const LazyMatchingOptions* options = nullptr) {
    const int n = static_cast<int>(costs.size());
    std::int64_t oracle_ns = 0;
    int radius_queries = 0, full_queries = 0;
    int geometric_queries = 0, geometric_fallbacks = 0, nearest_queries = 0;
    const bool provided_graph = options && options->provided_graph;
    auto query = [&](const std::vector<double>& radii) {
        if (!ensure_edges) return;
        const auto started = MatchingClock::now();
        (*ensure_edges)(radii);
        oracle_ns += matching_elapsed(started, MatchingClock::now());
        if (radii.empty()) ++full_queries;
        else ++radius_queries;
    };
    if (!provided_graph && (n < 8 || nearest_count <= 0 || nearest_count >= n - 1)) {
        query({});
        auto result = minimum_weight_perfect_matching(costs, preferred, stats);
        if (stats) {
            stats->oracle_ns = oracle_ns;
            stats->full_queries = full_queries;
        }
        return result;
    }
    if (costs.size() >= std::numeric_limits<std::uint16_t>::max())
        throw std::length_error("dense matching supports fewer than 65535 vertices");
    // Validate before looking up any off-diagonal input cell.
    for (const auto& row : costs)
        if (row.size() != costs.size()) throw std::invalid_argument("matching cost matrix is not square");
    if (preferred && preferred->size() != costs.size())
        throw std::invalid_argument("preferred matching has incorrect size");
    auto prepare_started = detail::MatchingClock::now();
    std::vector<unsigned char> included(static_cast<std::size_t>(n) * n);
    auto include = [&](int a, int b) {
        included[static_cast<std::size_t>(a) * n + b] = 1;
        included[static_cast<std::size_t>(b) * n + a] = 1;
    };
    auto add_nearest = [&](int count) {
        std::vector<int> candidates;
        candidates.reserve(n);
        for (int a = 0; a < n; ++a) {
            candidates.clear();
            for (int b = 0; b < n; ++b)
                if (a != b && std::isfinite(costs[a][b])) candidates.push_back(b);
            const int take = std::min<int>(count, candidates.size());
            auto before = [&](int x, int y) {
                return costs[a][x] < costs[a][y] ||
                    (costs[a][x] == costs[a][y] && x < y);
            };
            if (take < static_cast<int>(candidates.size()))
                std::nth_element(candidates.begin(), candidates.begin() + take, candidates.end(), before);
            for (int i = 0; i < take; ++i) include(a, candidates[i]);
        }
    };
    auto include_known = [&] {
        for (int a = 0; a < n; ++a) for (int b = a + 1; b < n; ++b)
            if (std::isfinite(costs[a][b])) include(a, b);
    };
    if (provided_graph) include_known();
    else add_nearest(nearest_count);
    if (preferred) for (int a = 0; a < n; ++a) {
        const int b = (*preferred)[a];
        if (b >= 0 && b < n && a != b && std::isfinite(costs[a][b])) include(a, b);
    }
    int rounds = 0, total_priced = 0, active_edges = 0;
    std::int64_t prepare_ns = 0, build_ns = 0, solve_ns = 0, pricing_ns = 0;
    for (; rounds < 5; ++rounds) {
        std::vector<std::vector<int>> neighbors(n);
        const int row_capacity = std::min(n, 2 * std::max(0, std::min(n, nearest_count)) + 4);
        for (auto& row : neighbors) row.reserve(row_capacity);
        active_edges = 0;
        for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b)
            if (included[static_cast<std::size_t>(a) * n + b]) {
                neighbors[a].push_back(b + 1);
                if (a < b) ++active_edges;
            }
        MatchingStats local;
        try {
            const auto build_started = detail::MatchingClock::now();
            prepare_ns += detail::matching_elapsed(prepare_started, build_started);
            detail::WeightedBlossom solver(costs, preferred, &local, &neighbors);
            const auto solve_started = detail::MatchingClock::now();
            build_ns += detail::matching_elapsed(build_started, solve_started);
            auto result = solver.solve();
            const auto solved = detail::MatchingClock::now();
            solve_ns += detail::matching_elapsed(solve_started, solved);
            bool geometric_certificate = false;
            if (ensure_edges && options && options->price_geometry) {
                std::vector<PricingPhase> phases;
                ++geometric_queries;
                const auto started = MatchingClock::now();
                geometric_certificate = solver.pricing_phases(phases) &&
                    options->price_geometry(phases);
                oracle_ns += matching_elapsed(started, MatchingClock::now());
                if (!geometric_certificate) ++geometric_fallbacks;
            }
            if (ensure_edges && !geometric_certificate) query(solver.vertex_radii());
            const auto pricing_started = detail::MatchingClock::now();
            std::vector<std::pair<int, int>> additions;
            int priced = 0;
            const bool reliable = solver.price_omitted(additions, &priced);
            pricing_ns += detail::matching_elapsed(pricing_started, detail::MatchingClock::now());
            total_priced += priced;
            if (!reliable) break;
            if (additions.empty()) {
                local.sparse_rounds = rounds + 1;
                local.priced_edges = total_priced;
                local.sparse_edges = active_edges;
                local.prepare_ns = prepare_ns;
                local.build_ns = build_ns;
                local.solve_ns = solve_ns;
                local.pricing_ns = pricing_ns;
                local.oracle_ns = oracle_ns;
                local.radius_queries = radius_queries;
                local.full_queries = full_queries;
                local.geometric_queries = geometric_queries;
                local.geometric_fallbacks = geometric_fallbacks;
                local.nearest_queries = nearest_queries;
                if (stats) *stats = local;
                return result;
            }
            for (auto [a, b] : additions) include(a, b);
        } catch (const std::runtime_error&) {
            nearest_count = std::min(n - 1, std::max(1, nearest_count) * 2);
            if (ensure_edges && options && options->grow_nearest) {
                const auto started = MatchingClock::now();
                options->grow_nearest(nearest_count);
                oracle_ns += matching_elapsed(started, MatchingClock::now());
                ++nearest_queries;
            } else if (ensure_edges && !full_queries) query({});
            if (provided_graph) include_known();
            else add_nearest(nearest_count);
        }
        prepare_started = detail::MatchingClock::now();
    }
    if (ensure_edges && !full_queries) query({});
    auto result = minimum_weight_perfect_matching(costs, preferred, stats);
    if (stats) {
        stats->sparse_rounds = rounds;
        stats->priced_edges = total_priced;
        stats->sparse_edges = active_edges;
        stats->dense_fallbacks = 1;
        stats->prepare_ns += prepare_ns;
        stats->build_ns += build_ns;
        stats->solve_ns += solve_ns;
        stats->pricing_ns += pricing_ns;
        stats->oracle_ns = oracle_ns;
        stats->radius_queries = radius_queries;
        stats->full_queries = full_queries;
        stats->geometric_queries = geometric_queries;
        stats->geometric_fallbacks = geometric_fallbacks;
        stats->nearest_queries = nearest_queries;
    }
    return result;
}

}  // namespace detail

inline std::vector<int> minimum_weight_perfect_matching_sparse(
    const std::vector<std::vector<double>>& costs,
    const std::vector<int>* preferred = nullptr,
    MatchingStats* stats = nullptr,
    int nearest_count = 12) {
    return detail::sparse_matching_impl(costs, preferred, stats, nearest_count, nullptr);
}

// The oracle must preserve every existing finite distance and resolve every
// unknown pair whose true distance is at most radii[i]+radii[j]. An empty radii
// vector requests all distances. Remaining unknown edges are then certified
// strictly feasible by the vertex-radius bound, not discarded heuristically.
inline std::vector<int> minimum_weight_perfect_matching_lazy(
    std::vector<std::vector<double>>& costs,
    const std::function<void(const std::vector<double>&)>& ensure_edges,
    const std::vector<int>* preferred = nullptr,
    MatchingStats* stats = nullptr,
    int nearest_count = 12,
    const LazyMatchingOptions* options = nullptr) {
    return detail::sparse_matching_impl(costs, preferred, stats, nearest_count, &ensure_edges, options);
}

}  // namespace owned_matching
