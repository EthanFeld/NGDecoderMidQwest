#include "owned_mwpm.h"
#include <chrono>
#include <functional>
#include <iostream>
#include <random>

static double exact(const std::vector<std::vector<double>>& c) {
    const int n = static_cast<int>(c.size());
    std::vector<double> memo(1u << n, -1);
    std::function<double(unsigned)> go = [&](unsigned mask) -> double {
        if (!mask) return 0;
        double& best = memo[mask];
        if (best >= 0) return best;
        best = std::numeric_limits<double>::infinity();
        int a = 0;
        while (!(mask & (1u << a))) ++a;
        const unsigned rest = mask ^ (1u << a);
        for (int b = a + 1; b < n; ++b)
            if (rest & (1u << b)) best = std::min(best, c[a][b] + go(rest ^ (1u << b)));
        return best;
    };
    return go((1u << n) - 1);
}

int main(int argc, char**) {
    std::mt19937_64 random(27451);
    int checked = 0;
    int sparse_checked = 0, sparse_certified = 0, sparse_fallbacks = 0, certified_blossoms = 0;
    for (int n = 0; n <= 16; n += 2) {
        const int trials = n <= 12 ? 3000 : 100;
        for (int trial = 0; trial < trials; ++trial) {
            std::vector<std::vector<double>> c(n, std::vector<double>(n));
            for (int a = 0; a < n; ++a)
                for (int b = a + 1; b < n; ++b) {
                    double value = double(random() % 1000) / (trial % 3 == 0 ? 7 : 1);
                    if (trial % 4 == 0) value = double(random() % 3);
                    if (trial % 5 == 0 && random() % 3 == 0)
                        value = std::numeric_limits<double>::infinity();
                    c[a][b] = c[b][a] = value;
                }
            double want = exact(c), got = std::numeric_limits<double>::infinity();
            try {
                auto mate = owned_matching::minimum_weight_perfect_matching(c);
                got = 0;
                for (int a = 0; a < n; ++a) {
                    if (mate[a] == a || mate[a] < 0 || mate[a] >= n || mate[mate[a]] != a)
                        throw std::logic_error("bad mate array");
                    got += c[a][mate[a]] / 2;
                }
                if (trial % 11 == 0) {
                    std::vector<int> order(n), preferred(n, -1);
                    for (int a = 0; a < n; ++a) order[a] = a;
                    std::shuffle(order.begin(), order.end(), random);
                    for (int a = 0; a < n; a += 2) {
                        preferred[order[a]] = order[a + 1];
                        preferred[order[a + 1]] = order[a];
                    }
                    owned_matching::MatchingStats stats;
                    auto warm = owned_matching::minimum_weight_perfect_matching(c, &preferred, &stats);
                    double warm_cost = 0;
                    for (int a = 0; a < n; ++a) warm_cost += c[a][warm[a]] / 2;
                    if (std::abs(warm_cost - want) > 1e-8 * std::max(1.0, want))
                        throw std::logic_error("preferred pairs changed optimum");
                    if (stats.initial_pairs + stats.augmentations != n / 2)
                        throw std::logic_error("matching counters disagree");
                }
                if (n >= 8) {
                    owned_matching::MatchingStats sparse_stats;
                    std::vector<int> sparse_preferred(n);
                    for (int a = 0; a < n; ++a) sparse_preferred[a] = a ^ 1;
                    auto sparse = owned_matching::minimum_weight_perfect_matching_sparse(
                        c, trial % 7 == 0 ? &sparse_preferred : nullptr, &sparse_stats, 1 + trial % 3);
                    ++sparse_checked;
                    if (!sparse_stats.dense_fallbacks) ++sparse_certified;
                    else ++sparse_fallbacks;
                    if (!sparse_stats.dense_fallbacks && sparse_stats.contractions) ++certified_blossoms;
                    double sparse_cost = 0;
                    for (int a = 0; a < n; ++a) sparse_cost += c[a][sparse[a]] / 2;
                    if (std::abs(sparse_cost - want) > 1e-8 * std::max(1.0, want)) {
                        std::cerr << "sparse mismatch n=" << n << " trial=" << trial << '\n';
                        throw std::logic_error("sparse pricing changed optimum");
                    }
                }
            } catch (const std::runtime_error&) {
                if (std::isfinite(want)) {
                    std::cerr << "unexpected failure n=" << n << " trial=" << trial << "\n";
                    throw;
                }
            }
            if (std::isfinite(got) != std::isfinite(want) ||
                (std::isfinite(want) && std::abs(want - got) > 1e-8 * std::max(1.0, want))) {
                std::cerr << "mismatch n=" << n << " trial=" << trial
                          << " got=" << got << " want=" << want << "\n";
                for (auto& row : c) { for (auto x : row) std::cerr << x << ' '; std::cerr << '\n'; }
                return 1;
            }
            ++checked;
        }
    }
    std::cout << "exact comparisons passed: " << checked << "\n";
    std::cout << "sparse comparisons=" << sparse_checked << " certified=" << sparse_certified
              << " dense_fallbacks=" << sparse_fallbacks << " with_blossoms=" << certified_blossoms << '\n';
    if (sparse_certified < 1000 || certified_blossoms < 100)
        throw std::logic_error("sparse certificate rarely succeeds");
    {
        // Each cheap triangle has odd cardinality. At least one expensive
        // cross-cluster edge must be chosen, requiring an odd-cycle blossom.
        const int n = 6;
        std::vector<std::vector<double>> c(n, std::vector<double>(n, 100));
        for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b)
            if (a / 3 == b / 3) c[a][b] = a == b ? 0 : 1;
        owned_matching::MatchingStats stats;
        auto mate = owned_matching::minimum_weight_perfect_matching(c, nullptr, &stats);
        double got = 0;
        for (int a = 0; a < n; ++a) got += c[a][mate[a]] / 2;
        if (got != 102 || stats.contractions == 0)
            throw std::logic_error("odd-cycle test failed");
    }
    {
        // The input difference is representable in double, despite a large
        // common offset. A loose global double epsilon would miss the optimum.
        const double large = 1e12;
        std::vector<std::vector<double>> c{
            {0, large + .015625, large, 2 * large},
            {large + .015625, 0, 2 * large, large},
            {large, 2 * large, 0, large + .015625},
            {2 * large, large, large + .015625, 0},
        };
        auto mate = owned_matching::minimum_weight_perfect_matching(c);
        double got = 0;
        for (int a = 0; a < 4; ++a) got += c[a][mate[a]] / 2;
        if (got != 2 * large) throw std::logic_error("close-cost precision test failed");
        std::vector<std::vector<double>> doubled(8, std::vector<double>(8, 3 * large));
        for (int offset : {0, 4}) for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b)
            doubled[a + offset][b + offset] = c[a][b];
        auto sparse = owned_matching::minimum_weight_perfect_matching_sparse(doubled, nullptr, nullptr, 1);
        got = 0;
        for (int a = 0; a < 8; ++a) got += doubled[a][sparse[a]] / 2;
        if (got != 4 * large) throw std::logic_error("sparse close-cost precision test failed");
    }
    {
        std::vector<std::vector<double>> invalid{{0, 1}, {2, 0}};
        bool rejected = false;
        try { (void)owned_matching::minimum_weight_perfect_matching(invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::logic_error("asymmetric cost validation failed");
    }
    std::cout << "warm-start, odd-cycle, close-cost and input-validation checks passed\n";
    int large_checked = 0;
    for (int n : {32, 64, 128}) {
        for (int trial = 0; trial < (n == 128 ? 5 : 25); ++trial) {
            std::vector<std::vector<double>> c(n, std::vector<double>(n));
            std::vector<std::pair<int, int>> points(n);
            for (auto& p : points) p = {int(random() % 15), int(random() % 15)};
            for (int a = 0; a < n; ++a) for (int b = a + 1; b < n; ++b) {
                const double distance = std::abs(points[a].first - points[b].first) +
                                        std::abs(points[a].second - points[b].second);
                c[a][b] = c[b][a] = trial % 2 ? distance : double(random() % 10000) / 7;
            }
            auto dense = owned_matching::minimum_weight_perfect_matching(c);
            auto sparse = owned_matching::minimum_weight_perfect_matching_sparse(c);
            long double dense_cost = 0, sparse_cost = 0;
            for (int a = 0; a < n; ++a) {
                dense_cost += static_cast<long double>(c[a][dense[a]]) / 2;
                sparse_cost += static_cast<long double>(c[a][sparse[a]]) / 2;
            }
            if (std::abs(dense_cost - sparse_cost) > 1e-10L * std::max(1.0L, dense_cost))
                throw std::logic_error("large sparse/dense cost mismatch");
            ++large_checked;
        }
    }
    std::cout << "large metric/nonmetric sparse/dense comparisons passed: " << large_checked << '\n';
    if (argc > 1) return 0;
    for (int n : {50, 100, 200, 300}) {
        std::vector<std::vector<double>> c(n, std::vector<double>(n));
        for (int a = 0; a < n; ++a)
            for (int b = a + 1; b < n; ++b) c[a][b] = c[b][a] = random() % 100000;
        auto start = std::chrono::steady_clock::now();
        auto mate = owned_matching::minimum_weight_perfect_matching(c);
        auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::cout << "n=" << n << " elapsed_ms=" << elapsed << "\n";
    }
    for (int n : {50, 100, 200}) {
        std::vector<std::pair<int, int>> points(n);
        for (auto& p : points) p = {int(random() % 15), int(random() % 15)};
        std::vector<std::vector<double>> c(n, std::vector<double>(n));
        for (int a = 0; a < n; ++a) for (int b = a + 1; b < n; ++b)
            c[a][b] = c[b][a] = std::abs(points[a].first - points[b].first) +
                                std::abs(points[a].second - points[b].second);
        owned_matching::MatchingStats stats;
        auto start = std::chrono::steady_clock::now();
        auto mate = owned_matching::minimum_weight_perfect_matching(c, nullptr, &stats);
        auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::cout << "grid n=" << n << " elapsed_ms=" << elapsed
                  << " initial_pairs=" << stats.initial_pairs
                  << " augmentations=" << stats.augmentations << '\n';
    }
}
