#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "policy.h"

namespace {
constexpr int kPlanes = 16, kBands = 14, kWidth = 8;
constexpr int kPlane = kBands * kWidth, kSites = kPlanes * kPlane, kBytes = kSites / 8;

struct Point { uint8_t t, u, v, band; };

Point position(int site) {
    const int band = (site % kPlane) / kWidth;
    const int x = site % kWidth;
    return {static_cast<uint8_t>(site / kPlane), static_cast<uint8_t>(x + (band + 2) / 2),
            static_cast<uint8_t>(x + 7 - (band + 1) / 2), static_cast<uint8_t>(band)};
}

int distance(const Point& a, const Point& b) {
    const int dt = int(b.t) - a.t;
    const int low = std::min(0, -dt), high = std::max(0, -dt);
    const auto excess = [&](int gap) { return std::max({0, low - gap, gap - high}); };
    return std::abs(dt) + excess(int(b.u) - a.u) + excess(int(b.v) - a.v);
}

const auto distances = [] {
    std::array<Point, kSites> points;
    for (int i = 0; i < kSites; ++i) points[i] = position(i);
    std::vector<uint8_t> out(kSites * kSites, 3);
    for (int i = 0; i < kSites; ++i) {
        const int end = std::min(kSites, (points[i].t + 3) * kPlane);
        for (int j = i + 1; j < end; ++j)
            out[i * kSites + j] = out[j * kSites + i] = std::min(3, distance(points[i], points[j]));
    }
    return out;
}();

struct Forest {
    std::array<uint8_t, QEC_EVENT_LIMIT> parent, size, low, high;
    uint8_t root(uint8_t i) {
        while (i != parent[i]) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    }
    void merge(uint8_t a, uint8_t b) {
        a = root(a);
        b = root(b);
        if (a == b) return;
        if (size[a] < size[b]) std::swap(a, b);
        parent[b] = a;
        size[a] += size[b];
        low[a] = std::min(low[a], low[b]);
        high[a] = std::max(high[a], high[b]);
    }
};

struct Link { uint8_t a, b; };

unsigned cuts_crossed(int low, int high) {
    if (high - low < QEC_EXTENDED_SPAN) return 0;
    return ((1u << high) - 1) ^ ((1u << low) - 1);
}

bool wide_even(const Forest& f, unsigned i) {
    return !(f.size[i] & 1) && f.high[i] - f.low[i] <= QEC_WIDE_SPAN &&
           f.low[i] >= QEC_WIDE_LOW && f.high[i] <= QEC_WIDE_HIGH;
}

template<bool extensions>
uint8_t decode(const uint8_t* plane_bytes, uint8_t* reason) {
    std::array<uint16_t, QEC_RAW_EVENT_LIMIT> events;
    unsigned n = 0;
    for (int i = 0; i < kBytes; ++i) {
        unsigned hits = plane_bytes[i];
        while (hits) {
            if (n == QEC_RAW_EVENT_LIMIT) {
                *reason = 1;
                return 0;
            }
            events[n++] = i * 8 + __builtin_ctz(hits);
            hits &= hits - 1;
        }
    }
    const unsigned raw_count = n;
    if (n > QEC_EVENT_LIMIT) {
        // The FPGA frees both slots of every isolated, neutral radius-one pair.
        std::array<uint16_t, QEC_RAW_EVENT_LIMIT> neighbor;
        std::array<uint8_t, QEC_RAW_EVENT_LIMIT> degree{};
        for (unsigned i = 0; i < n; ++i) {
            const int end = (events[i] / kPlane + 2) * kPlane;
            const auto row = distances.data() + events[i] * kSites;
            for (unsigned j = i + 1; j < n && events[j] < end; ++j) {
                if (row[events[j]] == 1) {
                    if (degree[i] < 2) ++degree[i];
                    if (degree[j] < 2) ++degree[j];
                    neighbor[i] = j;
                    neighbor[j] = i;
                }
            }
        }
        unsigned kept = 0;
        for (unsigned i = 0; i < n; ++i) {
            if (degree[i] != 1 || degree[neighbor[i]] != 1)
                events[kept++] = events[i];
        }
        n = kept;
        if (n > QEC_EVENT_LIMIT) {
            *reason = 1;
            return 0;
        }
    }
    const bool dense = raw_count > QEC_STANDARD_EVENT_LIMIT;
    const int span_one = dense ? QEC_DENSE_SPAN : QEC_FIRST_SPAN;
    const int span_two = dense ? QEC_DENSE_SPAN : QEC_SECOND_SPAN;
    const int depth_limit = dense ? QEC_DENSE_DEPTH : QEC_BOUNDARY_DEPTH;
    const unsigned cut_limit = dense ? QEC_DENSE_CROSSINGS : QEC_CROSSING_LIMIT;
    Forest initial{};
    std::vector<Link> two_hop;
    two_hop.reserve(n * 3);
    for (unsigned i = 0; i < n; ++i) {
        initial.parent[i] = i;
        initial.size[i] = 1;
        initial.low[i] = initial.high[i] = position(events[i]).band;
    }
    for (unsigned i = 0; i < n; ++i) {
        const int end = (events[i] / kPlane + 3) * kPlane;
        const auto row = distances.data() + events[i] * kSites;
        for (unsigned j = i + 1; j < n && events[j] < end; ++j) {
            if (row[events[j]] == 1) initial.merge(i, j);
            if (row[events[j]] == 2) two_hop.push_back({uint8_t(i), uint8_t(j)});
        }
    }
    unsigned extended = 0;
    bool used_wide = false;
    for (unsigned i = 0; i < n; ++i) {
        if (i != initial.parent[i]) continue;
        if (initial.high[i] - initial.low[i] > span_one) {
            if (!extensions || dense || !wide_even(initial, i)) {
                *reason = 2;
                return 0;
            }
            used_wide = true;
        }
        extended |= cuts_crossed(initial.low[i], initial.high[i]);
    }
    if (__builtin_popcount(extended) > cut_limit) {
        *reason = 7;
        return 0;
    }

    uint8_t predicted = 0;
    bool used_repair = false, used_near = false;
    int deepest = 0;
    for (int variant = 0; variant < QEC_VARIANTS; ++variant) {
        const unsigned freeze = (QEC_VARIANT_BITS >> (2 * variant)) & 3;
        Forest f = initial;
        std::array<bool, QEC_EVENT_LIMIT> remaining{};
        int top_depth = 0, bottom_depth = 0;
        unsigned flip = 0;
        for (unsigned i = 0; i < n; ++i) {
            if (i != f.parent[i] || !(f.size[i] & 1)) continue;
            const bool top = f.low[i] == 0 && (freeze & 1);
            const bool bottom = f.high[i] == kBands - 1 && (freeze & 2);
            if (top) {
                flip ^= 1;
                top_depth = std::max(top_depth, f.high[i] + 1);
            } else if (bottom) {
                bottom_depth = std::max(bottom_depth, kBands - f.low[i]);
            } else
                remaining[i] = true;
        }
        if (top_depth + bottom_depth > depth_limit) {
            *reason = 5;
            return 0;
        }
        for (const auto e : two_hop)
            if (remaining[f.root(e.a)] && remaining[f.root(e.b)]) f.merge(e.a, e.b);

        int orphans = 0, extended_groups = 0;
        Point first_orphan{};
        for (unsigned i = 0; i < n; ++i) {
            if (f.parent[i] != i || !remaining[i]) continue;
            if (f.high[i] - f.low[i] > span_two) {
                if (!extensions || dense || !wide_even(f, i)) {
                    *reason = 3;
                    return 0;
                }
                used_wide = true;
            }
            extended |= cuts_crossed(f.low[i], f.high[i]);
            if (!(f.size[i] & 1)) continue;
            bool top = f.low[i] == 0, bottom = f.high[i] == kBands - 1;
            if (extensions && !top && !bottom && raw_count <= QEC_NEAR_EVENT_LIMIT &&
                f.high[i] - f.low[i] <= QEC_NEAR_SPAN && (f.low[i] == 1 || f.high[i] == kBands - 2)) {
                top = f.low[i] == 1;
                bottom = f.high[i] == kBands - 2;
                used_near = true;
                if (++extended_groups > QEC_NEAR_GROUPS) {
                    *reason = 12;
                    return 0;
                }
            }
            if (top == bottom) {
                if (top || dense || f.size[i] != 1 || orphans == 2 ||
                    f.low[i] < QEC_REPAIR_LOW || f.high[i] > QEC_REPAIR_HIGH) {
                    *reason = 4;
                    return 0;
                }
                const Point p = position(events[i]);
                if (orphans == 0)
                    first_orphan = p;
                else if (distance(first_orphan, p) != 3 ||
                         std::abs(int(first_orphan.band) - p.band) > QEC_REPAIR_BANDS) {
                    *reason = 4;
                    return 0;
                }
                ++orphans;
            }
            if (top) {
                flip ^= 1;
                top_depth = std::max(top_depth, f.high[i] + 1);
            } else if (bottom)
                bottom_depth = std::max(bottom_depth, kBands - f.low[i]);
        }
        if (orphans == 1 || (orphans == 2 && top_depth + bottom_depth > QEC_REPAIR_DEPTH)) {
            *reason = 4;
            return 0;
        }
        used_repair |= orphans == 2;
        if (top_depth + bottom_depth > depth_limit) {
            *reason = 5;
            return 0;
        }
        deepest = std::max(deepest, top_depth + bottom_depth);
        if (__builtin_popcount(extended) > cut_limit) {
            *reason = 7;
            return 0;
        }
        if (variant == 0)
            predicted = flip;
        else if (predicted != flip) {
            *reason = 6;
            return 0;
        }
    }
    const unsigned crossings = __builtin_popcount(extended);
    if ((used_wide && (used_near || used_repair || deepest > QEC_WIDE_DEPTH ||
                       crossings > QEC_WIDE_CROSSINGS)) ||
        (used_near && (used_repair || crossings + deepest > QEC_NEAR_BUDGET))) {
        *reason = 12;
        return 0;
    }
    *reason = used_wide ? 11 : used_near ? 10 : dense ? 9 : used_repair ? 8 : 0;
    return 1 + predicted;
}
}

extern "C" void decode_batch(const uint8_t* packed, size_t n, uint8_t* codes, uint8_t* reasons) {
    for (size_t i = 0; i < n; ++i)
        codes[i] = decode<true>(packed + i * kBytes, reasons + i);
}

extern "C" void prior_batch(const uint8_t* packed, size_t n, uint8_t* codes, uint8_t* reasons) {
    for (size_t i = 0; i < n; ++i)
        codes[i] = decode<false>(packed + i * kBytes, reasons + i);
}
