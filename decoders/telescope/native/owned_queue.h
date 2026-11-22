#pragma once

// Original monotone queue for nonnegative IEEE-754 double distances.
// The integer key is the double's exact bit representation: no rounding or
// weight quantization is performed. Equal-distance entries retain the same
// smallest-node-first ordering as a priority_queue<pair<double,int>,greater>.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace owned_matching {

class MonotoneQueue {
    struct Entry {
        std::uint64_t key;
        int node;
    };
    struct NodeAfter {
        bool operator()(const Entry& a, const Entry& b) const noexcept {
            return a.node > b.node;
        }
    };

    std::array<std::vector<Entry>, 65> buckets_;
    std::uint64_t last_ = 0;
    std::size_t count_ = 0;

    static unsigned bucket(std::uint64_t key, std::uint64_t previous) noexcept {
        return 64U - static_cast<unsigned>(std::countl_zero(key ^ previous));
    }

    void refill() {
        if (!buckets_[0].empty()) return;
        if (!count_) throw std::out_of_range("top/pop on an empty monotone queue");
        unsigned source_index = 1;
        while (buckets_[source_index].empty()) ++source_index;
        auto& source = buckets_[source_index];
        auto& ready = buckets_[0];

        // Continuous-valued distances often expose one event at a time. This
        // avoids counting destinations and constructing a heap for that case.
        if (source.size() == 1) {
            const Entry event = source.front();
            ready.push_back(event);
            last_ = event.key;
            source.clear();
            return;
        }

        std::uint64_t next = source.front().key;
        for (const Entry& event : source) next = std::min(next, event.key);

        // Every redistributed destination is below source_index. Reserve all
        // needed space before modifying any queue contents, so allocation
        // failure leaves a usable queue with its original keys and clock.
        std::array<std::size_t, 65> incoming{};
        std::uint64_t destinations = 0;
        for (const Entry& event : source) {
            const unsigned target = bucket(event.key, next);
            ++incoming[target];
            destinations |= std::uint64_t{1} << target;
        }
        for (std::uint64_t mask = destinations; mask; mask &= mask - 1) {
            const unsigned target = static_cast<unsigned>(std::countr_zero(mask));
            auto& destination = buckets_[target];
            const std::size_t required = destination.size() + incoming[target];
            if (required > destination.capacity()) {
                const std::size_t grown = destination.capacity() > destination.max_size() / 2
                    ? destination.max_size() : destination.capacity() * 2;
                destination.reserve(std::max(required, grown));
            }
        }
        last_ = next;
        for (const Entry& event : source)
            buckets_[bucket(event.key, next)].push_back(event);
        source.clear();
        std::make_heap(ready.begin(), ready.end(), NodeAfter{});
    }

public:
    bool empty() const noexcept { return count_ == 0; }
    std::size_t size() const noexcept { return count_; }

    void emplace(double distance, int node) {
        if (!std::isfinite(distance) || distance < 0)
            throw std::invalid_argument("monotone queue distance must be finite and nonnegative");
        if (distance == 0) distance = 0;  // Canonicalize negative zero.
        const std::uint64_t key = std::bit_cast<std::uint64_t>(distance);
        if (key < last_)
            throw std::invalid_argument("monotone queue key precedes its current distance");
        const unsigned index = bucket(key, last_);
        auto& destination = buckets_[index];
        destination.push_back(Entry{key, node});
        if (index == 0)
            std::push_heap(destination.begin(), destination.end(), NodeAfter{});
        ++count_;
    }

    std::pair<double, int> top() {
        refill();
        const Entry& event = buckets_[0].front();
        return {std::bit_cast<double>(event.key), event.node};
    }

    void pop() {
        refill();
        auto& ready = buckets_[0];
        std::pop_heap(ready.begin(), ready.end(), NodeAfter{});
        ready.pop_back();
        --count_;
    }

    // Dynamic allocation capacity for the caller's route-cache accounting.
    std::size_t bytes() const noexcept {
        std::size_t result = 0;
        for (const auto& values : buckets_) result += values.capacity() * sizeof(Entry);
        return result;
    }

    // A cached frontier needs only its live buckets. Release buffers belonging
    // to past distances before evicting useful model-derived search state.
    void release_unused_capacity() noexcept {
        for (auto& values : buckets_)
            if (values.empty() && values.capacity()) std::vector<Entry>().swap(values);
    }
};

}  // namespace owned_matching
