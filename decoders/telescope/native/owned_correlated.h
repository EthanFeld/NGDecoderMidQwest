#pragma once

// Original native two-pass correlation pipeline. Graph objects remain owned by
// the caller and must outlive this object. Destroy the pipeline before closing
// those graphs. Scratch masks are overwritten each call, never answer caches.

#include "owned_workers.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace owned_matching {

template <class Graph>
class CorrelatedPipeline {
    std::array<Graph*, 2> prior_, updated_;
    int joint_n_, total_edges_ = 0;
    std::array<int, 2> offsets_{0, 0};
    std::array<std::vector<std::int32_t>, 2> indices_;
    std::array<std::vector<std::uint8_t>, 2> syndrome_;
    std::vector<double> base_, weights_, rule_weights_;
    std::vector<std::int32_t> causes_, targets_;
    std::vector<std::uint8_t> first_, second_, hint_;
    std::array<bool, 2> changed_{false, false};
    const std::uint8_t* removed_ = nullptr;
    int hint_used_ = 0;
    std::atomic<bool> active_{false};
    // Declared last so worker threads join before any referenced buffers die.
    std::unique_ptr<PathWorkers> workers_;

    struct ActiveGuard {
        std::atomic<bool>& flag;
        explicit ActiveGuard(std::atomic<bool>& value) : flag(value) {
            bool idle = false;
            if (!flag.compare_exchange_strong(idle, true, std::memory_order_acquire))
                throw std::logic_error("correlation pipeline is already active");
        }
        ~ActiveGuard() { flag.store(false, std::memory_order_release); }
    };

    template <class T>
    static T* slice(std::vector<T>& values, int offset) {
        return values.empty() ? nullptr : values.data() + offset;
    }
    template <class T>
    static const T* slice(const std::vector<T>& values, int offset) {
        return values.empty() ? nullptr : values.data() + offset;
    }

    void prior_pass(int basis) {
        const std::uint8_t* seed = nullptr;
        if (basis == 0 && removed_) {
            try {
                for (int i = 0; i < prior_[0]->n; ++i)
                    if (removed_[i] > 1 || (removed_[i] && !syndrome_[0][i]))
                        throw std::invalid_argument("invalid FPGA removed-detector hint");
                prior_[0]->hint(syndrome_[0].data(), removed_, base_.data(), hint_.data());
                seed = hint_.data();
                hint_used_ = 1;
            } catch (const std::exception&) {
                // Match the host's hint-only fallback: still solve the full
                // original syndrome using the unmodified prior probabilities.
                hint_used_ = 0;
            }
        }
        prior_[basis]->solve(syndrome_[basis].data(), slice(base_, offsets_[basis]),
                             seed, slice(first_, offsets_[basis]));
    }

    void updated_pass(int basis) {
        const int offset = offsets_[basis], length = prior_[basis]->m;
        if (changed_[basis]) {
            updated_[basis]->solve(syndrome_[basis].data(), slice(weights_, offset),
                                  slice(first_, offset), slice(second_, offset));
        } else {
            std::copy(first_.begin() + offset, first_.begin() + offset + length,
                      second_.begin() + offset);
        }
    }

public:
    CorrelatedPipeline(
        Graph* prior0, Graph* prior1, Graph* updated0, Graph* updated1,
        int joint_n, const std::int32_t* indices0, const std::int32_t* indices1,
        const double* base, int rules, const std::int32_t* causes,
        const std::int32_t* targets, const double* rule_weights, int basis_workers = 1)
        : prior_{prior0, prior1}, updated_{updated0, updated1}, joint_n_(joint_n) {
        if (!prior0 || !prior1 || !updated0 || !updated1)
            throw std::invalid_argument("missing correlation graph");
        if (prior0 == prior1 || updated0 == updated1)
            throw std::invalid_argument("check bases require independent graph workspaces");
        if (joint_n < 0 || rules < 0)
            throw std::invalid_argument("negative correlation model size");
        for (int b = 0; b < 2; ++b)
            if (prior_[b]->n < 0 || prior_[b]->m < 0 ||
                prior_[b]->n != updated_[b]->n || prior_[b]->m != updated_[b]->m)
                throw std::invalid_argument("incompatible correlation graph dimensions");
        if (prior0->m > std::numeric_limits<int>::max() - prior1->m)
            throw std::length_error("correlation edge count exceeds supported index range");
        total_edges_ = prior0->m + prior1->m;
        offsets_[1] = prior0->m;
        if ((total_edges_ && !base) || (rules && (!causes || !targets || !rule_weights)))
            throw std::invalid_argument("missing correlation model arrays");
        if (prior0->n > joint_n || prior1->n != joint_n - prior0->n)
            throw std::invalid_argument("check bases do not partition the joint syndrome");
        std::vector<std::uint8_t> seen(joint_n, 0);
        const std::int32_t* supplied_indices[2]{indices0, indices1};
        for (int b = 0; b < 2; ++b) {
            const int count = prior_[b]->n;
            if (count && !supplied_indices[b]) throw std::invalid_argument("missing detector indices");
            if (count) indices_[b].assign(supplied_indices[b], supplied_indices[b] + count);
            syndrome_[b].resize(count);
            for (std::int32_t index : indices_[b]) {
                if (index < 0 || index >= joint_n || seen[index])
                    throw std::invalid_argument("invalid or repeated detector index");
                seen[index] = 1;
            }
        }
        if (total_edges_) base_.assign(base, base + total_edges_);
        for (double value : base_)
            if (!std::isfinite(value)) throw std::invalid_argument("nonfinite prior matching weight");
        weights_.resize(total_edges_);
        first_.resize(total_edges_);
        second_.resize(total_edges_);
        hint_.resize(prior0->m);
        if (rules) {
            causes_.assign(causes, causes + rules);
            targets_.assign(targets, targets + rules);
            rule_weights_.assign(rule_weights, rule_weights + rules);
        }
        for (int i = 0; i < rules; ++i)
            if (causes_[i] < 0 || causes_[i] >= total_edges_ || targets_[i] < 0 ||
                targets_[i] >= total_edges_ || !std::isfinite(rule_weights_[i]))
                throw std::invalid_argument("invalid correlation rule");
        set_basis_workers(basis_workers);
    }

    CorrelatedPipeline(const CorrelatedPipeline&) = delete;
    CorrelatedPipeline& operator=(const CorrelatedPipeline&) = delete;

    void set_basis_workers(int count) {
        ActiveGuard idle(active_);
        if (count != 1 && count != 2)
            throw std::invalid_argument("basis worker count must be one or two");
        if (workers_ && workers_->workers() == count) return;
        auto replacement = std::make_unique<PathWorkers>(count);
        workers_.swap(replacement);
    }

    int basis_workers() const noexcept { return workers_->workers(); }
    int total_edges() const noexcept { return total_edges_; }
    int joint_detectors() const noexcept { return joint_n_; }

    void solve(const std::uint8_t* full_row, const std::uint8_t* removed_memory,
               bool correlations, std::uint8_t* output, int& hint_used) {
        ActiveGuard idle(active_);
        hint_used = hint_used_ = 0;
        if ((joint_n_ && !full_row) || (total_edges_ && !output))
            throw std::invalid_argument("missing correlation input/output");
        for (int b = 0; b < 2; ++b) for (int i = 0; i < prior_[b]->n; ++i) {
            const std::uint8_t value = full_row[indices_[b][i]];
            if (value > 1) throw std::invalid_argument("nonbinary joint syndrome");
            syndrome_[b][i] = value;
        }
        removed_ = removed_memory;
        std::copy(base_.begin(), base_.end(), weights_.begin());
        workers_->run(2, [this](int basis) { prior_pass(basis); });
        hint_used = hint_used_;
        if (!correlations) {
            std::copy(first_.begin(), first_.end(), output);
            return;
        }
        changed_ = {false, false};
        // Rules retain their original order, including duplicate targets.
        // With finite values this is the same minimum-at operation as the
        // host reduction, including a later equal-valued rule operand.
        for (std::size_t i = 0; i < causes_.size(); ++i) if (first_[causes_[i]]) {
            const int target = targets_[i];
            const double value = rule_weights_[i];
            if (value <= weights_[target]) {
                weights_[target] = value;
                // Minima only decrease: once a basis differs from its prior
                // it cannot revert. Value equality matches array_equal.
                if (value != base_[target]) changed_[target >= offsets_[1]] = true;
            }
        }
        workers_->run(2, [this](int basis) { updated_pass(basis); });
        std::copy(second_.begin(), second_.end(), output);
    }

    void run(const std::uint8_t* full_row, const std::uint8_t* removed_memory,
             bool correlations, std::uint8_t* output, int& hint_used) {
        solve(full_row, removed_memory, correlations, output, hint_used);
    }

    // Views are diagnostic scratch state, valid until the next solve call.
    const std::vector<std::uint8_t>& last_first() const noexcept { return first_; }
    const std::vector<double>& last_weights() const noexcept { return weights_; }
};

}  // namespace owned_matching
