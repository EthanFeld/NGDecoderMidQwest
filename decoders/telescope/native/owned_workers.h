#pragma once

// In-repository persistent workers for independent shortest-path sources.
// A batch is synchronous; the caller participates and tasks have no prescribed
// execution order. Jobs must keep their mutable source workspaces disjoint.

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace owned_matching {

struct FloatingEnvironment {
    std::fenv_t standard{};
    bool valid=false;
    std::uint16_t x87_control=0;
    std::uint32_t sse_control=0;
    static FloatingEnvironment capture() noexcept {
        FloatingEnvironment state;state.valid=std::fegetenv(&state.standard)==0;
#if defined(__i386__) || defined(__x86_64__)
        __asm__ volatile("fnstcw %0" : "=m"(state.x87_control));
        __asm__ volatile("stmxcsr %0" : "=m"(state.sse_control));
#endif
        return state;
    }
    void apply() const noexcept {
        if(valid) std::fesetenv(&standard);
#if defined(__i386__) || defined(__x86_64__)
        // CRT worker startup can select different x87 precision on Windows.
        // Standard rounding-mode propagation alone does not cover that field.
        __asm__ volatile("fldcw %0" : : "m"(x87_control));
        __asm__ volatile("ldmxcsr %0" : : "m"(sse_control));
#endif
    }
};

struct FloatingEnvironmentGuard {
    FloatingEnvironment previous=FloatingEnvironment::capture();
    explicit FloatingEnvironmentGuard(const FloatingEnvironment& state) noexcept {state.apply();}
    ~FloatingEnvironmentGuard() {previous.apply();}
};

class PathWorkers {
    int concurrency_ = 1;
    std::mutex mutex_;
    std::condition_variable wake_, finished_;
    std::vector<std::thread> threads_;
    std::function<void(int)> job_;
    std::atomic<int> next_{0};
    std::atomic<bool> cancelled_{false}, running_{false};
    int count_ = 0, pending_ = 0;
    std::size_t generation_ = 0;
    bool stopping_ = false;
    std::exception_ptr failure_;
    FloatingEnvironment environment_;

    struct RunGuard {
        std::atomic<bool>& running;
        explicit RunGuard(std::atomic<bool>& flag) : running(flag) {
            bool idle = false;
            if (!running.compare_exchange_strong(idle, true, std::memory_order_acquire))
                throw std::logic_error("nested or concurrent run on one path worker pool");
        }
        ~RunGuard() { running.store(false, std::memory_order_release); }
    };

    void consume() noexcept {
        while (!cancelled_.load(std::memory_order_relaxed)) {
            const int index = next_.fetch_add(1, std::memory_order_relaxed);
            if (index >= count_) break;
            try {
                job_(index);
            } catch (...) {
                const auto error = std::current_exception();
                if (!cancelled_.exchange(true, std::memory_order_acq_rel)) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    failure_ = error;
                }
                break;
            }
        }
    }

    void worker_loop() noexcept {
        std::size_t observed = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            wake_.wait(lock, [&] { return stopping_ || generation_ != observed; });
            if (stopping_) return;
            observed = generation_;
            const auto environment=environment_;
            lock.unlock();
            {FloatingEnvironmentGuard guard(environment);consume();}
            lock.lock();
            if (--pending_ == 0) finished_.notify_one();
        }
    }

    void stop_and_join() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_all();
        for (auto& thread : threads_) if (thread.joinable()) thread.join();
    }

public:
    // Count includes the caller. In particular, workers=1 creates no threads.
    explicit PathWorkers(int workers) {
        if (workers < 1) throw std::invalid_argument("path worker count must be positive");
        const unsigned hardware = std::thread::hardware_concurrency();
        concurrency_ = hardware ? std::min<unsigned>(workers, hardware) : workers;
        threads_.reserve(static_cast<std::size_t>(concurrency_ - 1));
        try {
            for (int i = 1; i < concurrency_; ++i)
                threads_.emplace_back([this] { worker_loop(); });
        } catch (...) {
            stop_and_join();
            throw;
        }
    }

    PathWorkers(const PathWorkers&) = delete;
    PathWorkers& operator=(const PathWorkers&) = delete;
    PathWorkers(PathWorkers&&) = delete;
    PathWorkers& operator=(PathWorkers&&) = delete;

    ~PathWorkers() { stop_and_join(); }

    int workers() const noexcept { return concurrency_; }

    void run(int count, std::function<void(int)> job) {
        RunGuard active(running_);
        if (count < 0) throw std::invalid_argument("negative path job count");
        if (!count) return;
        if (!job) throw std::invalid_argument("missing path worker job");
        if (concurrency_ == 1 || count == 1) {
            for (int index = 0; index < count; ++index) job(index);
            return;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        job_ = std::move(job);
        environment_=FloatingEnvironment::capture();
        count_ = count;
        pending_ = static_cast<int>(threads_.size());
        next_.store(0, std::memory_order_relaxed);
        cancelled_.store(false, std::memory_order_relaxed);
        failure_ = nullptr;
        ++generation_;
        lock.unlock();
        wake_.notify_all();
        consume();
        lock.lock();
        finished_.wait(lock, [&] { return pending_ == 0; });
        const auto error = failure_;
        job_ = {};
        lock.unlock();
        if (error) std::rethrow_exception(error);
    }
};

}  // namespace owned_matching
