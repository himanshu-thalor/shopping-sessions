#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>

class ThreadPool {
   public:
    explicit ThreadPool(size_t ncpu) : ncpu_(ncpu >= 1 ? ncpu : 1) {
        if (ncpu_ > 1) {
            workers_.reserve(ncpu_ - 1);
            for (size_t t = 1; t < ncpu_; ++t) {
                workers_.emplace_back([this, t]() { loop(t); });
            }
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool() {
        if (ncpu_ > 1) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                stop_ = true;
                ++gen_;
            }
            cv_start_.notify_all();
            for (auto& th : workers_) th.join();
        }
    }

    size_t size() const noexcept { return ncpu_; }

    template <typename Fn>
    void run(Fn&& fn) {
        if (ncpu_ == 1) {
            fn(size_t(0));
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            ctx_ = &fn;
            thunk_ = [](void* p, size_t tid) { (*static_cast<Fn*>(p))(tid); };
            remaining_ = ncpu_ - 1;
            ++gen_;
        }
        cv_start_.notify_all();
        fn(size_t(0));
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_done_.wait(lk, [&]() { return remaining_ == 0; });
            ctx_ = nullptr;
            thunk_ = nullptr;
        }
    }

   private:
    void loop(size_t tid) {
        uint64_t seen_gen = 0;
        for (;;) {
            void* ctx = nullptr;
            void (*thunk)(void*, size_t) = nullptr;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_start_.wait(lk, [&]() { return gen_ != seen_gen; });
                if (stop_) return;
                seen_gen = gen_;
                ctx = ctx_;
                thunk = thunk_;
            }
            thunk(ctx, tid);
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (--remaining_ == 0) {
                    cv_done_.notify_one();
                }
            }
        }
    }

    size_t ncpu_;
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    uint64_t gen_ = 0;
    size_t remaining_ = 0;
    bool stop_ = false;
    void* ctx_ = nullptr;
    void (*thunk_)(void*, size_t) = nullptr;
};
