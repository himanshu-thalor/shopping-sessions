#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "csv_parse.hpp"
#include "hash_util.hpp"
#include "io_util.hpp"

class LinkTable {
   public:
    using UserId = FastId64<'u'>;
    static constexpr uint32_t kNone = UINT32_MAX;

    struct LinkedRoot {
        UserId name;
        uint32_t root_lid;
        uint32_t pad;
    };
    static_assert(sizeof(LinkedRoot) == 16);

    struct Partition {
        struct Entry {
            UserId key;
            uint32_t local;
            uint32_t count;
        };
        static_assert(sizeof(Entry) == 16);

        std::vector<Entry> table;
        std::vector<UserId> names;
        uint32_t num_linked = 0;
        uint32_t linked_base = 0;

        [[gnu::noinline]] void reserve(size_t expected, const OverflowArena* const* arenas) {
            const size_t need = std::max<size_t>(64, (expected * 10 + 6) / 7);
            const size_t cap = std::bit_ceil(need);
            if (cap <= table.size()) return;
            std::vector<Entry> bigger(cap);
            const size_t mask = cap - 1;
            for (const auto& e : table) {
                if (e.key.empty()) continue;
                size_t i = static_cast<size_t>(e.key.hash(arenas)) & mask;
                while (!bigger[i].key.empty()) i = (i + 1) & mask;
                bigger[i] = e;
            }
            table.swap(bigger);
        }

        [[gnu::noinline]] Entry& intern_grow(UserId key, uint64_t hash,
                                             const OverflowArena* const* arenas) {
            reserve(std::max<size_t>(64, table.size() * 2), arenas);
            const size_t mask = table.size() - 1;
            size_t i = static_cast<size_t>(hash) & mask;
            for (;;) {
                Entry& e = table[i];
                if (UserId::equals(e.key, key, arenas)) return e;
                if (e.key.empty()) {
                    const uint32_t n = static_cast<uint32_t>(names.size());
                    e = Entry{key, n, 0};
                    names.push_back(key);
                    return e;
                }
                i = (i + 1) & mask;
            }
        }

        [[gnu::always_inline]] inline Entry& intern(UserId key, uint64_t hash,
                                                    const OverflowArena* const* arenas) {
            if (__builtin_expect(table.empty(), 0)) return intern_grow(key, hash, arenas);
            const size_t mask = table.size() - 1;
            size_t i = static_cast<size_t>(hash) & mask;
            for (;;) {
                Entry& e = table[i];
                if (__builtin_expect(UserId::equals(e.key, key, arenas), 1)) return e;
                if (e.key.empty()) {
                    if (__builtin_expect((names.size() + 1) * 10 >= table.size() * 7, 0)) {
                        return intern_grow(key, hash, arenas);
                    }
                    const uint32_t n = static_cast<uint32_t>(names.size());
                    e = Entry{key, n, 0};
                    names.push_back(key);
                    return e;
                }
                i = (i + 1) & mask;
            }
        }

        [[gnu::always_inline]] inline Entry* find_existing(
            UserId key, uint64_t hash, const OverflowArena* const* arenas) noexcept {
            if (__builtin_expect(table.empty(), 0)) return nullptr;
            const size_t mask = table.size() - 1;
            size_t i = static_cast<size_t>(hash) & mask;
            for (;;) {
                Entry& e = table[i];
                if (__builtin_expect(UserId::equals(e.key, key, arenas), 1)) return &e;
                if (e.key.empty()) return nullptr;
                i = (i + 1) & mask;
            }
        }

        void release_table() noexcept {
            std::vector<Entry>().swap(table);
            std::vector<UserId>().swap(names);
        }
    };

    struct Endpoint {
        UserId key;
        uint32_t hash;
        uint32_t local_ep;
    };
    static_assert(sizeof(Endpoint) == 16);
    static constexpr size_t kEpCap = ThreadBlockArena::kBlockBytes / sizeof(Endpoint);

    LinkTable(size_t num_shards, size_t max_link_threads)
        : shard_shift_(64u - static_cast<unsigned>(std::countr_zero(num_shards))),
          parts_(num_shards),
          arenas_(std::max<size_t>(1, max_link_threads)) {}

    size_t num_arenas() const noexcept { return arenas_.size(); }
    OverflowArena& arena(size_t i) noexcept { return arenas_[i]; }
    const OverflowArena& arena(size_t i) const noexcept { return arenas_[i]; }
    Partition& part(size_t s) noexcept { return parts_[s]; }
    const Partition& part(size_t s) const noexcept { return parts_[s]; }
    uint32_t total_linked() const noexcept { return total_linked_; }
    const std::vector<LinkedRoot>& linked_roots() const noexcept { return linked_roots_; }
    uint32_t next_alias(uint32_t lid) const noexcept { return next_alias_[lid]; }

    // Prepare overlapped multi-thread link build state BEFORE Phase A's pool.run().
    // Returns the number of Phase A workers (tid 0..n_link-1) that will build links
    // before immediately joining events scanning!
    size_t prepare_overlapped(std::vector<PreadFile>& files, size_t ncpu) {
        files_ = &files;
        chunks_ = build_pread_chunks(files, size_t(64) << 10);
        if (chunks_.empty()) {
            n_workers_ = 0;
            return 0;
        }
        size_t nw = 1;
        if (ncpu >= 6 && chunks_.size() >= 8) nw = 4;
        else if (ncpu >= 3 && chunks_.size() >= 2) nw = 2;
        nw = std::min(nw, arenas_.size());
        n_workers_ = nw;

        shard_eps_.assign(nw, std::vector<std::vector<BlockSlice<Endpoint>>>(parts_.size()));
        thread_ep_cnt_.assign(nw, 0);
        thread_ep_base_.assign(nw, 0);
        next_chunk_.store(0, std::memory_order_relaxed);
        read_done_.store(0, std::memory_order_relaxed);
        stage1_ready_.store(0, std::memory_order_relaxed);
        next_shard_.store(0, std::memory_order_relaxed);
        global_id_.store(0, std::memory_order_relaxed);
        intern_done_.store(0, std::memory_order_relaxed);
        return nw;
    }

    // Called inside Phase A's pool.run() ONLY by tid < n_workers_.
    // Stage 1 (all n_workers): parallel read + shard links CSVs.
    // Stage 2 (all n_workers): parallel per-shard interning + edge assembly.
    //   -> tid 1..n_workers-1 return IMMEDIATELY after Stage 2 to scan events!
    // Stage 3 (tid 0 only): zero-atomic flat Union-Find + canonical min_node + next_alias_.
    //   -> tid 0 returns immediately after Stage 3 to join events!
    void run_overlapped_worker(size_t tid, ThreadBlockArena& my_arena,
                               const OverflowArena* const* arenas, uint16_t link_aid_base,
                               bool use_avx512) {
        const size_t nw = n_workers_;
        const size_t nshards = parts_.size();
        const unsigned shift = shard_shift_;
        const uint16_t my_aid = static_cast<uint16_t>(link_aid_base + tid);
        OverflowArena& my_str = arenas_[tid];

        // Stage 1: Read & shard links chunks.
        {
            struct EpOut {
                Endpoint* cur = nullptr;
                Endpoint* end = nullptr;
            };
            std::vector<EpOut> eout(nshards);
            auto& my_eps = shard_eps_[tid];
            uint32_t local_ep = 0;

            auto push_ep = [&](size_t s, const Endpoint& ep) __attribute__((always_inline)) {
                EpOut& o = eout[s];
                if (__builtin_expect(o.cur == o.end, 0)) {
                    if (o.cur != nullptr) {
                        my_eps[s].push_back(BlockSlice<Endpoint>{o.cur - kEpCap,
                                                                 static_cast<uint32_t>(kEpCap)});
                    }
                    Endpoint* blk = static_cast<Endpoint*>(my_arena.alloc());
                    o.cur = blk;
                    o.end = blk + kEpCap;
                }
                *o.cur++ = ep;
            };

            auto on_row_impl = [&](const std::string_view (&f)[2],
                                   auto padded_tag) __attribute__((always_inline)) {
                constexpr bool pad = decltype(padded_tag)::value;
                if (__builtin_expect(f[0].empty() || f[1].empty(), 0)) return;
                const UserId a = pad ? UserId::make_str_padded(f[0], my_aid, my_str)
                                     : UserId::make_str_safe(f[0], my_aid, my_str);
                const UserId b = pad ? UserId::make_str_padded(f[1], my_aid, my_str)
                                     : UserId::make_str_safe(f[1], my_aid, my_str);
                if (__builtin_expect(a.raw == b.raw && a.is_inline(), 0)) return;
                const uint64_t ha = a.hash(arenas);
                const uint64_t hb = b.hash(arenas);
                const uint32_t ep0 = local_ep;
                local_ep += 2;
                push_ep(static_cast<size_t>(ha >> shift),
                        Endpoint{a, static_cast<uint32_t>(ha), ep0});
                push_ep(static_cast<size_t>(hb >> shift),
                        Endpoint{b, static_cast<uint32_t>(hb), ep0 + 1});
            };

            auto on_fast = [&](const std::string_view (&f)[2]) __attribute__((always_inline)) {
                on_row_impl(f, std::true_type{});
            };
            auto on_slow = [&](const std::string_view (&f)[2]) {
                on_row_impl(f, std::false_type{});
            };

            RawBuf buf;
            PreadView view;
            const auto& files = *files_;
            for (;;) {
                const size_t ci = next_chunk_.fetch_add(1, std::memory_order_relaxed);
                if (ci >= chunks_.size()) break;
                const PreadChunk& pc = chunks_[ci];
                if (!read_chunk_into(files[pc.file_idx], pc, buf, view)) continue;
                if (use_avx512) {
                    csv_simd::scan_pread_buf_avx512<2>(view, on_fast, on_slow);
                } else {
                    csv_simd::scan_pread_buf_avx2<2>(view, on_fast, on_slow);
                }
            }

            for (size_t s = 0; s < nshards; ++s) {
                if (eout[s].cur != nullptr) {
                    Endpoint* base = eout[s].end - kEpCap;
                    const uint32_t len = static_cast<uint32_t>(eout[s].cur - base);
                    if (len > 0) my_eps[s].push_back(BlockSlice<Endpoint>{base, len});
                }
            }
            thread_ep_cnt_[tid] = local_ep;
        }

        // Mini-barrier among the n_workers link threads; last arriver allocates edges_.
        if (read_done_.fetch_add(1, std::memory_order_acq_rel) + 1 == nw) {
            size_t tot_eps = 0;
            for (size_t t = 0; t < nw; ++t) {
                thread_ep_base_[t] = tot_eps;
                tot_eps += thread_ep_cnt_[t];
            }
            total_eps_ = tot_eps;
            if (tot_eps > 0) {
                edges_ = std::make_unique_for_overwrite<uint32_t[]>(tot_eps);
            }
            chunks_.clear();
            stage1_ready_.store(1, std::memory_order_release);
            stage1_ready_.notify_all();
        } else {
            while (stage1_ready_.load(std::memory_order_acquire) == 0) {
                stage1_ready_.wait(0, std::memory_order_relaxed);
            }
        }

        if (total_eps_ == 0) return;

        // Stage 2: Parallel per-shard L2-hot interning with 8-ahead prefetch!
        {
            struct EpLoc {
                uint32_t global_ep;
                uint32_t loc;
            };
            std::vector<EpLoc> tmp_ep_loc;
            tmp_ep_loc.reserve(8192);
            uint32_t* const ep_dst = edges_.get();

            for (;;) {
                const size_t s = next_shard_.fetch_add(1, std::memory_order_relaxed);
                if (s >= nshards) break;

                size_t tot = 0;
                for (size_t t = 0; t < nw; ++t) {
                    for (const auto& sl : shard_eps_[t][s]) tot += sl.len;
                }
                Partition& part = parts_[s];
                if (tot == 0) {
                    part.num_linked = 0;
                    part.linked_base = 0;
                    continue;
                }
                // Reserve ~tot/2 + 64 (leaves headroom for C1 unlinked users without rehashing).
                part.reserve(tot / 2 + 64, arenas);
                part.names.reserve(tot / 2 + 64);
                size_t mask = part.table.size() - 1;
                tmp_ep_loc.clear();

                for (size_t t = 0; t < nw; ++t) {
                    const uint32_t ep_base = static_cast<uint32_t>(thread_ep_base_[t]);
                    for (auto sl : shard_eps_[t][s]) {
                        const Endpoint* const ptr = sl.ptr;
                        const uint32_t len = sl.len;
                        for (uint32_t i = 0; i < len; ++i) {
                            if (i + 8 < len) {
                                __builtin_prefetch(&part.table[ptr[i + 8].hash & mask], 1, 1);
                            }
                            const Endpoint& ep = ptr[i];
                            const uint32_t loc = part.intern(ep.key, ep.hash, arenas).local;
                            mask = part.table.size() - 1;
                            tmp_ep_loc.push_back(EpLoc{ep_base + ep.local_ep, loc});
                        }
                        my_arena.recycle(sl.ptr);
                    }
                    std::vector<BlockSlice<Endpoint>>().swap(shard_eps_[t][s]);
                }

                const uint32_t nloc = static_cast<uint32_t>(part.names.size());
                const uint32_t gbase = global_id_.fetch_add(nloc, std::memory_order_relaxed);
                part.num_linked = nloc;
                part.linked_base = gbase;

                for (const EpLoc& el : tmp_ep_loc) {
                    ep_dst[el.global_ep] = gbase + el.loc;
                }
            }
        }

        // Workers tid 1..nw-1 signal completion and IMMEDIATELY return to scan events!
        if (tid != 0) {
            if (intern_done_.fetch_add(1, std::memory_order_acq_rel) + 1 == nw) {
                intern_done_.notify_one();
            }
            return;
        }

        // tid 0 waits for remaining Stage 2 workers, then runs flat zero-atomic Union-Find.
        intern_done_.fetch_add(1, std::memory_order_acq_rel);
        for (;;) {
            const size_t done = intern_done_.load(std::memory_order_acquire);
            if (done >= nw) break;
            intern_done_.wait(done, std::memory_order_relaxed);
        }

        total_linked_ = global_id_.load(std::memory_order_relaxed);
        if (total_linked_ == 0) {
            edges_.reset();
            return;
        }

        // Stage 3: Flat zero-atomic Union-Find (16-ahead software prefetched!).
        const uint32_t N = total_linked_;
        auto parent = std::make_unique_for_overwrite<uint32_t[]>(N);
        auto rank = std::make_unique<uint8_t[]>(N);
        for (uint32_t i = 0; i < N; ++i) parent[i] = i;

        auto find = [&](uint32_t x) noexcept -> uint32_t {
            while (parent[x] != x) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        };

        const uint32_t* ep_ptr = edges_.get();
        const size_t nedges = total_eps_;
        for (size_t i = 0; i < nedges; i += 2) {
            if (i + 32 < nedges) {
                __builtin_prefetch(&parent[ep_ptr[i + 32]], 1, 1);
                __builtin_prefetch(&parent[ep_ptr[i + 33]], 1, 1);
            }
            uint32_t ra = find(ep_ptr[i]);
            uint32_t rb = find(ep_ptr[i + 1]);
            if (ra == rb) continue;
            if (rank[ra] < rank[rb]) std::swap(ra, rb);
            parent[rb] = ra;
            if (rank[ra] == rank[rb]) ++rank[ra];
        }
        edges_.reset();
        rank.reset();

        // Flatten names for cache-friendly canonical min_node selection.
        auto all_names = std::make_unique_for_overwrite<UserId[]>(N);
        for (size_t s = 0; s < nshards; ++s) {
            const Partition& part = parts_[s];
            if (part.num_linked > 0) {
                std::memcpy(all_names.get() + part.linked_base, part.names.data(),
                            part.num_linked * sizeof(UserId));
            }
        }

        auto min_node = std::make_unique_for_overwrite<uint32_t[]>(N);
        for (uint32_t i = 0; i < N; ++i) min_node[i] = i;
        for (uint32_t i = 0; i < N; ++i) {
            if (i + 16 < N) {
                const uint32_t pr = parent[i + 16];
                __builtin_prefetch(&parent[pr], 0, 1);
                __builtin_prefetch(&min_node[pr], 1, 1);
            }
            uint32_t r = i;
            while (parent[r] != r) r = parent[r];
            parent[i] = r;
            const uint32_t cur = min_node[r];
            if (i != cur && UserId::less(all_names[i], all_names[cur], arenas)) {
                min_node[r] = i;
            }
        }

        // Build compact flat linked_roots_ and next_alias_ list.
        next_alias_ = std::make_unique_for_overwrite<uint32_t[]>(N);
        std::memset(next_alias_.get(), 0xFF, N * sizeof(uint32_t));
        linked_roots_.reserve(N / 2);
        for (uint32_t i = 0; i < N; ++i) {
            if (i + 16 < N) {
                __builtin_prefetch(&min_node[parent[i + 16]], 0, 1);
            }
            const uint32_t c = min_node[parent[i]];
            if (c == i) {
                linked_roots_.push_back(LinkedRoot{all_names[i], i, 0});
            } else {
                next_alias_[i] = next_alias_[c];
                next_alias_[c] = i;
            }
        }
        std::vector<std::vector<std::vector<BlockSlice<Endpoint>>>>().swap(shard_eps_);
    }

   private:
    unsigned shard_shift_;
    std::vector<Partition> parts_;
    std::vector<OverflowArena> arenas_;
    uint32_t total_linked_ = 0;
    std::vector<LinkedRoot> linked_roots_;
    std::unique_ptr<uint32_t[]> next_alias_;

    // Overlapped build state
    std::vector<PreadFile>* files_ = nullptr;
    std::vector<PreadChunk> chunks_;
    size_t n_workers_ = 0;
    std::vector<std::vector<std::vector<BlockSlice<Endpoint>>>> shard_eps_;
    std::vector<size_t> thread_ep_cnt_;
    std::vector<size_t> thread_ep_base_;
    size_t total_eps_ = 0;
    std::unique_ptr<uint32_t[]> edges_;
    std::atomic<size_t> next_chunk_{0};
    std::atomic<size_t> read_done_{0};
    std::atomic<uint32_t> stage1_ready_{0};
    std::atomic<size_t> next_shard_{0};
    std::atomic<uint32_t> global_id_{0};
    std::atomic<size_t> intern_done_{0};
};
