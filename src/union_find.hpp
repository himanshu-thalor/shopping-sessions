#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "csv_parse.hpp"
#include "hash_util.hpp"
#include "io_util.hpp"
#include "thread_pool.hpp"

class LinkTable {
   public:
    static constexpr size_t kShards = 32;
    static constexpr unsigned kShardShift = 64u - 5u;

    struct LinkSlot {
        CompactStr key;
        uint32_t val;
    };
    static_assert(sizeof(LinkSlot) == 16);

    struct LookupShard {
        const LinkSlot* ptr = nullptr;
        size_t mask = 0;
    };

    void build(const std::string& links_dir, ThreadPool& pool,
               std::vector<ThreadBlockCache>& caches) {
        const auto lt0 = std::chrono::steady_clock::now();
        size_t total_bytes = 0;
        auto files = open_all_pread_files(links_dir, &total_bytes);
        auto chunks = build_pread_chunks(files, size_t(512) << 10);
        if (chunks.empty()) return;

        const size_t ncpu = pool.size();

        struct StrItem {
            uint64_t h;
            CompactStr s;
            uint32_t endpoint;
        };
        static_assert(sizeof(StrItem) == 24);
        static constexpr size_t kStrBlockCap = 2048;

        // Adjacent entries are one input edge. Each shard writes its own
        // interned node at the saved endpoint position, preserving the pairing.
        struct Endpoint {
            uint32_t local;
            uint32_t shard;
        };
        std::vector<OverflowArena> tmp_arenas(ncpu);
        std::vector<std::vector<Endpoint>> endpoints(ncpu);
        std::vector<std::vector<ChunkedVec<StrItem, kStrBlockCap>>> shard_items(ncpu);
        for (size_t t = 0; t < ncpu; ++t) {
            shard_items[t].resize(kShards);
            for (size_t s = 0; s < kShards; ++s) {
                shard_items[t][s].set_cache(&caches[t]);
            }
        }

        // Sub-phase 1: Parallel 1 MB pread + SIMD CSV scan
        {
            std::atomic<size_t> next_chunk{0};
            pool.run([&](size_t tid) {
                ThreadBlockCache& my_cache = caches[tid];
                OverflowArena& ta = tmp_arenas[tid];
                size_t endpoint_count = 0;
                auto& out_shards = shard_items[tid];
                const uint16_t aid = static_cast<uint16_t>(tid);

                std::vector<char> buf;
                buf.reserve((size_t(1) << 20) + 65536 + 64);

                for (;;) {
                    const size_t ci = next_chunk.fetch_add(1, std::memory_order_relaxed);
                    if (ci >= chunks.size()) break;
                    size_t data_start = 0, limit = 0, blen = 0;
                    read_chunk_into(chunks[ci], buf, data_start, limit, blen);
                    if (data_start >= limit) continue;

                    for_each_csv_line<2>(
                        buf.data(), data_start, limit, blen, [&](const std::string_view (&f)[2]) {
                            if (f[0].empty() || f[1].empty()) return;
                            const uint64_t ha = fast_str_hash(f[0]);
                            const uint64_t hb = fast_str_hash(f[1]);
                            const CompactStr ca = CompactStr::make_str(f[0], aid, ta);
                            const CompactStr cb = CompactStr::make_str(f[1], aid, ta);
                            if (endpoint_count > UINT32_MAX - 1u) {
                                throw std::length_error("too many link endpoints per worker");
                            }
                            const auto first = static_cast<uint32_t>(endpoint_count);
                            endpoint_count += 2;
                            out_shards[static_cast<size_t>(ha >> kShardShift)].push_back(
                                StrItem{ha, ca, first});
                            out_shards[static_cast<size_t>(hb >> kShardShift)].push_back(
                                StrItem{hb, cb, first + 1});
                        });
                }
                endpoints[tid].resize(endpoint_count);
                my_cache.flush_all();
            });
        }
        const auto lt1 = std::chrono::steady_clock::now();

        // Close file descriptors promptly now that reading is done
        std::vector<PreadChunk>().swap(chunks);
        std::vector<PreadFile>().swap(files);

        // Sub-phase 2: Parallel 32-shard LinkSlot deduplication & interning
        std::vector<CompactStr> shard_nodes[kShards];
        {
            std::atomic<size_t> next_shard{0};
            pool.run([&](size_t tid) {
                ThreadBlockCache& my_cache = caches[tid];
                const OverflowArena* ta_ptr = tmp_arenas.data();

                for (;;) {
                    const size_t s = next_shard.fetch_add(1, std::memory_order_relaxed);
                    if (s >= kShards) break;

                    size_t tot = 0;
                    for (size_t t = 0; t < ncpu; ++t) tot += shard_items[t][s].size();
                    if (tot == 0) continue;

                    size_t cap = 64;
                    while (cap * 6 < tot * 10 && cap < (size_t(1) << 18)) cap <<= 1;

                    HugeBuffer<LinkSlot> slots;
                    slots.assign_zero(cap);
                    size_t mask = cap - 1;
                    size_t count = 0;
                    auto& arena = node_arenas_[s];
                    auto& local_nodes = shard_nodes[s];
                    local_nodes.reserve(std::min(tot, cap));

                    auto rehash_shard = [&]() {
                        const size_t ncap = slots.size() * 2;
                        HugeBuffer<LinkSlot> nxt;
                        nxt.assign_zero(ncap);
                        const size_t nmask = ncap - 1;
                        for (size_t i = 0; i < slots.size(); ++i) {
                            const LinkSlot& sl = slots[i];
                            if (sl.key.empty_sentinel()) continue;
                            const uint64_t h = fast_compact_hash(sl.key, node_arenas_);
                            size_t idx = static_cast<size_t>(h) & nmask;
                            while (!nxt[idx].key.empty_sentinel()) idx = (idx + 1) & nmask;
                            nxt[idx] = sl;
                        }
                        slots = std::move(nxt);
                        mask = nmask;
                    };

                    for (size_t t = 0; t < ncpu; ++t) {
                        shard_items[t][s].for_each_block([&](const StrItem* blk, size_t len) {
                            for (size_t i = 0; i < len; ++i) {
                                if (i + 4 < len) {
                                    __builtin_prefetch(&slots[static_cast<size_t>(blk[i + 4].h) & mask],
                                                       0, 1);
                                }
                                if ((count + 1) * 10 >= slots.size() * 6) {
                                    rehash_shard();
                                }
                                const StrItem& it = blk[i];
                                size_t idx = static_cast<size_t>(it.h) & mask;
                                for (;;) {
                                    LinkSlot& sl = slots[idx];
                                    if (sl.key.empty_sentinel()) {
                                        const uint32_t local_id =
                                            static_cast<uint32_t>(local_nodes.size());
                                        const std::string_view sv = it.s.raw_view(ta_ptr);
                                        const CompactStr interned =
                                            CompactStr::make_str(sv, static_cast<uint16_t>(s), arena);
                                        sl.key = interned;
                                        sl.val = local_id;
                                        local_nodes.push_back(interned);
                                        ++count;
                                        break;
                                    }
                                    if (CompactStr::equals_cross(sl.key, node_arenas_, it.s,
                                                                 ta_ptr)) {
                                        break;
                                    }
                                    idx = (idx + 1) & mask;
                                }
                                endpoints[t][it.endpoint] = {slots[idx].val, static_cast<uint32_t>(s)};
                            }
                        });
                        shard_items[t][s].free_to(&my_cache);
                    }

                    shard_slots_[s] = std::move(slots);
                    lookup_[s] = LookupShard{shard_slots_[s].data(), mask};
                }
                my_cache.flush_all();
            });
        }
        const auto lt2 = std::chrono::steady_clock::now();

        uint32_t shard_base[kShards]{};
        size_t num_nodes_sz = 0;
        for (size_t s = 0; s < kShards; ++s) {
            shard_base[s] = static_cast<uint32_t>(num_nodes_sz);
            num_nodes_sz += shard_nodes[s].size();
        }
        if (num_nodes_sz == 0) return;
        has_links_ = true;
        const uint32_t num_nodes = static_cast<uint32_t>(num_nodes_sz);
        nodes_.resize(num_nodes);
        for (size_t s = 0; s < kShards; ++s) {
            std::copy(shard_nodes[s].begin(), shard_nodes[s].end(),
                      nodes_.begin() + shard_base[s]);
            std::vector<CompactStr>().swap(shard_nodes[s]);
        }
        tmp_arenas.clear();
        const auto lt3 = std::chrono::steady_clock::now();

        // Join the interned endpoints by rank, then name each component once.
        std::vector<uint32_t> parent(num_nodes);
        std::vector<uint8_t> rank(num_nodes, 0);
        for (uint32_t i = 0; i < num_nodes; ++i) parent[i] = i;

        auto find = [&](uint32_t x) -> uint32_t {
            uint32_t r = x;
            while (parent[r] != r) r = parent[r];
            while (parent[x] != r) {
                const uint32_t nxt = parent[x];
                parent[x] = r;
                x = nxt;
            }
            return r;
        };

        for (const auto& worker : endpoints) {
            for (size_t i = 0; i < worker.size(); i += 2) {
                const Endpoint a = worker[i], b = worker[i + 1];
                uint32_t ra = find(shard_base[a.shard] + a.local);
                uint32_t rb = find(shard_base[b.shard] + b.local);
                if (ra == rb) continue;
                if (rank[ra] < rank[rb]) std::swap(ra, rb);
                parent[rb] = ra;
                if (rank[ra] == rank[rb]) ++rank[ra];
            }
        }
        std::vector<uint8_t>().swap(rank);

        // Compute byte-wise minimum node per component once per node
        std::vector<uint32_t> min_node(num_nodes);
        for (uint32_t i = 0; i < num_nodes; ++i) min_node[i] = i;
        for (uint32_t i = 0; i < num_nodes; ++i) {
            const uint32_t r = find(i);
            parent[i] = r;
            if (i != min_node[r] && byte_less(node_sv(i), node_sv(min_node[r]))) {
                min_node[r] = i;
            }
        }

        // Publish canonical representatives to the lookup shards.
        {
            std::atomic<size_t> next_shard{0};
            pool.run([&](size_t) {
                for (;;) {
                    const size_t s = next_shard.fetch_add(1, std::memory_order_relaxed);
                    if (s >= kShards) break;
                    LinkSlot* sl = shard_slots_[s].data();
                    const size_t cap = shard_slots_[s].size();
                    for (size_t i = 0; i < cap; ++i) {
                        if (!sl[i].key.empty_sentinel()) {
                            sl[i].val = min_node[parent[shard_base[s] + sl[i].val]];
                        }
                    }
                }
            });
        }
        const auto lt4 = std::chrono::steady_clock::now();
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        std::fprintf(stderr, "links breakdown: read=%.1f shard=%.1f flatten=%.1f uf+remap=%.1f\n",
                     ms(lt0, lt1), ms(lt1, lt2), ms(lt2, lt3), ms(lt3, lt4));
    }

    bool empty() const noexcept { return !has_links_; }

    void prefetch(uint64_t h) const noexcept {
        if (!has_links_) return;
        const LookupShard& sh = lookup_[static_cast<size_t>(h >> kShardShift)];
        if (sh.ptr) {
            __builtin_prefetch(&sh.ptr[static_cast<size_t>(h) & sh.mask], 0, 1);
        }
    }

    bool resolve_with_hash(const CompactStr& key, uint64_t h, uint32_t& out_canon_id,
                           const OverflowArena* key_arenas) const noexcept {
        if (!has_links_) return false;
        const LookupShard& sh = lookup_[static_cast<size_t>(h >> kShardShift)];
        if (!sh.ptr) return false;
        size_t idx = static_cast<size_t>(h) & sh.mask;
        for (;;) {
            const LinkSlot& sl = sh.ptr[idx];
            if (sl.key.empty_sentinel()) return false;
            if (CompactStr::equals_cross(sl.key, node_arenas_, key, key_arenas)) {
                out_canon_id = sl.val;
                return true;
            }
            idx = (idx + 1) & sh.mask;
        }
    }

    bool resolve_sv_with_hash(std::string_view sv, uint64_t h,
                              uint32_t& out_canon_id) const noexcept {
        if (!has_links_) return false;
        const LookupShard& sh = lookup_[static_cast<size_t>(h >> kShardShift)];
        if (!sh.ptr) return false;
        size_t idx = static_cast<size_t>(h) & sh.mask;
        for (;;) {
            const LinkSlot& sl = sh.ptr[idx];
            if (sl.key.empty_sentinel()) return false;
            if (CompactStr::equals_sv(sl.key, node_arenas_, sv)) {
                out_canon_id = sl.val;
                return true;
            }
            idx = (idx + 1) & sh.mask;
        }
    }

    std::string_view node_sv(uint32_t node_id) const noexcept {
        return nodes_[node_id].raw_view(node_arenas_);
    }

    static uint64_t canonical_hash(uint32_t canon_node_id) noexcept {
        return wyhash_mix(canon_node_id + uint64_t{1}, 0x9e3779b97f4a7c15ULL);
    }

   private:
    OverflowArena node_arenas_[kShards];
    HugeBuffer<LinkSlot> shard_slots_[kShards];
    LookupShard lookup_[kShards]{};
    std::vector<CompactStr> nodes_;
    bool has_links_ = false;
};
