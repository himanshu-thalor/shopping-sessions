#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "csv_parse.hpp"
#include "hash_util.hpp"
#include "io_util.hpp"
#include "thread_pool.hpp"

class LinkTable {
   public:
    using UserId64 = FastId64<'u'>;
    static constexpr size_t kShards = 64;
    static constexpr unsigned kShardShift = 64u - 6u;
    static constexpr size_t kStrBlockCap = 4096;  // 4096 * 16B = 64 KB

    struct LinkSlot {
        UserId64 key;
        UserId64 canon;
    };
    static_assert(sizeof(LinkSlot) == 16);

    struct StrItem {
        UserId64 s;
        uint32_t h32;
        uint32_t ep;
    };
    static_assert(sizeof(StrItem) == 16);

    OverflowArena& node_arena(size_t s) noexcept { return node_arenas_[s]; }
    const OverflowArena& node_arena(size_t s) const noexcept { return node_arenas_[s]; }

    void build(const std::string& links_dir, ThreadPool& pool,
               std::vector<ThreadBlockArena>& arenas,
               const OverflowArena* const* all_user_arenas, uint16_t link_arena_base) {
        const auto lt0 = std::chrono::steady_clock::now();
        size_t total_bytes = 0;
        auto files = open_all_pread_files(links_dir, pool, &total_bytes);
        auto chunks = build_pread_chunks(files, size_t(256) << 10);
        if (chunks.empty()) return;

        const size_t ncpu = pool.size();
        std::vector<OverflowArena> tmp_arenas(ncpu);
        std::vector<const OverflowArena*> tmp_arena_ptrs(ncpu);
        for (size_t t = 0; t < ncpu; ++t) tmp_arena_ptrs[t] = &tmp_arenas[t];

        std::vector<std::vector<uint32_t>> endpoints(ncpu);
        std::vector<std::vector<std::vector<BlockSlice<StrItem>>>> shard_items(
            ncpu, std::vector<std::vector<BlockSlice<StrItem>>>(kShards));

        // Sub-phase 1: Parallel 256 KB pread + SIMD CSV scan
        {
            std::atomic<size_t> next_chunk{0};
            const bool use_avx512 = cpu_has_avx512bw();
            pool.run([&](size_t tid) {
                ThreadBlockArena& my_arena = arenas[tid];
                OverflowArena& ta = tmp_arenas[tid];
                const OverflowArena* const* ta_ptrs = tmp_arena_ptrs.data();
                const uint16_t aid = static_cast<uint16_t>(tid);
                auto& out_shards = shard_items[tid];
                std::vector<uint32_t>& my_eps = endpoints[tid];
                my_eps.reserve(16384);

                std::vector<char> chunk_buf;
                PreadView view;

                StrItem* cur[kShards]{};
                StrItem* endp[kShards]{};
                auto push_item = [&](size_t s, const StrItem& it) {
                    if (__builtin_expect(cur[s] == endp[s], 0)) {
                        if (cur[s] != nullptr) {
                            out_shards[s].push_back(
                                BlockSlice<StrItem>{cur[s] - kStrBlockCap, kStrBlockCap});
                        }
                        StrItem* blk = static_cast<StrItem*>(my_arena.alloc());
                        cur[s] = blk;
                        endp[s] = blk + kStrBlockCap;
                    }
                    *cur[s]++ = it;
                };

                auto handle_fast = [&](const std::string_view (&f)[2]) {
                    if (__builtin_expect(f[0].empty() || f[1].empty(), 0)) return;
                    const UserId64 ca = UserId64::make_str_padded(f[0], aid, ta);
                    const UserId64 cb = UserId64::make_str_padded(f[1], aid, ta);
                    if (__builtin_expect(ca.raw == cb.raw && ca.is_inline(), 0)) return;
                    const uint64_t ha = ca.hash(ta_ptrs);
                    const uint64_t hb = cb.hash(ta_ptrs);
                    const uint32_t ep0 = static_cast<uint32_t>(my_eps.size());
                    my_eps.push_back(0);
                    my_eps.push_back(0);
                    push_item(static_cast<size_t>(ha >> kShardShift),
                              StrItem{ca, static_cast<uint32_t>(ha), ep0});
                    push_item(static_cast<size_t>(hb >> kShardShift),
                              StrItem{cb, static_cast<uint32_t>(hb), ep0 + 1});
                };

                auto handle_slow = [&](const std::string_view (&f)[2]) {
                    if (__builtin_expect(f[0].empty() || f[1].empty(), 0)) return;
                    const UserId64 ca = UserId64::make_str_safe(f[0], aid, ta);
                    const UserId64 cb = UserId64::make_str_safe(f[1], aid, ta);
                    if (__builtin_expect(ca.raw == cb.raw && ca.is_inline(), 0)) return;
                    const uint64_t ha = ca.hash(ta_ptrs);
                    const uint64_t hb = cb.hash(ta_ptrs);
                    const uint32_t ep0 = static_cast<uint32_t>(my_eps.size());
                    my_eps.push_back(0);
                    my_eps.push_back(0);
                    push_item(static_cast<size_t>(ha >> kShardShift),
                              StrItem{ca, static_cast<uint32_t>(ha), ep0});
                    push_item(static_cast<size_t>(hb >> kShardShift),
                              StrItem{cb, static_cast<uint32_t>(hb), ep0 + 1});
                };

                for (;;) {
                    const size_t ci = next_chunk.fetch_add(1, std::memory_order_relaxed);
                    if (ci >= chunks.size()) break;
                    const PreadChunk& pc = chunks[ci];
                    if (!read_chunk_into(files[pc.file_idx], pc, chunk_buf, view)) continue;
                    if (use_avx512) {
                        csv_simd::scan_pread_buf_avx512<2>(view, handle_fast, handle_slow);
                    } else {
                        csv_simd::scan_pread_buf_avx2<2>(view, handle_fast, handle_slow);
                    }
                }

                for (size_t s = 0; s < kShards; ++s) {
                    if (cur[s] != nullptr) {
                        StrItem* base = endp[s] - kStrBlockCap;
                        const uint32_t len = static_cast<uint32_t>(cur[s] - base);
                        if (len > 0) out_shards[s].push_back(BlockSlice<StrItem>{base, len});
                    }
                }
            });
        }
        const auto lt1 = std::chrono::steady_clock::now();
        std::vector<PreadChunk>().swap(chunks);
        std::vector<PreadFile>().swap(files);

        // Sub-phase 2: Parallel 64-shard L2-resident interning
        struct InternSlot {
            UserId64 key;
            uint32_t local_id;
        };
        std::vector<UserId64> shard_nodes[kShards];
        {
            std::atomic<size_t> next_shard{0};
            pool.run([&](size_t tid) {
                ThreadBlockArena& my_arena = arenas[tid];
                std::vector<InternSlot> table;

                for (;;) {
                    const size_t s = next_shard.fetch_add(1, std::memory_order_relaxed);
                    if (s >= kShards) break;

                    size_t tot = 0;
                    for (size_t t = 0; t < ncpu; ++t) {
                        for (const auto& sl : shard_items[t][s]) tot += sl.len;
                    }
                    if (tot == 0) continue;

                    size_t cap = 64;
                    while (cap * 6 < tot * 10 && cap < (size_t(1) << 18)) cap <<= 1;
                    table.assign(cap, InternSlot{});
                    size_t mask = cap - 1;
                    size_t count = 0;
                    OverflowArena& arena = node_arenas_[s];
                    const uint16_t perm_aid = static_cast<uint16_t>(link_arena_base + s);
                    auto& local_nodes = shard_nodes[s];
                    local_nodes.reserve(std::min(tot, cap));

                    auto rehash_shard = [&]() {
                        const size_t ncap = table.size() * 2;
                        std::vector<InternSlot> nxt(ncap, InternSlot{});
                        const size_t nmask = ncap - 1;
                        for (const InternSlot& sl : table) {
                            if (sl.key.empty()) continue;
                            const uint64_t h = sl.key.hash(all_user_arenas);
                            size_t idx = static_cast<size_t>(h) & nmask;
                            while (!nxt[idx].key.empty()) idx = (idx + 1) & nmask;
                            nxt[idx] = sl;
                        }
                        table.swap(nxt);
                        mask = nmask;
                    };

                    for (size_t t = 0; t < ncpu; ++t) {
                        const OverflowArena& ta = tmp_arenas[t];
                        for (const BlockSlice<StrItem>& blk : shard_items[t][s]) {
                            const StrItem* items = blk.ptr;
                            const size_t len = blk.len;
                            for (size_t i = 0; i < len; ++i) {
                                if (i + 4 < len) {
                                    __builtin_prefetch(&table[items[i + 4].h32 & mask], 0, 1);
                                }
                                if ((count + 1) * 10 >= table.size() * 6) {
                                    rehash_shard();
                                }
                                const StrItem& it = items[i];
                                size_t idx = static_cast<size_t>(it.h32) & mask;
                                for (;;) {
                                    InternSlot& sl = table[idx];
                                    if (sl.key.empty()) {
                                        const uint32_t local_id =
                                            static_cast<uint32_t>(local_nodes.size());
                                        UserId64 interned = it.s;
                                        if (it.s.is_overflow()) {
                                            interned = UserId64::make_overflow(
                                                ta.get_len_str(it.s.overflow_off()), perm_aid,
                                                arena);
                                        }
                                        sl.key = interned;
                                        sl.local_id = local_id;
                                        local_nodes.push_back(interned);
                                        ++count;
                                        break;
                                    }
                                    if (sl.key.raw == it.s.raw) break;
                                    if (((sl.key.raw & it.s.raw) >> 63) != 0) {
                                        const std::string_view sa =
                                            arena.get_len_str(sl.key.overflow_off());
                                        const std::string_view sb =
                                            ta.get_len_str(it.s.overflow_off());
                                        if (sa.size() == sb.size() &&
                                            (sa.empty() ||
                                             std::memcmp(sa.data(), sb.data(), sa.size()) == 0)) {
                                            break;
                                        }
                                    }
                                    idx = (idx + 1) & mask;
                                }
                                endpoints[t][it.ep] =
                                    (static_cast<uint32_t>(s) << 26) | table[idx].local_id;
                            }
                            my_arena.recycle(blk.ptr);
                        }
                        std::vector<BlockSlice<StrItem>>().swap(shard_items[t][s]);
                    }
                }
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

        const uint32_t num_nodes = static_cast<uint32_t>(num_nodes_sz);
        auto nodes = std::make_unique_for_overwrite<UserId64[]>(num_nodes);
        if (num_nodes >= 8192 && ncpu > 1) {
            std::atomic<size_t> next_s{0};
            pool.run([&](size_t) {
                for (;;) {
                    const size_t s = next_s.fetch_add(1, std::memory_order_relaxed);
                    if (s >= kShards) break;
                    if (!shard_nodes[s].empty()) {
                        std::memcpy(nodes.get() + shard_base[s], shard_nodes[s].data(),
                                    shard_nodes[s].size() * sizeof(UserId64));
                        std::vector<UserId64>().swap(shard_nodes[s]);
                    }
                }
            });
        } else {
            for (size_t s = 0; s < kShards; ++s) {
                if (!shard_nodes[s].empty()) {
                    std::memcpy(nodes.get() + shard_base[s], shard_nodes[s].data(),
                                shard_nodes[s].size() * sizeof(UserId64));
                    std::vector<UserId64>().swap(shard_nodes[s]);
                }
            }
        }
        tmp_arenas.clear();
        const auto lt3 = std::chrono::steady_clock::now();

        // Sub-phase 3: Union-Find + canonical min_node + 64-subtable lock-free LinkTable build
        size_t shard_non_id[kShards]{};
        auto canon_idx = std::make_unique_for_overwrite<uint32_t[]>(num_nodes);

        if (num_nodes < 8192 || ncpu == 1) {
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
                    const uint32_t ea = worker[i];
                    const uint32_t eb = worker[i + 1];
                    uint32_t ra = find(shard_base[ea >> 26] + (ea & ((1u << 26) - 1u)));
                    uint32_t rb = find(shard_base[eb >> 26] + (eb & ((1u << 26) - 1u)));
                    if (ra == rb) continue;
                    if (rank[ra] < rank[rb]) std::swap(ra, rb);
                    parent[rb] = ra;
                    if (rank[ra] == rank[rb]) ++rank[ra];
                }
            }

            std::vector<uint32_t> min_node(num_nodes);
            for (uint32_t i = 0; i < num_nodes; ++i) min_node[i] = i;
            for (uint32_t i = 0; i < num_nodes; ++i) {
                const uint32_t r = find(i);
                parent[i] = r;
                if (i != min_node[r] &&
                    UserId64::less(nodes[i], nodes[min_node[r]], all_user_arenas)) {
                    min_node[r] = i;
                }
            }

            for (size_t s = 0; s < kShards; ++s) {
                const uint32_t lo = shard_base[s];
                const uint32_t hi = (s + 1 < kShards) ? shard_base[s + 1] : num_nodes;
                size_t sn = 0;
                for (uint32_t i = lo; i < hi; ++i) {
                    const uint32_t c = min_node[parent[i]];
                    canon_idx[i] = c;
                    if (c != i) ++sn;
                }
                shard_non_id[s] = sn;
            }
        } else {
            auto parent = std::make_unique_for_overwrite<std::atomic<uint32_t>[]>(num_nodes);
            auto min_node = std::make_unique_for_overwrite<std::atomic<uint32_t>[]>(num_nodes);

            pool.run([&](size_t tid) {
                const uint32_t lo = static_cast<uint32_t>((uint64_t(num_nodes) * tid) / ncpu);
                const uint32_t hi = static_cast<uint32_t>((uint64_t(num_nodes) * (tid + 1)) / ncpu);
                for (uint32_t i = lo; i < hi; ++i) {
                    parent[i].store(i, std::memory_order_relaxed);
                    min_node[i].store(i, std::memory_order_relaxed);
                }
            });

            struct EdgeChunk {
                const uint32_t* ptr;
                size_t count;
            };
            std::vector<EdgeChunk> edge_chunks;
            for (const auto& w : endpoints) {
                for (size_t pos = 0; pos < w.size(); pos += 8192) {
                    edge_chunks.push_back(
                        EdgeChunk{w.data() + pos, std::min<size_t>(8192, w.size() - pos)});
                }
            }

            {
                std::atomic<size_t> next_ec{0};
                pool.run([&](size_t) {
                    auto unite = [&](uint32_t u, uint32_t v) {
                        for (;;) {
                            for (;;) {
                                const uint32_t pu = parent[u].load(std::memory_order_relaxed);
                                if (pu == u) break;
                                const uint32_t ppu = parent[pu].load(std::memory_order_relaxed);
                                if (pu != ppu) {
                                    uint32_t exp = pu;
                                    parent[u].compare_exchange_weak(exp, ppu,
                                                                    std::memory_order_relaxed);
                                }
                                u = ppu;
                            }
                            for (;;) {
                                const uint32_t pv = parent[v].load(std::memory_order_relaxed);
                                if (pv == v) break;
                                const uint32_t ppv = parent[pv].load(std::memory_order_relaxed);
                                if (pv != ppv) {
                                    uint32_t exp = pv;
                                    parent[v].compare_exchange_weak(exp, ppv,
                                                                    std::memory_order_relaxed);
                                }
                                v = ppv;
                            }
                            if (u == v) return;
                            if (u > v) std::swap(u, v);
                            uint32_t expected = v;
                            if (parent[v].compare_exchange_weak(expected, u,
                                                                std::memory_order_relaxed)) {
                                return;
                            }
                        }
                    };

                    for (;;) {
                        const size_t ci = next_ec.fetch_add(1, std::memory_order_relaxed);
                        if (ci >= edge_chunks.size()) break;
                        const uint32_t* ep = edge_chunks[ci].ptr;
                        const size_t n = edge_chunks[ci].count;
                        for (size_t i = 0; i < n; i += 2) {
                            const uint32_t ea = ep[i];
                            const uint32_t eb = ep[i + 1];
                            unite(shard_base[ea >> 26] + (ea & ((1u << 26) - 1u)),
                                  shard_base[eb >> 26] + (eb & ((1u << 26) - 1u)));
                        }
                    }
                });
            }

            pool.run([&](size_t tid) {
                const uint32_t lo = static_cast<uint32_t>((uint64_t(num_nodes) * tid) / ncpu);
                const uint32_t hi = static_cast<uint32_t>((uint64_t(num_nodes) * (tid + 1)) / ncpu);
                for (uint32_t i = lo; i < hi; ++i) {
                    uint32_t r = i;
                    for (;;) {
                        const uint32_t p = parent[r].load(std::memory_order_relaxed);
                        if (p == r) break;
                        r = p;
                    }
                    canon_idx[i] = r;
                    uint32_t cur = min_node[r].load(std::memory_order_relaxed);
                    while (i != cur &&
                           UserId64::less(nodes[i], nodes[cur], all_user_arenas)) {
                        if (min_node[r].compare_exchange_weak(cur, i, std::memory_order_relaxed)) {
                            break;
                        }
                    }
                }
            });

            std::atomic<size_t> next_s{0};
            pool.run([&](size_t) {
                for (;;) {
                    const size_t s = next_s.fetch_add(1, std::memory_order_relaxed);
                    if (s >= kShards) break;
                    const uint32_t lo = shard_base[s];
                    const uint32_t hi = (s + 1 < kShards) ? shard_base[s + 1] : num_nodes;
                    size_t sn = 0;
                    for (uint32_t i = lo; i < hi; ++i) {
                        const uint32_t c = min_node[canon_idx[i]].load(std::memory_order_relaxed);
                        canon_idx[i] = c;
                        if (c != i) ++sn;
                    }
                    shard_non_id[s] = sn;
                }
            });
        }

        std::vector<std::vector<uint32_t>>().swap(endpoints);
        size_t non_id_count = 0;
        size_t max_sn = 0;
        for (size_t s = 0; s < kShards; ++s) {
            non_id_count += shard_non_id[s];
            if (shard_non_id[s] > max_sn) max_sn = shard_non_id[s];
        }
        if (non_id_count == 0) return;
        has_links_ = true;

        auto log2_p2 = [](size_t v) -> unsigned {
            unsigned r = 0;
            while ((size_t(1) << r) < v) ++r;
            return r;
        };

        size_t sub_bloom_words = 64;
        while (sub_bloom_words * 4 < max_sn) sub_bloom_words <<= 1;
        sub_bloom_bits_ = log2_p2(sub_bloom_words);
        sub_bloom_mask_ = sub_bloom_words - 1;
        bloom_.assign_zero(kShards << sub_bloom_bits_);

        size_t sub_slot_cap = 16;
        while (sub_slot_cap * 82 < max_sn * 100) sub_slot_cap <<= 1;
        sub_slot_bits_ = log2_p2(sub_slot_cap);
        sub_slot_mask_ = sub_slot_cap - 1;
        slots_.assign_zero(kShards << sub_slot_bits_);

        auto populate_shard = [&](size_t s) {
            const uint32_t lo = shard_base[s];
            const uint32_t hi = (s + 1 < kShards) ? shard_base[s + 1] : num_nodes;
            const size_t b_base = s << sub_bloom_bits_;
            const size_t s_base = s << sub_slot_bits_;
            for (uint32_t i = lo; i < hi; ++i) {
                const uint32_t c = canon_idx[i];
                if (c == i) continue;
                const UserId64 k = nodes[i];
                const UserId64 v = nodes[c];
                const uint64_t h = k.hash(all_user_arenas);
                bloom_[b_base + (static_cast<size_t>(h) & sub_bloom_mask_)] |= bloom_bits(h);
                size_t off = static_cast<size_t>(h) & sub_slot_mask_;
                while (!slots_[s_base + off].key.empty()) off = (off + 1) & sub_slot_mask_;
                slots_[s_base + off] = LinkSlot{k, v};
            }
        };

        if (num_nodes < 8192 || ncpu == 1) {
            for (size_t s = 0; s < kShards; ++s) populate_shard(s);
        } else {
            std::atomic<size_t> next_s{0};
            pool.run([&](size_t) {
                for (;;) {
                    const size_t s = next_s.fetch_add(1, std::memory_order_relaxed);
                    if (s >= kShards) break;
                    populate_shard(s);
                }
            });
        }

        const auto lt4 = std::chrono::steady_clock::now();
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        std::fprintf(stderr,
                     "links breakdown: read=%.1f shard=%.1f flatten=%.1f uf+tbl=%.1f | "
                     "nodes=%u non_id=%zu slots=%.1fMB bloom=%.1fMB\n",
                     ms(lt0, lt1), ms(lt1, lt2), ms(lt2, lt3), ms(lt3, lt4), num_nodes,
                     non_id_count, slots_.size() * 16.0 / (1024.0 * 1024.0),
                     bloom_.size() * 8.0 / (1024.0 * 1024.0));
    }

    bool empty() const noexcept { return !has_links_; }

    [[gnu::always_inline]] inline void prefetch_bloom(uint64_t h) const noexcept {
        const size_t s = static_cast<size_t>(h >> kShardShift);
        __builtin_prefetch(
            &bloom_[(s << sub_bloom_bits_) | (static_cast<size_t>(h) & sub_bloom_mask_)], 0, 1);
    }

    [[gnu::always_inline]] inline bool may_contain(uint64_t h) const noexcept {
        const size_t s = static_cast<size_t>(h >> kShardShift);
        const uint64_t w =
            bloom_[(s << sub_bloom_bits_) | (static_cast<size_t>(h) & sub_bloom_mask_)];
        const uint64_t b = bloom_bits(h);
        return (w & b) == b;
    }

    [[gnu::always_inline]] inline void prefetch_slot(uint64_t h) const noexcept {
        const size_t s = static_cast<size_t>(h >> kShardShift);
        __builtin_prefetch(
            &slots_[(s << sub_slot_bits_) | (static_cast<size_t>(h) & sub_slot_mask_)], 0, 1);
    }

    [[gnu::always_inline]] inline bool resolve_with_hash(
        UserId64 key, uint64_t h, UserId64& out_canon,
        const OverflowArena* const* all_user_arenas) const noexcept {
        const size_t base = static_cast<size_t>(h >> kShardShift) << sub_slot_bits_;
        size_t off = static_cast<size_t>(h) & sub_slot_mask_;
        for (;;) {
            const LinkSlot& sl = slots_[base + off];
            if (sl.key.raw == 0) return false;
            if (UserId64::equals(sl.key, key, all_user_arenas)) {
                out_canon = sl.canon;
                return true;
            }
            off = (off + 1) & sub_slot_mask_;
        }
    }

    [[gnu::always_inline]] inline bool resolve_sv_with_hash(
        std::string_view sv, uint64_t h, UserId64& out_canon,
        const OverflowArena* const* all_user_arenas) const noexcept {
        const size_t base = static_cast<size_t>(h >> kShardShift) << sub_slot_bits_;
        size_t off = static_cast<size_t>(h) & sub_slot_mask_;
        for (;;) {
            const LinkSlot& sl = slots_[base + off];
            if (sl.key.raw == 0) return false;
            if (UserId64::equals_sv(sl.key, sv, all_user_arenas)) {
                out_canon = sl.canon;
                return true;
            }
            off = (off + 1) & sub_slot_mask_;
        }
    }

   private:
    [[gnu::always_inline]] static inline uint64_t bloom_bits(uint64_t h) noexcept {
        return (1ULL << ((h >> 24) & 63u)) | (1ULL << ((h >> 30) & 63u)) |
               (1ULL << ((h >> 36) & 63u));
    }

    OverflowArena node_arenas_[kShards];
    HugeBuffer<uint64_t> bloom_;
    HugeBuffer<LinkSlot> slots_;
    size_t sub_bloom_mask_ = 0;
    size_t sub_slot_mask_ = 0;
    unsigned sub_bloom_bits_ = 0;
    unsigned sub_slot_bits_ = 0;
    bool has_links_ = false;
};
