#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

#include "csv_parse.hpp"
#include "hash_util.hpp"
#include "io_util.hpp"
#include "thread_pool.hpp"

class LinkTable {
   public:
    static constexpr size_t kShards = 32;

    struct LinkSlot {
        CompactStr key;
        uint32_t val;
    };
    static_assert(sizeof(LinkSlot) == 16);

    struct SubTable {
        std::vector<LinkSlot> slots;
        size_t mask = 0;
        size_t count = 0;
    };

    struct NodeSlice {
        uint32_t off;
        uint32_t len;
    };

    void build(const std::string& links_dir, ThreadPool& pool) {
        auto files = open_all_csvs(links_dir);
        auto chunks = build_pread_chunks(files);
        if (chunks.empty()) return;

        const size_t ncpu = pool.size();

        struct RawEdge {
            CompactStr a;
            CompactStr b;
            uint64_t ha;
            uint64_t hb;
        };
        struct StrItem {
            CompactStr s;
            uint64_t h;
        };

        std::vector<OverflowArena> tmp_arenas(ncpu);
        std::vector<ChunkedVec<RawEdge>> raw_edges(ncpu);
        std::vector<std::vector<ChunkedVec<StrItem>>> str_buckets(ncpu);
        for (size_t t = 0; t < ncpu; ++t) str_buckets[t].resize(kShards);

        {
            std::atomic<size_t> next_chunk{0};
            pool.run([&](size_t tid) {
                std::vector<char> buf;
                buf.reserve((size_t(2) << 20) + 65536 + 64);
                auto& arena = tmp_arenas[tid];
                auto& edges = raw_edges[tid];
                auto& sbuckets = str_buckets[tid];
                const uint16_t aid = static_cast<uint16_t>(tid);

                for (;;) {
                    const size_t ci = next_chunk.fetch_add(1, std::memory_order_relaxed);
                    if (ci >= chunks.size()) break;
                    size_t data_start = 0, limit = 0, blen = 0;
                    read_chunk_into(chunks[ci], buf, data_start, limit, blen);
                    if (data_start >= limit) continue;

                    for_each_csv_line<2>(
                        buf.data(), data_start, limit, blen, [&](const std::string_view (&f)[2]) {
                            if (f[0].empty() || f[1].empty()) return;
                            uint64_t ha = wyhash64(f[0]);
                            uint64_t hb = wyhash64(f[1]);
                            if (ha == 0) ha = 1;
                            if (hb == 0) hb = 1;
                            const CompactStr ca = CompactStr::make_str(f[0], aid, arena);
                            const CompactStr cb = CompactStr::make_str(f[1], aid, arena);
                            edges.push_back(RawEdge{ca, cb, ha, hb});
                            sbuckets[static_cast<size_t>(ha >> 59)].push_back(StrItem{ca, ha});
                            sbuckets[static_cast<size_t>(hb >> 59)].push_back(StrItem{cb, hb});
                        });
                }
            });
        }
        files.clear();

        size_t total_edges = 0;
        for (const auto& re : raw_edges) total_edges += re.size();
        if (total_edges == 0) return;

        // Build per-shard LinkSlot hash tables + local unique strings in parallel
        struct ShardUniq {
            std::vector<uint64_t> hashes;
        };
        std::vector<ShardUniq> shard_uniqs(kShards);
        node_arenas_.resize(kShards);
        sub_tables_.resize(kShards);

        {
            std::atomic<size_t> next_b{0};
            pool.run([&](size_t) {
                const OverflowArena* tmp_ptr = tmp_arenas.data();
                const OverflowArena* node_ptr = node_arenas_.data();
                for (;;) {
                    const size_t b = next_b.fetch_add(1, std::memory_order_relaxed);
                    if (b >= kShards) break;

                    size_t raw_cnt = 0;
                    for (size_t t = 0; t < ncpu; ++t) raw_cnt += str_buckets[t][b].size();
                    if (raw_cnt == 0) continue;

                    size_t cap = 64;
                    while (cap * 11 < raw_cnt * 15) cap <<= 1;
                    auto& st = sub_tables_[b];
                    st.slots.assign(cap, LinkSlot{});
                    st.mask = cap - 1;
                    st.count = 0;

                    auto& su = shard_uniqs[b];
                    auto& b_arena = node_arenas_[b];
                    const uint16_t b_aid = static_cast<uint16_t>(b);

                    auto rehash_st = [&]() {
                        const size_t ncap = st.slots.size() * 2;
                        std::vector<LinkSlot> nxt(ncap);
                        const size_t nmask = ncap - 1;
                        for (const auto& sl : st.slots) {
                            if (sl.key.empty_sentinel()) continue;
                            const uint64_t h = su.hashes[sl.val];
                            size_t idx = static_cast<size_t>(h) & nmask;
                            while (!nxt[idx].key.empty_sentinel()) idx = (idx + 1) & nmask;
                            nxt[idx] = sl;
                        }
                        st.slots.swap(nxt);
                        st.mask = nmask;
                    };

                    for (size_t t = 0; t < ncpu; ++t) {
                        str_buckets[t][b].for_each_block([&](const StrItem* ptr, size_t n) {
                            for (size_t i = 0; i < n; ++i) {
                                const StrItem& it = ptr[i];
                                if ((st.count + 1) * 10 >= st.slots.size() * 6) rehash_st();
                                size_t idx = static_cast<size_t>(it.h) & st.mask;
                                for (;;) {
                                    LinkSlot& sl = st.slots[idx];
                                    if (sl.key.empty_sentinel()) {
                                        const uint32_t local_id =
                                            static_cast<uint32_t>(su.hashes.size());
                                        std::string_view sv = it.s.raw_view(tmp_ptr);
                                        if (sv.size() <= 11) {
                                            sl.key = it.s;
                                        } else {
                                            const uint32_t off = b_arena.push(sv);
                                            sl.key = CompactStr::make_overflow(
                                                off, static_cast<uint32_t>(sv.size()), b_aid);
                                        }
                                        sl.val = local_id;
                                        su.hashes.push_back(it.h);
                                        ++st.count;
                                        break;
                                    }
                                    if (CompactStr::equals_cross(sl.key, node_ptr, it.s, tmp_ptr)) {
                                        break;
                                    }
                                    idx = (idx + 1) & st.mask;
                                }
                            }
                        });
                        str_buckets[t][b].clear_and_free();
                    }
                }
            });
        }
        std::vector<std::vector<ChunkedVec<StrItem>>>().swap(str_buckets);

        std::vector<uint32_t> shard_base(kShards + 1, 0);
        for (size_t b = 0; b < kShards; ++b) {
            shard_base[b + 1] = shard_base[b] + static_cast<uint32_t>(shard_uniqs[b].hashes.size());
        }
        const uint32_t num_nodes = shard_base[kShards];
        nodes_.resize(num_nodes);
        node_hash_.resize(num_nodes);

        // Flatten node metadata, offset sub_table slot values to global node IDs, and convert RawEdges to (u, v)
        struct EdgePair {
            uint32_t a;
            uint32_t b;
        };
        std::vector<std::vector<EdgePair>> resolved_edges(ncpu);
        {
            std::atomic<size_t> next_b{0};
            pool.run([&](size_t) {
                for (;;) {
                    const size_t b = next_b.fetch_add(1, std::memory_order_relaxed);
                    if (b >= kShards) break;
                    const uint32_t base = shard_base[b];
                    auto& st = sub_tables_[b];
                    const auto& su = shard_uniqs[b];
                    const size_t n = su.hashes.size();
                    for (size_t i = 0; i < n; ++i) {
                        nodes_[base + i] = CompactStr{};  // filled below
                        node_hash_[base + i] = su.hashes[i];
                    }
                    for (auto& sl : st.slots) {
                        if (!sl.key.empty_sentinel()) {
                            const uint32_t gid = base + sl.val;
                            sl.val = gid;
                            nodes_[gid] = sl.key;
                        }
                    }
                }
            });
        }
        // Now resolve RawEdges in parallel across ncpu threads (read-only on sub_tables_)
        {
            const OverflowArena* tmp_ptr = tmp_arenas.data();
            const OverflowArena* node_ptr = node_arenas_.data();
            pool.run([&](size_t tid) {
                resolved_edges[tid].reserve(raw_edges[tid].size());
                raw_edges[tid].for_each_block([&](const RawEdge* ptr, size_t n) {
                    for (size_t i = 0; i < n; ++i) {
                        const RawEdge& e = ptr[i];
                        uint32_t ia = 0, ib = 0;
                        lookup_cross_raw(e.a, e.ha, tmp_ptr, node_ptr, ia);
                        lookup_cross_raw(e.b, e.hb, tmp_ptr, node_ptr, ib);
                        resolved_edges[tid].push_back(EdgePair{ia, ib});
                    }
                });
                raw_edges[tid].clear_and_free();
                tmp_arenas[tid].free_all();
            });
        }
        std::vector<ChunkedVec<RawEdge>>().swap(raw_edges);
        std::vector<OverflowArena>().swap(tmp_arenas);
        std::vector<ShardUniq>().swap(shard_uniqs);

        // Fast integer Union-Find
        std::vector<uint32_t> parent(num_nodes);
        std::vector<uint8_t> rank(num_nodes, 0);
        std::vector<uint32_t> min_node(num_nodes);
        for (uint32_t i = 0; i < num_nodes; ++i) {
            parent[i] = i;
            min_node[i] = i;
        }
        auto find = [&](uint32_t x) -> uint32_t {
            uint32_t r = x;
            while (parent[r] != r) r = parent[r];
            while (parent[x] != r) {
                uint32_t nxt = parent[x];
                parent[x] = r;
                x = nxt;
            }
            return r;
        };
        for (auto& vec : resolved_edges) {
            for (const EdgePair& ep : vec) {
                uint32_t ra = find(ep.a);
                uint32_t rb = find(ep.b);
                if (ra == rb) continue;
                const uint32_t mn =
                    byte_less(node_sv(min_node[ra]), node_sv(min_node[rb])) ? min_node[ra]
                                                                             : min_node[rb];
                if (rank[ra] < rank[rb]) std::swap(ra, rb);
                parent[rb] = ra;
                if (rank[ra] == rank[rb]) ++rank[ra];
                min_node[ra] = mn;
            }
            std::vector<EdgePair>().swap(vec);
        }
        std::vector<std::vector<EdgePair>>().swap(resolved_edges);

        std::vector<uint32_t> canonical_id(num_nodes);
        for (uint32_t i = 0; i < num_nodes; ++i) {
            canonical_id[i] = min_node[find(i)];
        }

        // Update sub_tables_ values directly to canonical_id in parallel
        {
            std::atomic<size_t> next_b{0};
            pool.run([&](size_t) {
                for (;;) {
                    const size_t b = next_b.fetch_add(1, std::memory_order_relaxed);
                    if (b >= kShards) break;
                    for (auto& sl : sub_tables_[b].slots) {
                        if (!sl.key.empty_sentinel()) {
                            sl.val = canonical_id[sl.val];
                        }
                    }
                }
            });
        }
    }

    bool empty() const noexcept { return nodes_.empty(); }

    void prefetch(uint64_t h) const noexcept {
        if (nodes_.empty()) return;
        const SubTable& st = sub_tables_[static_cast<size_t>(h >> 59)];
        if (!st.slots.empty()) {
            __builtin_prefetch(&st.slots[static_cast<size_t>(h) & st.mask], 0, 1);
        }
    }

    bool resolve_with_hash(const CompactStr& key, uint64_t h, uint32_t& out_canon_id,
                           const OverflowArena* key_arenas) const noexcept {
        if (nodes_.empty()) return false;
        const SubTable& st = sub_tables_[static_cast<size_t>(h >> 59)];
        if (st.slots.empty()) return false;
        const OverflowArena* node_ptr = node_arenas_.data();
        size_t idx = static_cast<size_t>(h) & st.mask;
        for (;;) {
            const LinkSlot& sl = st.slots[idx];
            if (sl.key.empty_sentinel()) return false;
            if (CompactStr::equals_cross(sl.key, node_ptr, key, key_arenas)) {
                out_canon_id = sl.val;
                return true;
            }
            idx = (idx + 1) & st.mask;
        }
    }

    bool resolve_sv_with_hash(std::string_view sv, uint64_t h,
                              uint32_t& out_canon_id) const noexcept {
        if (nodes_.empty()) return false;
        const SubTable& st = sub_tables_[static_cast<size_t>(h >> 59)];
        if (st.slots.empty()) return false;
        const OverflowArena* node_ptr = node_arenas_.data();
        size_t idx = static_cast<size_t>(h) & st.mask;
        for (;;) {
            const LinkSlot& sl = st.slots[idx];
            if (sl.key.empty_sentinel()) return false;
            if (CompactStr::equals_sv(sl.key, node_ptr, sv)) {
                out_canon_id = sl.val;
                return true;
            }
            idx = (idx + 1) & st.mask;
        }
    }

    std::string_view node_sv(uint32_t node_id) const noexcept {
        return nodes_[node_id].raw_view(node_arenas_.data());
    }

    uint64_t canonical_hash(uint32_t canon_node_id) const noexcept {
        return node_hash_[canon_node_id];
    }

   private:
    void lookup_cross_raw(const CompactStr& key, uint64_t h, const OverflowArena* key_arenas,
                          const OverflowArena* node_ptr, uint32_t& out_gid) const noexcept {
        const SubTable& st = sub_tables_[static_cast<size_t>(h >> 59)];
        size_t idx = static_cast<size_t>(h) & st.mask;
        for (;;) {
            const LinkSlot& sl = st.slots[idx];
            if (CompactStr::equals_cross(sl.key, node_ptr, key, key_arenas)) {
                out_gid = sl.val;
                return;
            }
            idx = (idx + 1) & st.mask;
        }
    }

    std::vector<OverflowArena> node_arenas_;
    std::vector<SubTable> sub_tables_;
    std::vector<CompactStr> nodes_;
    std::vector<uint64_t> node_hash_;
};
