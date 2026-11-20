#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "csv_parse.hpp"
#include "hash_util.hpp"
#include "io_util.hpp"

class LinkTable {
   public:
    struct NodeSlice {
        uint32_t off;
        uint32_t len;
    };

    void build(const std::string& links_dir, size_t ncpu) {
        auto files = map_all_csvs(links_dir);
        auto slices = build_slices(files);
        if (slices.empty()) return;

        struct RawPair {
            uint32_t off_a, len_a;
            uint32_t off_b, len_b;
        };
        struct ThreadBuf {
            OverflowArena arena;
            std::vector<RawPair> pairs;
        };

        const size_t nthreads = std::min(std::max<size_t>(1, ncpu), slices.size());
        std::vector<ThreadBuf> tbufs(nthreads);
        std::atomic<size_t> next_slice{0};

        auto worker = [&](size_t tid) {
            auto& tb = tbufs[tid];
            std::string_view f[2];
            for (;;) {
                const size_t si = next_slice.fetch_add(1, std::memory_order_relaxed);
                if (si >= slices.size()) break;
                const FileSlice& s = slices[si];
                const char* p = s.base + s.start;
                const char* end = s.base + s.end;
                while (p < end) {
                    const char* nl =
                        static_cast<const char*>(std::memchr(p, '\n', static_cast<size_t>(end - p)));
                    const char* lend = nl ? nl : end;
                    const size_t len = static_cast<size_t>(lend - p);
                    if (split_csv_line<2>(p, len, f) && !f[0].empty() && !f[1].empty()) {
                        const uint32_t oa = tb.arena.push(f[0]);
                        const uint32_t ob = tb.arena.push(f[1]);
                        tb.pairs.push_back(RawPair{oa, static_cast<uint32_t>(f[0].size()), ob,
                                                   static_cast<uint32_t>(f[1].size())});
                    }
                    p = nl ? (nl + 1) : end;
                }
            }
        };

        if (nthreads == 1) {
            worker(0);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(nthreads);
            for (size_t t = 0; t < nthreads; ++t) {
                threads.emplace_back(worker, t);
            }
            for (auto& th : threads) th.join();
        }
        files.clear();

        size_t total_pairs = 0;
        for (const auto& tb : tbufs) total_pairs += tb.pairs.size();
        if (total_pairs == 0) return;

        id_map_.reserve(total_pairs * 2);
        nodes_.reserve(total_pairs);
        std::vector<uint32_t> parent;
        std::vector<uint8_t> rank;
        std::vector<uint32_t> min_node;
        parent.reserve(total_pairs);
        rank.reserve(total_pairs);
        min_node.reserve(total_pairs);

        auto intern = [&](std::string_view sv) -> uint32_t {
            const uint32_t next_id = static_cast<uint32_t>(nodes_.size());
            auto [val_ptr, inserted] = id_map_.emplace(sv, next_id);
            if (!inserted) return *val_ptr;
            const uint32_t off = node_arena_.push(sv);
            nodes_.push_back(NodeSlice{off, static_cast<uint32_t>(sv.size())});
            parent.push_back(next_id);
            rank.push_back(0);
            min_node.push_back(next_id);
            return next_id;
        };

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

        for (auto& tb : tbufs) {
            for (const RawPair& rp : tb.pairs) {
                const uint32_t a = intern(tb.arena.get(rp.off_a, rp.len_a));
                const uint32_t b = intern(tb.arena.get(rp.off_b, rp.len_b));
                uint32_t ra = find(a);
                uint32_t rb = find(b);
                if (ra == rb) continue;
                const uint32_t mn =
                    byte_less(node_sv(min_node[ra]), node_sv(min_node[rb])) ? min_node[ra]
                                                                             : min_node[rb];
                if (rank[ra] < rank[rb]) std::swap(ra, rb);
                parent[rb] = ra;
                if (rank[ra] == rank[rb]) ++rank[ra];
                min_node[ra] = mn;
            }
            tb.arena.free_all();
            std::vector<RawPair>().swap(tb.pairs);
        }

        const size_t n = nodes_.size();
        canonical_id_.resize(n);
        node_hash_.resize(n);
        for (size_t i = 0; i < n; ++i) {
            canonical_id_[i] = min_node[find(static_cast<uint32_t>(i))];
            node_hash_[i] = wyhash64(node_sv(static_cast<uint32_t>(i)));
        }
    }

    bool empty() const noexcept { return id_map_.empty(); }

    std::optional<uint32_t> resolve_user(std::string_view sv) const noexcept {
        if (id_map_.empty()) return std::nullopt;
        const uint32_t* id = id_map_.find(sv);
        if (!id) return std::nullopt;
        return canonical_id_[*id];
    }

    std::string_view node_sv(uint32_t node_id) const noexcept {
        const NodeSlice& ns = nodes_[node_id];
        return node_arena_.get(ns.off, ns.len);
    }

    uint64_t canonical_hash(uint32_t canon_node_id) const noexcept {
        return node_hash_[canon_node_id];
    }

   private:
    OverflowArena node_arena_;
    std::vector<NodeSlice> nodes_;
    FlatStrMap<uint32_t> id_map_;
    std::vector<uint32_t> canonical_id_;
    std::vector<uint64_t> node_hash_;
};
