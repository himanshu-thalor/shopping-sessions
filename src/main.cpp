#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <thread>
#include <unistd.h>
#include <vector>

#include "csv_parse.hpp"
#include "hash_util.hpp"
#include "io_util.hpp"
#include "parquet_writer.hpp"
#include "thread_pool.hpp"
#include "union_find.hpp"

namespace {

constexpr int64_t kLateCutoffMs = 86400000LL;
constexpr int64_t kSessionGapMs = 1800000LL;
constexpr int64_t kClickReachMs = 7LL * 86400000LL;
constexpr uint32_t kFastTombstone = std::numeric_limits<uint32_t>::max();
constexpr int64_t kSlowTombstone = std::numeric_limits<int64_t>::min();

template <bool FastPack>
struct Traits;

template <>
struct Traits<true> {
    static constexpr size_t kEvBlockCap = 1024;
    static constexpr size_t kSurvEvBlockCap = 2048;
    static constexpr size_t kSurvClkBlockCap = 1024;
    static constexpr size_t kPoolBlockBytes = 49152;  // 48 KB: fits 1024*40B, 2048*24B, 1024*36B

    struct RawEvent {
        CompactStr event_id;
        CompactStr user_id;
        uint32_t ts_rel;          // kFastTombstone == invalid/late/duplicate-loser
        int32_t ingest_rel_sec;   // (ingest_ms - event_win_start) / 1000, clamped
        uint32_t price_cat;       // (cat_id << 24) | price_cents
        uint32_t eid_hash32;

        bool is_tombstone() const noexcept { return ts_rel == kFastTombstone; }
        void mark_tombstone() noexcept { ts_rel = kFastTombstone; }
        bool older_than(const RawEvent& other) const noexcept {
            return ingest_rel_sec < other.ingest_rel_sec;
        }
    };
    static_assert(sizeof(RawEvent) == 40);

    struct SurvivingEvent {
        CompactStr user_id;
        uint32_t ts_rel;
        uint32_t price_cat;
        uint32_t person_hash32;
    };
    static_assert(sizeof(SurvivingEvent) == 24);

    struct SurvivingClick {
        CompactStr user_id;
        CompactStr click_id;
        uint32_t ts_rel;
        uint32_t camp_id;
        uint32_t person_hash32;
    };
    static_assert(sizeof(SurvivingClick) == 36);

    struct ShardEvent {
        uint64_t packed;  // (uint64_t(ts_rel) << 32) | price_cat
    };
    static_assert(sizeof(ShardEvent) == 8);

    struct ShardClick {
        uint32_t ts_rel;
        uint32_t camp_id;
        CompactStr click_id;
    };
};

template <>
struct Traits<false> {
    static constexpr size_t kEvBlockCap = 1024;
    static constexpr size_t kSurvEvBlockCap = 1024;
    static constexpr size_t kSurvClkBlockCap = 1024;
    static constexpr size_t kPoolBlockBytes = 65536;  // 64 KB: fits 1024*56B

    struct RawEvent {
        uint64_t eid_hash;
        int64_t ingest_ms;
        int64_t ts_ms;  // kSlowTombstone == invalid/late/duplicate-loser
        CompactStr event_id;
        CompactStr user_id;
        uint32_t price_cents;
        uint32_t cat_id;
        uint32_t eid_hash32;
        uint32_t pad;

        bool is_tombstone() const noexcept { return ts_ms == kSlowTombstone; }
        void mark_tombstone() noexcept { ts_ms = kSlowTombstone; }
        bool older_than(const RawEvent& other) const noexcept {
            return ingest_ms < other.ingest_ms;
        }
    };

    struct SurvivingEvent {
        CompactStr user_id;
        uint32_t price_cents;
        int64_t ts_ms;
        uint64_t person_hash;
        uint32_t cat_id;
        uint32_t pad;
    };

    struct SurvivingClick {
        int64_t ts_ms;
        uint64_t person_hash;
        CompactStr user_id;
        CompactStr click_id;
        uint32_t camp_id;
        uint32_t pad;
    };

    struct ShardEvent {
        int64_t ts_ms;
        uint32_t price_cents;
        uint32_t cat_id;
    };

    struct ShardClick {
        int64_t ts_ms;
        uint32_t camp_id;
        CompactStr click_id;
    };
};

struct SessionInfo {
    int64_t start_key;
    uint64_t rev;
    int32_t day;
    uint8_t has_purch;
};

struct CampCreditAgg {
    int32_t sessions = 0;
    int32_t purchasing_sessions = 0;
    int64_t revenue_cents = 0;
};
static_assert(sizeof(CampCreditAgg) == 16);

struct CampDayFull {
    int32_t sessions = 0;
    int32_t purchasing_sessions = 0;
    int64_t revenue_cents = 0;
    int32_t assisted_sessions = 0;
};

struct CampDayHashMap {
    struct Slot {
        uint64_t key = std::numeric_limits<uint64_t>::max();
        CampDayFull agg{};
    };
    std::vector<Slot> table;
    size_t count = 0;

    static uint64_t pack_key(uint32_t camp_id, int32_t day) noexcept {
        return (static_cast<uint64_t>(camp_id) << 32) | static_cast<uint32_t>(day);
    }

    static uint64_t hash_key(uint64_t k) noexcept {
        k ^= k >> 33;
        k *= 0xff51afd7ed558ccdULL;
        k ^= k >> 33;
        k *= 0xc4ceb9fe1a85ec53ULL;
        k ^= k >> 33;
        return k;
    }

    CampDayFull& get(uint32_t camp_id, int32_t day) {
        if ((count + 1) * 2 >= table.size()) {
            rehash(table.empty() ? 1024 : table.size() * 2);
        }
        const uint64_t k = pack_key(camp_id, day);
        const size_t mask = table.size() - 1;
        size_t idx = static_cast<size_t>(hash_key(k)) & mask;
        for (;;) {
            if (table[idx].key == std::numeric_limits<uint64_t>::max()) {
                table[idx].key = k;
                ++count;
                return table[idx].agg;
            }
            if (table[idx].key == k) return table[idx].agg;
            idx = (idx + 1) & mask;
        }
    }

    void rehash(size_t new_cap) {
        std::vector<Slot> next(new_cap, Slot{std::numeric_limits<uint64_t>::max(), {}});
        const size_t mask = new_cap - 1;
        for (const auto& s : table) {
            if (s.key == std::numeric_limits<uint64_t>::max()) continue;
            size_t idx = static_cast<size_t>(hash_key(s.key)) & mask;
            while (next[idx].key != std::numeric_limits<uint64_t>::max()) idx = (idx + 1) & mask;
            next[idx] = s;
        }
        table.swap(next);
    }
};

struct DedupIdx {
    uint32_t idx_plus1 = 0;  // 0 == empty
    uint32_t hash32 = 0;
};

struct PersonMapSlot {
    uint32_t hash32 = 0;  // 0 == empty
    uint32_t local_pid = 0;
    CompactStr user_id{};
};
static_assert(sizeof(PersonMapSlot) == 20);

inline void radix_sort_events_u64(uint64_t* arr, size_t n, std::vector<uint64_t>& scratch) {
    scratch.resize(n);
    uint64_t* src = arr;
    uint64_t* dst = scratch.data();
    for (unsigned shift = 32; shift < 64; shift += 8) {
        uint32_t cnt[256]{};
        for (size_t i = 0; i < n; ++i) {
            ++cnt[(src[i] >> shift) & 0xFFu];
        }
        bool single = false;
        for (size_t b = 0; b < 256; ++b) {
            if (cnt[b] == n) {
                single = true;
                break;
            }
        }
        if (single) continue;
        uint32_t off[256];
        uint32_t sum = 0;
        for (size_t b = 0; b < 256; ++b) {
            off[b] = sum;
            sum += cnt[b];
        }
        for (size_t i = 0; i < n; ++i) {
            const uint8_t key = static_cast<uint8_t>((src[i] >> shift) & 0xFFu);
            dst[off[key]++] = src[i];
        }
        std::swap(src, dst);
    }
    if (src != arr) {
        std::memcpy(arr, src, n * sizeof(uint64_t));
    }
}

inline unsigned log2_pow2(size_t v) noexcept {
    unsigned r = 0;
    while ((size_t(1) << r) < v) ++r;
    return r;
}

inline size_t parse_ncpu() {
    if (const char* s = std::getenv("NCPU")) {
        int v = std::atoi(s);
        if (v >= 1) return static_cast<size_t>(v);
    }
    unsigned hc = std::thread::hardware_concurrency();
    return hc >= 1 ? static_cast<size_t>(hc) : 4;
}

inline std::string env_or(const char* name, const char* def_val) {
    if (const char* s = std::getenv(name); s && *s) return std::string(s);
    return std::string(def_val);
}

inline std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty() || a.back() == '/') return a + b;
    return a + "/" + b;
}

struct CampaignInfo {
    std::string id;
    std::string channel;
};

template <bool FastPack>
void run_pipeline(const std::string& data_dir, const std::string& output_dir, ThreadPool& pool,
                  int64_t event_win_start, int64_t event_win_end, int64_t click_win_start,
                  int64_t click_win_end, int32_t day_start, int32_t day_end,
                  const DateLut& date_lut, const std::vector<std::string>& cat_names,
                  uint32_t unknown_cat_id, FlatStrMap<uint32_t>& product_map,
                  const std::vector<CampaignInfo>& campaigns,
                  const std::vector<uint32_t>& fast_camp_by_num,
                  FlatStrMap<uint32_t>& campaign_map,
                  std::chrono::steady_clock::time_point t0) {
    using T = Traits<FastPack>;
    using RawEv = typename T::RawEvent;
    using SurvEv = typename T::SurvivingEvent;
    using SurvClk = typename T::SurvivingClick;
    using ShardEv = typename T::ShardEvent;
    using ShardClk = typename T::ShardClick;

    const size_t ncpu = pool.size();

    size_t total_ev_bytes = 0;
    auto ev_files = open_all_pread_files(join_path(data_dir, "events"), &total_ev_bytes);
    auto ev_chunks = build_pread_chunks(ev_files);

    const size_t est_events = total_ev_bytes / 70;
    size_t dedup_buckets = 64;
    if (est_events >= 35000000ULL) dedup_buckets = 2048;
    else if (est_events >= 12000000ULL) dedup_buckets = 1024;
    else if (est_events >= 2000000ULL) dedup_buckets = 512;
    else if (est_events >= 200000ULL) dedup_buckets = 128;

    size_t person_shards = 64;
    if (est_events >= 35000000ULL) person_shards = 512;
    else if (est_events >= 2000000ULL) person_shards = 256;

    const unsigned dedup_shift = 64u - log2_pow2(dedup_buckets);
    const unsigned person_shift = 64u - log2_pow2(person_shards);

    HugeBlockPool block_pool(T::kPoolBlockBytes);
    std::vector<ThreadBlockCache> caches(ncpu);
    for (size_t t = 0; t < ncpu; ++t) caches[t].init(&block_pool);

    std::vector<OverflowArena> event_arenas(ncpu);
    // Deduplication reads event user IDs while click workers append their own.
    std::vector<OverflowArena> user_arenas(ncpu * 2);
    std::vector<OverflowArena> click_arenas(ncpu);

    std::vector<std::vector<ChunkedVec<RawEv, T::kEvBlockCap>>> raw_ev_buckets(ncpu);
    for (size_t t = 0; t < ncpu; ++t) {
        raw_ev_buckets[t].resize(dedup_buckets);
        for (size_t b = 0; b < dedup_buckets; ++b) raw_ev_buckets[t][b].set_cache(&caches[t]);
    }

    // Phase 1: Parallel 32-shard LinkTable build across all cores
    LinkTable links;
    links.build(join_path(data_dir, "links"), pool, caches);
    const auto t_links = std::chrono::steady_clock::now();

    // Phase 2: Scan events/*.csv across all cores with 1 MB pread chunks
    if (!ev_chunks.empty()) {
        std::atomic<size_t> next_ev_chunk{0};
        pool.run([&](size_t tid) {
            std::vector<char> buf;
            buf.reserve((size_t(1) << 20) + 65536 + 64);
            auto& ea = event_arenas[tid];
            auto& ua = user_arenas[tid];
            auto& buckets = raw_ev_buckets[tid];
            const uint16_t aid = static_cast<uint16_t>(tid);

            for (;;) {
                const size_t ci = next_ev_chunk.fetch_add(1, std::memory_order_relaxed);
                if (ci >= ev_chunks.size()) break;
                size_t data_start = 0, limit = 0, blen = 0;
                read_chunk_into(ev_chunks[ci], buf, data_start, limit, blen);
                if (data_start >= limit) continue;

                for_each_csv_line<7>(
                    buf.data(), data_start, limit, blen, [&](const std::string_view (&f)[7]) {
                        if (f[0].empty()) return;

                        int64_t ingest_ms = 0;
                        if (!parse_ts_f1_fast(f[3], date_lut, ingest_ms)) return;

                        const CompactStr eid = CompactStr::make_str(f[0], aid, ea);
                        const uint64_t eh = fast_compact_hash(eid, event_arenas.data());
                        const size_t b = static_cast<size_t>(eh >> dedup_shift);

                        bool valid = !f[1].empty() && !f[5].empty();
                        int64_t ts_ms = 0;
                        if (valid) {
                            valid = parse_ts_any_fast(f[2], date_lut, ts_ms) &&
                                    ts_ms >= event_win_start && ts_ms < event_win_end &&
                                    (ingest_ms - ts_ms) <= kLateCutoffMs;
                        }

                        uint32_t price_cents = 0;
                        uint32_t cat_id = 0;
                        if (valid) {
                            const std::string_view et = f[4];
                            switch (et.size()) {
                                case 4:
                                    if (std::memcmp(et.data(), "view", 4) != 0 &&
                                        std::memcmp(et.data(), "cart", 4) != 0) {
                                        valid = false;
                                    }
                                    break;
                                case 8:
                                    if (std::memcmp(et.data(), "purchase", 8) == 0) {
                                        valid = parse_price_cents(f[6], price_cents);
                                        if (valid) {
                                            const uint32_t* pc = product_map.find(f[5]);
                                            cat_id = pc ? *pc : unknown_cat_id;
                                        }
                                    } else {
                                        valid = false;
                                    }
                                    break;
                                default:
                                    valid = false;
                                    break;
                            }
                        }

                        RawEv re{};
                        re.event_id = eid;
                        re.eid_hash32 = static_cast<uint32_t>(eh);
                        if constexpr (FastPack) {
                            int64_t sec = (ingest_ms - event_win_start) / 1000LL;
                            if (sec < std::numeric_limits<int32_t>::min())
                                sec = std::numeric_limits<int32_t>::min();
                            else if (sec > std::numeric_limits<int32_t>::max())
                                sec = std::numeric_limits<int32_t>::max();
                            re.ingest_rel_sec = static_cast<int32_t>(sec);
                            if (valid) {
                                re.user_id = CompactStr::make_str(f[1], aid, ua);
                                re.ts_rel = static_cast<uint32_t>(ts_ms - event_win_start);
                                re.price_cat = (cat_id << 24) | price_cents;
                            } else {
                                re.ts_rel = kFastTombstone;
                            }
                        } else {
                            re.eid_hash = eh;
                            re.ingest_ms = ingest_ms;
                            if (valid) {
                                re.user_id = CompactStr::make_str(f[1], aid, ua);
                                re.ts_ms = ts_ms;
                                re.price_cents = price_cents;
                                re.cat_id = cat_id;
                            } else {
                                re.ts_ms = kSlowTombstone;
                            }
                        }
                        buckets[b].push_back(re);
                    });
            }
        });
    }
    ev_chunks.clear();
    ev_files.clear();
    product_map.clear_and_free();
    const auto t_phase_a = std::chrono::steady_clock::now();

    // Phase B: Overlap zero-copy sequential dedup (128..1024 buckets) with clicks/*.csv scan
    std::vector<std::vector<ChunkedVec<SurvEv, T::kSurvEvBlockCap>>> ev_person_buckets(ncpu);
    std::vector<std::vector<ChunkedVec<SurvClk, T::kSurvClkBlockCap>>> clk_person_buckets(ncpu);
    for (size_t t = 0; t < ncpu; ++t) {
        ev_person_buckets[t].resize(person_shards);
        clk_person_buckets[t].resize(person_shards);
        for (size_t s = 0; s < person_shards; ++s) {
            ev_person_buckets[t][s].set_cache(&caches[t]);
            clk_person_buckets[t][s].set_cache(&caches[t]);
        }
    }

    auto clk_files = open_all_pread_files(join_path(data_dir, "clicks"));
    auto clk_chunks = build_pread_chunks(clk_files);
    const size_t clk_starter_threads = clk_chunks.empty() ? 0 : (ncpu > 4 ? 2 : 1);
    const bool has_links = !links.empty();

    {
        std::atomic<size_t> next_dedup_b{0};
        std::atomic<size_t> next_clk_chunk{0};

        pool.run([&](size_t tid) {
            const OverflowArena* ea_ptr = event_arenas.data();
            const OverflowArena* ua_ptr = user_arenas.data();
            auto& my_cache = caches[tid];
            auto& out_ev_buckets = ev_person_buckets[tid];
            auto& out_clk_buckets = clk_person_buckets[tid];
            auto& ua = user_arenas[ncpu + tid];
            auto& ca = click_arenas[tid];
            const uint16_t aid = static_cast<uint16_t>(tid);

            struct BlockRef {
                RawEv* ptr;
                uint32_t len;
            };
            std::vector<BlockRef> block_refs;
            std::vector<DedupIdx> idx_table;
            auto run_dedup = [&]() {
                for (;;) {
                    const size_t b = next_dedup_b.fetch_add(1, std::memory_order_relaxed);
                    if (b >= dedup_buckets) break;

                    block_refs.clear();
                    size_t total = 0;
                    for (size_t t = 0; t < ncpu; ++t) {
                        auto& cv = raw_ev_buckets[t][b];
                        const size_t nb = cv.num_blocks();
                        for (size_t bi = 0; bi < nb; ++bi) {
                            const uint32_t blen = static_cast<uint32_t>(cv.block_len(bi));
                            if (blen > 0) {
                                block_refs.push_back(BlockRef{cv.block_ptr(bi), blen});
                                total += blen;
                            }
                        }
                    }
                    if (total == 0) continue;

                    size_t cap = 64;
                    while (cap * 11 < total * 20) cap <<= 1;
                    idx_table.assign(cap, DedupIdx{});
                    const size_t mask = cap - 1;

                    const uint32_t nrefs = static_cast<uint32_t>(block_refs.size());
                    for (uint32_t bi = 0; bi < nrefs; ++bi) {
                        RawEv* const ptr = block_refs[bi].ptr;
                        const uint32_t blen = block_refs[bi].len;
                        for (uint32_t i = 0; i < blen; ++i) {
                            if (i + 16 < blen) {
                                __builtin_prefetch(&idx_table[ptr[i + 16].eid_hash32 & mask], 1, 1);
                            }
                            RawEv& re = ptr[i];
                            const uint32_t h32 = re.eid_hash32;
                            const uint32_t code = (bi << 10) | i;
                            size_t slot = static_cast<size_t>(h32) & mask;
                            for (;;) {
                                DedupIdx& sl = idx_table[slot];
                                if (sl.idx_plus1 == 0) {
                                    sl.idx_plus1 = code + 1;
                                    sl.hash32 = h32;
                                    break;
                                }
                                if (sl.hash32 == h32) {
                                    const uint32_t bcode = sl.idx_plus1 - 1;
                                    RawEv& best = block_refs[bcode >> 10].ptr[bcode & 1023u];
                                    if (CompactStr::raw_equals(best.event_id, re.event_id, ea_ptr)) {
                                        if (re.older_than(best)) {
                                            best.mark_tombstone();
                                            sl.idx_plus1 = code + 1;
                                        } else {
                                            re.mark_tombstone();
                                        }
                                        break;
                                    }
                                }
                                slot = (slot + 1) & mask;
                            }
                        }
                    }

                    // Keep the hashed survivors in L1 while their link lookups are in flight.
                    constexpr size_t batch_size = 32;
                    for (const BlockRef& block : block_refs) {
                        for (size_t start = 0; start < block.len; start += batch_size) {
                            const RawEv* survivors[batch_size];
                            uint64_t hashes[batch_size];
                            size_t count = 0;
                            const size_t end = std::min(start + batch_size, size_t(block.len));
                            for (size_t i = start; i < end; ++i) {
                                const RawEv& re = block.ptr[i];
                                if (re.is_tombstone()) continue;
                                const uint64_t h = fast_compact_hash(re.user_id, ua_ptr);
                                if (has_links) links.prefetch(h);
                                survivors[count] = &re;
                                hashes[count++] = h;
                            }
                            for (size_t i = 0; i < count; ++i) {
                                const RawEv& re = *survivors[i];
                                const uint64_t uh = hashes[i];
                                CompactStr uid = re.user_id;
                                uint64_t ph = uh;
                                if (has_links) {
                                    uint32_t canon_id = 0;
                                    if (links.resolve_with_hash(re.user_id, uh, canon_id, ua_ptr)) {
                                        uid = CompactStr::make_linked(canon_id);
                                        ph = links.canonical_hash(canon_id);
                                    }
                                }
                                if constexpr (FastPack) {
                                    uint32_t ph32 = static_cast<uint32_t>(ph);
                                    if (ph32 == 0) ph32 = 1;
                                    out_ev_buckets[static_cast<size_t>(ph >> person_shift)].push_back(
                                        SurvEv{uid, re.ts_rel, re.price_cat, ph32});
                                } else {
                                    out_ev_buckets[static_cast<size_t>(ph >> person_shift)].push_back(
                                        SurvEv{uid, re.price_cents, re.ts_ms, ph, re.cat_id, 0});
                                }
                            }
                        }
                    }

                    for (size_t t = 0; t < ncpu; ++t) {
                        raw_ev_buckets[t][b].free_to(&my_cache);
                    }
                }
            };

            auto run_clicks = [&]() {
                if (clk_chunks.empty()) return;
                std::vector<char> buf;
                buf.reserve((size_t(1) << 20) + 65536 + 64);

                struct PendingClk {
                    int64_t ts_ms;
                    uint64_t uh;
                    std::string_view kid;
                    std::string_view uid;
                    uint32_t camp_id;
                };
                PendingClk batch[16];
                size_t bsz = 0;

                auto flush_batch = [&]() {
                    for (size_t i = 0; i < bsz; ++i) {
                        const PendingClk& pc = batch[i];
                        CompactStr uid{};
                        uint64_t ph = pc.uh;
                        uint32_t canon_id = 0;
                        if (has_links && links.resolve_sv_with_hash(pc.uid, pc.uh, canon_id)) {
                            uid = CompactStr::make_linked(canon_id);
                            ph = links.canonical_hash(canon_id);
                        } else {
                            uid = CompactStr::make_str(pc.uid, static_cast<uint16_t>(ncpu + tid), ua);
                        }
                        const CompactStr kid = CompactStr::make_str(pc.kid, aid, ca);
                        if constexpr (FastPack) {
                            uint32_t ph32 = static_cast<uint32_t>(ph);
                            if (ph32 == 0) ph32 = 1;
                            out_clk_buckets[static_cast<size_t>(ph >> person_shift)].push_back(
                                SurvClk{uid, kid, static_cast<uint32_t>(pc.ts_ms - click_win_start),
                                        pc.camp_id, ph32});
                        } else {
                            out_clk_buckets[static_cast<size_t>(ph >> person_shift)].push_back(
                                SurvClk{pc.ts_ms, ph, uid, kid, pc.camp_id, 0});
                        }
                    }
                    bsz = 0;
                };

                auto lookup_camp = [&](std::string_view sv) -> uint32_t {
                    if (sv.size() >= 3 && sv[0] == 'c' && sv[1] == '_' &&
                        static_cast<uint8_t>(sv[2] - '1') <= 8) {
                        uint64_t num = static_cast<uint64_t>(sv[2] - '0');
                        bool is_canon = true;
                        for (size_t i = 3; i < sv.size(); ++i) {
                            const uint8_t d = static_cast<uint8_t>(sv[i] - '0');
                            if (d > 9 || num > 1000000ULL) {
                                is_canon = false;
                                break;
                            }
                            num = num * 10 + d;
                        }
                        if (is_canon) {
                            return (num < fast_camp_by_num.size())
                                       ? fast_camp_by_num[num]
                                       : std::numeric_limits<uint32_t>::max();
                        }
                    }
                    const uint32_t* p = campaign_map.find(sv);
                    return p ? *p : std::numeric_limits<uint32_t>::max();
                };

                for (;;) {
                    const size_t ci = next_clk_chunk.fetch_add(1, std::memory_order_relaxed);
                    if (ci >= clk_chunks.size()) break;
                    size_t data_start = 0, limit = 0, blen = 0;
                    read_chunk_into(clk_chunks[ci], buf, data_start, limit, blen);
                    if (data_start >= limit) continue;

                    for_each_csv_line<4>(
                        buf.data(), data_start, limit, blen, [&](const std::string_view (&f)[4]) {
                            if (f[0].empty() || f[1].empty()) return;
                            const uint32_t cid = lookup_camp(f[2]);
                            if (cid == std::numeric_limits<uint32_t>::max()) return;

                            int64_t ts_ms = 0;
                            if (!parse_ts_any_fast(f[3], date_lut, ts_ms) ||
                                ts_ms < click_win_start || ts_ms >= click_win_end) {
                                return;
                            }

                            const uint64_t uh = fast_str_hash(f[1]);
                            if (has_links) links.prefetch(uh);
                            batch[bsz++] = PendingClk{ts_ms, uh, f[0], f[1], cid};
                            if (bsz == 16) flush_batch();
                        });
                    flush_batch();
                }
            };

            if (tid < clk_starter_threads) {
                run_clicks();
                run_dedup();
            } else {
                run_dedup();
                run_clicks();
            }
        });
    }
    clk_chunks.clear();
    clk_files.clear();
    campaign_map.clear_and_free();
    for (auto& ea : event_arenas) ea.free_all();
    const auto t_phase_b = std::chrono::steady_clock::now();

    // Phase C: Parallel processing of person shards (LPT descending) with separate L2 int32 assists
    std::vector<size_t> shard_order(person_shards);
    std::iota(shard_order.begin(), shard_order.end(), 0);
    std::vector<size_t> shard_load(person_shards, 0);
    for (size_t s = 0; s < person_shards; ++s) {
        size_t load = 0;
        for (size_t t = 0; t < ncpu; ++t) {
            load += ev_person_buckets[t][s].size() + clk_person_buckets[t][s].size();
        }
        shard_load[s] = load;
    }
    std::sort(shard_order.begin(), shard_order.end(),
              [&](size_t a, size_t b) { return shard_load[a] > shard_load[b]; });

    const size_t num_cats = cat_names.size();
    const size_t num_camps = campaigns.size();
    const int64_t num_days_i64 =
        static_cast<int64_t>(day_end) - static_cast<int64_t>(day_start) + 1;
    const bool use_flat_camp =
        (num_days_i64 > 0) && (num_camps <= (2000000ULL / static_cast<size_t>(num_days_i64)));
    const size_t flat_camp_size =
        use_flat_camp ? (num_camps * static_cast<size_t>(num_days_i64)) : 0;

    struct ThreadAgg {
        std::vector<CampCreditAgg> flat_credits;
        std::vector<int32_t> flat_assists;
        CampDayHashMap map;
        int64_t total_sessions = 0;
        int64_t total_revenue = 0;
    };
    std::vector<ThreadAgg> thread_aggs(ncpu);
    for (size_t t = 0; t < ncpu; ++t) {
        if (use_flat_camp) {
            thread_aggs[t].flat_credits.resize(flat_camp_size);
            thread_aggs[t].flat_assists.resize(flat_camp_size, 0);
        }
    }

    {
        std::atomic<size_t> next_order_idx{0};
        pool.run([&](size_t tid) {
            const OverflowArena* ua_ptr = user_arenas.data();
            const OverflowArena* ca_ptr = click_arenas.data();
            auto& my_cache = caches[tid];
            auto& agg = thread_aggs[tid];

            const std::string out_path =
                join_path(join_path(output_dir, "user_daily"),
                          "part-" + std::to_string(tid) + ".parquet");
            UserDailyParquetWriter writer(out_path);

            auto add_credit = [&](uint32_t camp_id, int32_t day, uint8_t has_purch, uint64_t rev) {
                if (use_flat_camp) {
                    CampCreditAgg& cd =
                        agg.flat_credits[static_cast<size_t>(camp_id) *
                                             static_cast<size_t>(num_days_i64) +
                                         static_cast<size_t>(day - day_start)];
                    ++cd.sessions;
                    cd.purchasing_sessions += has_purch;
                    cd.revenue_cents += static_cast<int64_t>(rev);
                } else {
                    CampDayFull& cd = agg.map.get(camp_id, day);
                    ++cd.sessions;
                    cd.purchasing_sessions += has_purch;
                    cd.revenue_cents += static_cast<int64_t>(rev);
                }
            };
            auto add_assist = [&](uint32_t camp_id, int32_t day) {
                if (use_flat_camp) {
                    ++agg.flat_assists[static_cast<size_t>(camp_id) *
                                           static_cast<size_t>(num_days_i64) +
                                       static_cast<size_t>(day - day_start)];
                } else {
                    ++agg.map.get(camp_id, day).assisted_sessions;
                }
            };

            std::vector<PersonMapSlot> pmap;
            std::vector<CompactStr> pid_user;
            std::vector<uint32_t> ev_cnt, clk_cnt, ev_off, clk_off, ev_cur, clk_cur;
            std::vector<uint32_t> tmp_ev_pids, tmp_clk_pids;
            std::vector<ShardEv> tmp_evs, shard_events;
            std::vector<ShardClk> tmp_clks, shard_clicks;
            std::vector<uint64_t> radix_scratch;
            std::vector<SessionInfo> sessions;
            std::vector<uint64_t> cat_rev(num_cats, 0);
            std::vector<uint32_t> touched_cats;
            std::vector<uint32_t> camp_count(num_camps, 0);
            std::vector<uint32_t> active_camps;
            std::vector<uint32_t> active_pos(num_camps, std::numeric_limits<uint32_t>::max());

            for (;;) {
                const size_t oi = next_order_idx.fetch_add(1, std::memory_order_relaxed);
                if (oi >= person_shards) break;
                const size_t s = shard_order[oi];

                size_t total_ev = 0;
                for (size_t t = 0; t < ncpu; ++t) total_ev += ev_person_buckets[t][s].size();
                if (total_ev == 0) {
                    for (size_t t = 0; t < ncpu; ++t) clk_person_buckets[t][s].free_to(&my_cache);
                    continue;
                }

                size_t pcap = 64;
                // People recur within a shard. Start with a modest estimate and
                // grow at the usual load factor if the users are mostly distinct.
                while (pcap < total_ev / 8) pcap <<= 1;
                if (pcap < 128) pcap = 128;
                pmap.assign(pcap, PersonMapSlot{});
                size_t pmask = pcap - 1;
                pid_user.clear();
                ev_cnt.clear();
                tmp_ev_pids.clear();
                tmp_ev_pids.reserve(total_ev);
                tmp_evs.clear();
                tmp_evs.reserve(total_ev);

                auto rehash_pmap = [&]() {
                    const size_t ncap = pmap.size() * 2;
                    std::vector<PersonMapSlot> nxt(ncap);
                    const size_t nmask = ncap - 1;
                    for (const auto& sl : pmap) {
                        if (sl.hash32 == 0) continue;
                        size_t idx = static_cast<size_t>(sl.hash32) & nmask;
                        while (nxt[idx].hash32 != 0) idx = (idx + 1) & nmask;
                        nxt[idx] = sl;
                    }
                    pmap.swap(nxt);
                    pmask = nmask;
                };

                auto find_pid = [&](uint32_t h32, const CompactStr& uid) -> uint32_t {
                    size_t idx = static_cast<size_t>(h32) & pmask;
                    for (;;) {
                        const PersonMapSlot& sl = pmap[idx];
                        if (sl.hash32 == 0) return std::numeric_limits<uint32_t>::max();
                        if (sl.hash32 == h32 && CompactStr::raw_equals(sl.user_id, uid, ua_ptr)) {
                            return sl.local_pid;
                        }
                        idx = (idx + 1) & pmask;
                    }
                };

                for (size_t t = 0; t < ncpu; ++t) {
                    ev_person_buckets[t][s].for_each_block([&](const SurvEv* ptr, size_t n) {
                        for (size_t i = 0; i < n; ++i) {
                            const SurvEv& se = ptr[i];
                            uint32_t h32;
                            if constexpr (FastPack) {
                                h32 = se.person_hash32;
                            } else {
                                h32 = static_cast<uint32_t>(se.person_hash);
                                if (h32 == 0) h32 = 1;
                            }
                            if ((pid_user.size() + 1) * 20 >= pmap.size() * 11) rehash_pmap();
                            size_t idx = static_cast<size_t>(h32) & pmask;
                            uint32_t pid = 0;
                            for (;;) {
                                PersonMapSlot& sl = pmap[idx];
                                if (sl.hash32 == 0) {
                                    pid = static_cast<uint32_t>(pid_user.size());
                                    sl.hash32 = h32;
                                    sl.local_pid = pid;
                                    sl.user_id = se.user_id;
                                    pid_user.push_back(se.user_id);
                                    ev_cnt.push_back(1);
                                    break;
                                }
                                if (sl.hash32 == h32 &&
                                    CompactStr::raw_equals(sl.user_id, se.user_id, ua_ptr)) {
                                    pid = sl.local_pid;
                                    ++ev_cnt[pid];
                                    break;
                                }
                                idx = (idx + 1) & pmask;
                            }
                            tmp_ev_pids.push_back(pid);
                            if constexpr (FastPack) {
                                tmp_evs.push_back(ShardEv{
                                    (static_cast<uint64_t>(se.ts_rel) << 32) | se.price_cat});
                            } else {
                                tmp_evs.push_back(ShardEv{se.ts_ms, se.price_cents, se.cat_id});
                            }
                        }
                    });
                    ev_person_buckets[t][s].free_to(&my_cache);
                }

                const size_t num_persons = pid_user.size();
                ev_off.resize(num_persons + 1);
                ev_off[0] = 0;
                for (size_t p = 0; p < num_persons; ++p) ev_off[p + 1] = ev_off[p] + ev_cnt[p];
                ev_cur.assign(ev_off.begin(), ev_off.begin() + num_persons);
                shard_events.resize(total_ev);
                for (size_t i = 0; i < total_ev; ++i) {
                    shard_events[ev_cur[tmp_ev_pids[i]]++] = tmp_evs[i];
                }

                clk_cnt.assign(num_persons, 0);
                tmp_clk_pids.clear();
                tmp_clks.clear();
                for (size_t t = 0; t < ncpu; ++t) {
                    clk_person_buckets[t][s].for_each_block([&](const SurvClk* ptr, size_t n) {
                        for (size_t i = 0; i < n; ++i) {
                            const SurvClk& sc = ptr[i];
                            uint32_t h32;
                            if constexpr (FastPack) {
                                h32 = sc.person_hash32;
                            } else {
                                h32 = static_cast<uint32_t>(sc.person_hash);
                                if (h32 == 0) h32 = 1;
                            }
                            const uint32_t pid = find_pid(h32, sc.user_id);
                            if (pid != std::numeric_limits<uint32_t>::max()) {
                                ++clk_cnt[pid];
                                tmp_clk_pids.push_back(pid);
                                if constexpr (FastPack) {
                                    tmp_clks.push_back(ShardClk{sc.ts_rel, sc.camp_id, sc.click_id});
                                } else {
                                    tmp_clks.push_back(ShardClk{sc.ts_ms, sc.camp_id, sc.click_id});
                                }
                            }
                        }
                    });
                    clk_person_buckets[t][s].free_to(&my_cache);
                }

                const size_t total_clk = tmp_clks.size();
                clk_off.resize(num_persons + 1);
                clk_off[0] = 0;
                for (size_t p = 0; p < num_persons; ++p) clk_off[p + 1] = clk_off[p] + clk_cnt[p];
                clk_cur.assign(clk_off.begin(), clk_off.begin() + num_persons);
                shard_clicks.resize(total_clk);
                for (size_t i = 0; i < total_clk; ++i) {
                    shard_clicks[clk_cur[tmp_clk_pids[i]]++] = tmp_clks[i];
                }

                for (size_t p = 0; p < num_persons; ++p) {
                    ShardEv* pev = shard_events.data() + ev_off[p];
                    const size_t nev = ev_off[p + 1] - ev_off[p];
                    if constexpr (FastPack) {
                        if (nev > 2048) {
                            radix_sort_events_u64(reinterpret_cast<uint64_t*>(pev), nev,
                                                  radix_scratch);
                        } else {
                            std::sort(pev, pev + nev, [](const ShardEv& a, const ShardEv& b) {
                                return (a.packed >> 32) < (b.packed >> 32);
                            });
                        }
                    } else {
                        std::sort(pev, pev + nev, [](const ShardEv& a, const ShardEv& b) {
                            return a.ts_ms < b.ts_ms;
                        });
                    }

                    const CompactStr& uid = pid_user[p];
                    const std::string_view user_sv =
                        uid.is_linked() ? links.node_sv(uid.linked_id()) : uid.raw_view(ua_ptr);

                    auto get_ev = [&](size_t idx, int64_t& ts_key, uint32_t& price,
                                      uint32_t& cat) {
                        if constexpr (FastPack) {
                            const uint64_t pk = pev[idx].packed;
                            ts_key = static_cast<int64_t>(pk >> 32);
                            const uint32_t pc = static_cast<uint32_t>(pk);
                            price = pc & 0xFFFFFFu;
                            cat = pc >> 24;
                        } else {
                            ts_key = pev[idx].ts_ms;
                            price = pev[idx].price_cents;
                            cat = pev[idx].cat_id;
                        }
                    };

                    auto day_of_ev_key = [&](int64_t ts_key) -> int32_t {
                        if constexpr (FastPack) {
                            return floor_div_day(event_win_start + ts_key);
                        } else {
                            return floor_div_day(ts_key);
                        }
                    };

                    sessions.clear();
                    int64_t first_key = 0;
                    uint32_t dummy_p = 0, dummy_c = 0;
                    get_ev(0, first_key, dummy_p, dummy_c);

                    int64_t sess_start = first_key;
                    int64_t prev_ts = first_key;
                    int32_t sess_day = day_of_ev_key(sess_start);
                    uint64_t sess_rev = 0;
                    uint8_t sess_purch = 0;

                    int32_t cur_day = sess_day;
                    int32_t day_sessions = 1;
                    int32_t day_events = 0;
                    uint64_t day_rev = 0;
                    touched_cats.clear();

                    auto flush_day = [&]() {
                        uint32_t best_cat = std::numeric_limits<uint32_t>::max();
                        uint64_t best_rev = 0;
                        for (uint32_t c : touched_cats) {
                            const uint64_t r = cat_rev[c];
                            cat_rev[c] = 0;
                            if (r > best_rev || (r == best_rev && c < best_cat)) {
                                best_rev = r;
                                best_cat = c;
                            }
                        }
                        touched_cats.clear();
                        const bool has_top = (day_rev > 0);
                        writer.add_row(user_sv, cur_day, day_sessions, day_events,
                                       static_cast<int64_t>(day_rev), has_top,
                                       has_top ? std::string_view(cat_names[best_cat])
                                               : std::string_view{});
                        agg.total_sessions += day_sessions;
                        agg.total_revenue += static_cast<int64_t>(day_rev);
                    };

                    for (size_t i = 0; i < nev; ++i) {
                        int64_t ts_key = 0;
                        uint32_t price = 0, cat = 0;
                        get_ev(i, ts_key, price, cat);

                        if (i > 0 && ts_key - prev_ts > kSessionGapMs) {
                            sessions.push_back(SessionInfo{sess_start, sess_rev, sess_day, sess_purch});
                            sess_start = ts_key;
                            sess_day = day_of_ev_key(sess_start);
                            sess_rev = 0;
                            sess_purch = 0;
                            if (sess_day != cur_day) {
                                flush_day();
                                cur_day = sess_day;
                                day_sessions = 0;
                                day_events = 0;
                                day_rev = 0;
                            }
                            ++day_sessions;
                        }
                        prev_ts = ts_key;
                        ++day_events;
                        if (price > 0) {
                            sess_rev += price;
                            sess_purch = 1;
                            day_rev += price;
                            if (cat_rev[cat] == 0) touched_cats.push_back(cat);
                            cat_rev[cat] += price;
                        }
                    }
                    sessions.push_back(SessionInfo{sess_start, sess_rev, sess_day, sess_purch});
                    flush_day();

                    const size_t nclk = clk_off[p + 1] - clk_off[p];
                    if (nclk == 0) {
                        for (const auto& sess : sessions) {
                            add_credit(0, sess.day, sess.has_purch, sess.rev);
                        }
                    } else {
                        ShardClk* pclk = shard_clicks.data() + clk_off[p];
                        std::sort(pclk, pclk + nclk, [&](const ShardClk& a, const ShardClk& b) {
                            if constexpr (FastPack) {
                                if (a.ts_rel != b.ts_rel) return a.ts_rel < b.ts_rel;
                            } else {
                                if (a.ts_ms != b.ts_ms) return a.ts_ms < b.ts_ms;
                            }
                            return byte_less(b.click_id.raw_view(ca_ptr),
                                             a.click_id.raw_view(ca_ptr));
                        });

                        size_t l = 0, r = 0;
                        for (const auto& sess : sessions) {
                            int64_t win_lo, win_hi;
                            if constexpr (FastPack) {
                                win_lo = sess.start_key;
                                win_hi = sess.start_key + kClickReachMs;
                            } else {
                                win_lo = sess.start_key - kClickReachMs;
                                win_hi = sess.start_key;
                            }

                            auto clk_ts = [&](size_t idx) -> int64_t {
                                if constexpr (FastPack) {
                                    return static_cast<int64_t>(pclk[idx].ts_rel);
                                } else {
                                    return pclk[idx].ts_ms;
                                }
                            };

                            while (r < nclk && clk_ts(r) <= win_hi) {
                                const uint32_t c = pclk[r].camp_id;
                                if (camp_count[c]++ == 0) {
                                    active_pos[c] = static_cast<uint32_t>(active_camps.size());
                                    active_camps.push_back(c);
                                }
                                ++r;
                            }
                            while (l < r && clk_ts(l) < win_lo) {
                                const uint32_t c = pclk[l].camp_id;
                                if (--camp_count[c] == 0) {
                                    const uint32_t pos = active_pos[c];
                                    const uint32_t last = active_camps.back();
                                    active_camps[pos] = last;
                                    active_pos[last] = pos;
                                    active_camps.pop_back();
                                    active_pos[c] = std::numeric_limits<uint32_t>::max();
                                }
                                ++l;
                            }
                            const uint32_t credit_camp = (l < r) ? pclk[r - 1].camp_id : 0u;
                            add_credit(credit_camp, sess.day, sess.has_purch, sess.rev);
                            for (uint32_t ac : active_camps) {
                                add_assist(ac, sess.day);
                            }
                        }
                        for (uint32_t ac : active_camps) {
                            camp_count[ac] = 0;
                            active_pos[ac] = std::numeric_limits<uint32_t>::max();
                        }
                        active_camps.clear();
                    }
                }
            }

            writer.finish();
        });
    }
    const auto t_phase_c = std::chrono::steady_clock::now();

    // Phase D: Reduce campaign_daily across threads & write campaign_daily/part-0.parquet
    CampaignDailyParquetWriter camp_writer(
        join_path(join_path(output_dir, "campaign_daily"), "part-0.parquet"));
    int64_t camp_sessions_sum = 0;
    int64_t camp_revenue_sum = 0;
    int64_t camp_assists_sum = 0;
    int64_t non_organic_sessions_sum = 0;

    if (use_flat_camp) {
        for (size_t c = 0; c < num_camps; ++c) {
            const std::string_view cid = campaigns[c].id;
            const std::string_view ch = campaigns[c].channel;
            const size_t base = c * static_cast<size_t>(num_days_i64);
            for (int64_t d = 0; d < num_days_i64; ++d) {
                int32_t sess = 0, purch = 0, assist = 0;
                int64_t rev = 0;
                const size_t idx = base + static_cast<size_t>(d);
                for (size_t t = 0; t < ncpu; ++t) {
                    const CampCreditAgg& cr = thread_aggs[t].flat_credits[idx];
                    sess += cr.sessions;
                    purch += cr.purchasing_sessions;
                    rev += cr.revenue_cents;
                    assist += thread_aggs[t].flat_assists[idx];
                }
                if (sess > 0 || assist > 0) {
                    camp_writer.add_row(cid, ch, static_cast<int32_t>(day_start + d), sess, purch,
                                        rev, assist);
                    camp_sessions_sum += sess;
                    camp_revenue_sum += rev;
                    camp_assists_sum += assist;
                    if (c != 0) non_organic_sessions_sum += sess;
                }
            }
        }
    } else {
        CampDayHashMap merged;
        for (size_t t = 0; t < ncpu; ++t) {
            for (const auto& s : thread_aggs[t].map.table) {
                if (s.key == std::numeric_limits<uint64_t>::max()) continue;
                const uint32_t c = static_cast<uint32_t>(s.key >> 32);
                const int32_t d = static_cast<int32_t>(static_cast<uint32_t>(s.key));
                CampDayFull& m = merged.get(c, d);
                m.sessions += s.agg.sessions;
                m.purchasing_sessions += s.agg.purchasing_sessions;
                m.revenue_cents += s.agg.revenue_cents;
                m.assisted_sessions += s.agg.assisted_sessions;
            }
        }
        for (const auto& s : merged.table) {
            if (s.key == std::numeric_limits<uint64_t>::max()) continue;
            if (s.agg.sessions > 0 || s.agg.assisted_sessions > 0) {
                const uint32_t c = static_cast<uint32_t>(s.key >> 32);
                const int32_t d = static_cast<int32_t>(static_cast<uint32_t>(s.key));
                camp_writer.add_row(campaigns[c].id, campaigns[c].channel, d, s.agg.sessions,
                                    s.agg.purchasing_sessions, s.agg.revenue_cents,
                                    s.agg.assisted_sessions);
                camp_sessions_sum += s.agg.sessions;
                camp_revenue_sum += s.agg.revenue_cents;
                camp_assists_sum += s.agg.assisted_sessions;
                if (c != 0) non_organic_sessions_sum += s.agg.sessions;
            }
        }
    }
    camp_writer.finish();

    int64_t user_sessions_sum = 0, user_revenue_sum = 0;
    for (const auto& ta : thread_aggs) {
        user_sessions_sum += ta.total_sessions;
        user_revenue_sum += ta.total_revenue;
    }

    const auto t_end = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::fprintf(stderr,
                 "done in %.1f ms (links %.1f, events %.1f, dedup+clk %.1f, shards %.1f, "
                 "camp %.1f) | sessions=%lld/%lld rev=%lld/%lld assists=%lld>=%lld\n",
                 ms(t0, t_end), ms(t0, t_links), ms(t_links, t_phase_a),
                 ms(t_phase_a, t_phase_b), ms(t_phase_b, t_phase_c), ms(t_phase_c, t_end),
                 static_cast<long long>(user_sessions_sum),
                 static_cast<long long>(camp_sessions_sum),
                 static_cast<long long>(user_revenue_sum),
                 static_cast<long long>(camp_revenue_sum),
                 static_cast<long long>(camp_assists_sum),
                 static_cast<long long>(non_organic_sessions_sum));
    std::fflush(nullptr);
    if (!std::getenv("SS_PGO_RUN")) {
        ::_exit(0);
    }
    return;
}

}  // namespace

int main(int argc, char** argv) {
    const auto t0 = std::chrono::steady_clock::now();

    std::string data_dir = env_or("DATA_DIR", "/data");
    std::string output_dir = env_or("OUTPUT_DIR", "/output");
    if (argc >= 3) {
        data_dir = argv[1];
        output_dir = argv[2];
    }
    const size_t ncpu = parse_ncpu();
    ThreadPool pool(ncpu);

    std::filesystem::create_directories(join_path(output_dir, "user_daily"));
    std::filesystem::create_directories(join_path(output_dir, "campaign_daily"));

    int64_t event_win_start = 0, event_win_end = 0;
    parse_manifest(join_path(data_dir, "manifest.json"), event_win_start, event_win_end);
    const int64_t click_win_start = event_win_start - kClickReachMs;
    const int64_t click_win_end = event_win_end;
    const int32_t day_start = floor_div_day(event_win_start);
    const int32_t day_end = floor_div_day(event_win_end - 1);

    DateLut date_lut;
    date_lut.build(click_win_start, event_win_end);

    std::vector<std::string> cat_names;
    FlatStrMap<uint32_t> product_map;
    uint32_t unknown_cat_id = 0;
    {
        auto pf = read_whole_file(join_path(data_dir, "products.csv"));
        struct ProdTmp {
            std::string_view pid;
            std::string_view cat;
        };
        std::vector<ProdTmp> rows;
        FlatStrMap<uint32_t> tmp_cats;
        tmp_cats.emplace("unknown", 0);
        cat_names.emplace_back("unknown");

        if (!pf.empty()) {
            const char* nl = static_cast<const char*>(std::memchr(pf.data(), '\n', pf.size()));
            if (nl) {
                const char* p = nl + 1;
                const char* end = pf.data() + pf.size();
                std::string_view f[2];
                while (p < end) {
                    const char* lnl = static_cast<const char*>(
                        std::memchr(p, '\n', static_cast<size_t>(end - p)));
                    const char* lend = lnl ? lnl : end;
                    if (split_csv_line<2>(p, static_cast<size_t>(lend - p), f) && !f[0].empty()) {
                        rows.push_back(ProdTmp{f[0], f[1]});
                        const uint32_t next_cid = static_cast<uint32_t>(cat_names.size());
                        if (tmp_cats.emplace(f[1], next_cid).second) {
                            cat_names.emplace_back(f[1]);
                        }
                    }
                    p = lnl ? (lnl + 1) : end;
                }
            }
        }
        std::sort(cat_names.begin(), cat_names.end(),
                  [](const std::string& a, const std::string& b) { return byte_less(a, b); });
        FlatStrMap<uint32_t> sorted_cat_map;
        sorted_cat_map.reserve(cat_names.size());
        for (uint32_t i = 0; i < cat_names.size(); ++i) {
            sorted_cat_map.emplace(cat_names[i], i);
            if (cat_names[i] == "unknown") unknown_cat_id = i;
        }
        product_map.reserve(rows.size());
        for (const auto& r : rows) {
            const uint32_t* cid = sorted_cat_map.find(r.cat);
            product_map.emplace(r.pid, cid ? *cid : unknown_cat_id);
        }
    }

    std::vector<CampaignInfo> campaigns;
    campaigns.push_back({"organic", "organic"});
    FlatStrMap<uint32_t> campaign_map;
    std::vector<uint32_t> fast_camp_by_num;
    {
        auto cf = read_whole_file(join_path(data_dir, "campaigns.csv"));
        if (!cf.empty()) {
            const char* nl = static_cast<const char*>(std::memchr(cf.data(), '\n', cf.size()));
            if (nl) {
                const char* p = nl + 1;
                const char* end = cf.data() + cf.size();
                std::string_view f[2];
                while (p < end) {
                    const char* lnl = static_cast<const char*>(
                        std::memchr(p, '\n', static_cast<size_t>(end - p)));
                    const char* lend = lnl ? lnl : end;
                    if (split_csv_line<2>(p, static_cast<size_t>(lend - p), f) && !f[0].empty()) {
                        const uint32_t cid = static_cast<uint32_t>(campaigns.size());
                        if (campaign_map.emplace(f[0], cid).second) {
                            campaigns.push_back({std::string(f[0]), std::string(f[1])});
                            if (f[0].size() >= 3 && f[0][0] == 'c' && f[0][1] == '_' &&
                                static_cast<uint8_t>(f[0][2] - '1') <= 8) {
                                uint64_t num = static_cast<uint64_t>(f[0][2] - '0');
                                bool is_canon = true;
                                for (size_t i = 3; i < f[0].size(); ++i) {
                                    const uint8_t d = static_cast<uint8_t>(f[0][i] - '0');
                                    if (d > 9 || num > 1000000ULL) {
                                        is_canon = false;
                                        break;
                                    }
                                    num = num * 10 + d;
                                }
                                if (is_canon) {
                                    if (num >= fast_camp_by_num.size()) {
                                        fast_camp_by_num.resize(
                                            num + 1, std::numeric_limits<uint32_t>::max());
                                    }
                                    fast_camp_by_num[num] = cid;
                                }
                            }
                        }
                    }
                    p = lnl ? (lnl + 1) : end;
                }
            }
        }
    }

    const bool can_fast_pack =
        (event_win_end > event_win_start) &&
        (static_cast<uint64_t>(event_win_end - event_win_start) <= 0xFFFFFFFEULL) &&
        (static_cast<uint64_t>(click_win_end - click_win_start) <= 0xFFFFFFFEULL) &&
        (cat_names.size() <= 255);

    if (can_fast_pack) {
        run_pipeline<true>(data_dir, output_dir, pool, event_win_start, event_win_end,
                           click_win_start, click_win_end, day_start, day_end, date_lut,
                           cat_names, unknown_cat_id, product_map, campaigns, fast_camp_by_num,
                           campaign_map, t0);
    } else {
        run_pipeline<false>(data_dir, output_dir, pool, event_win_start, event_win_end,
                            click_win_start, click_win_end, day_start, day_end, date_lut,
                            cat_names, unknown_cat_id, product_map, campaigns, fast_camp_by_num,
                            campaign_map, t0);
    }

    return 0;
}
