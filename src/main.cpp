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
#include "sort_u64.hpp"
#include "thread_pool.hpp"
#include "union_find.hpp"

namespace {

constexpr int64_t kLateCutoffMs = 86400000LL;
constexpr int64_t kSessionGapMs = 1800000LL;
constexpr int64_t kClickReachMs = 7LL * 86400000LL;
constexpr uint32_t kFastTombstone = std::numeric_limits<uint32_t>::max();
constexpr int64_t kSlowTombstone = std::numeric_limits<int64_t>::min();

using EventId64 = FastId64<'e'>;
using UserId64 = FastId64<'u'>;
using ClickId64 = FastId64<'k'>;

template <bool FastPack>
struct Traits;

template <>
struct Traits<true> {
    static constexpr size_t kEvBlockCap = 2048;
    static constexpr unsigned kEvBlockShift = 11;
    static constexpr uint32_t kEvBlockMask = 2047u;
    static constexpr size_t kSurvEvBlockCap = 4096;
    static constexpr size_t kSurvClkBlockCap = 2048;

    struct alignas(32) RawEvent {
        EventId64 event_id;
        UserId64 user_id;
        uint32_t ts_rel;        // kFastTombstone == invalid/late/duplicate-loser
        int32_t ingest_rel_sec; // (ingest_ms - event_win_start) / 1000, clamped
        uint32_t price_cat;     // (cat_id << 24) | price_cents
        uint32_t eid_hash32;

        bool is_tombstone() const noexcept { return ts_rel == kFastTombstone; }
        void mark_tombstone() noexcept { ts_rel = kFastTombstone; }
        bool older_than(const RawEvent& other) const noexcept {
            return ingest_rel_sec < other.ingest_rel_sec;
        }
    };
    static_assert(sizeof(RawEvent) == 32);

    struct SurvivingEvent {
        UserId64 user_id;
        uint64_t ts_price_cat;  // (uint64_t(ts_rel) << 32) | price_cat
    };
    static_assert(sizeof(SurvivingEvent) == 16);

    struct SurvivingClick {
        UserId64 user_id;
        ClickId64 click_id;
        uint32_t ts_rel;
        uint32_t camp_id;
    };
    static_assert(sizeof(SurvivingClick) == 24);

    struct ShardEvent {
        uint64_t packed;  // (uint64_t(ts_rel) << 32) | price_cat
    };
    static_assert(sizeof(ShardEvent) == 8);

    struct ShardClick {
        uint32_t ts_rel;
        uint32_t camp_id;
        ClickId64 click_id;
    };
    static_assert(sizeof(ShardClick) == 16);
};

template <>
struct Traits<false> {
    static constexpr size_t kEvBlockCap = 1024;
    static constexpr unsigned kEvBlockShift = 10;
    static constexpr uint32_t kEvBlockMask = 1023u;
    static constexpr size_t kSurvEvBlockCap = 2048;
    static constexpr size_t kSurvClkBlockCap = 2048;

    struct RawEvent {
        EventId64 event_id;
        UserId64 user_id;
        int64_t ingest_ms;
        int64_t ts_ms;  // kSlowTombstone == invalid/late/duplicate-loser
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
        UserId64 user_id;
        int64_t ts_ms;
        uint32_t price_cents;
        uint32_t cat_id;
    };

    struct SurvivingClick {
        int64_t ts_ms;
        UserId64 user_id;
        ClickId64 click_id;
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
        ClickId64 click_id;
        uint32_t camp_id;
        uint32_t pad;
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
    UserId64 user_id;
    uint32_t local_pid = 0;
    uint32_t ev_cnt = 0;
};
static_assert(sizeof(PersonMapSlot) == 16);

struct CampState {
    uint32_t count = 0;
    uint32_t enter_and_pos = 0x00FFFFFFu;  // [24..31]=enter_s, [0..23]=active_pos

    [[gnu::always_inline]] uint32_t pos() const noexcept { return enter_and_pos & 0x00FFFFFFu; }
    [[gnu::always_inline]] uint32_t enter() const noexcept { return enter_and_pos >> 24; }
    [[gnu::always_inline]] void set_pos(uint32_t p) noexcept {
        enter_and_pos = (enter_and_pos & 0xFF000000u) | (p & 0x00FFFFFFu);
    }
    [[gnu::always_inline]] void set_enter_pos(uint32_t k, uint32_t p) noexcept {
        enter_and_pos = (k << 24) | (p & 0x00FFFFFFu);
    }
    [[gnu::always_inline]] void clear_enter() noexcept { enter_and_pos &= 0x00FFFFFFu; }
    [[gnu::always_inline]] void reset() noexcept {
        count = 0;
        enter_and_pos = 0x00FFFFFFu;
    }
};
static_assert(sizeof(CampState) == 8);

template <typename ShardClk>
inline void sort_shard_clicks_fast(ShardClk* arr, size_t n, std::vector<ShardClk>& scratch,
                                   const OverflowArena* const* ca_ptrs) {
    if (n <= 1) return;
    if (n < 256) {
        std::sort(arr, arr + n, [&](const ShardClk& a, const ShardClk& b) {
            if (a.ts_rel != b.ts_rel) return a.ts_rel < b.ts_rel;
            return ClickId64::less(b.click_id, a.click_id, ca_ptrs);
        });
        return;
    }

    uint32_t cnt0[2048]{};
    uint32_t cnt1[2048]{};
    uint32_t cnt2[512]{};
    for (size_t i = 0; i < n; ++i) {
        const uint32_t k = arr[i].ts_rel;
        ++cnt0[k & 0x7FFu];
        ++cnt1[(k >> 11) & 0x7FFu];
        ++cnt2[k >> 22];
    }

    scratch.resize(n);
    ShardClk* src = arr;
    ShardClk* dst = scratch.data();

    auto do_pass = [&](const uint32_t* cnt, size_t rad, unsigned shift, uint32_t mask) {
        for (size_t b = 0; b < rad; ++b) {
            if (cnt[b] == n) return;
        }
        uint32_t off[2048];
        uint32_t sum = 0;
        for (size_t b = 0; b < rad; ++b) {
            off[b] = sum;
            sum += cnt[b];
        }
        for (size_t i = 0; i < n; ++i) {
            const uint32_t k = (src[i].ts_rel >> shift) & mask;
            dst[off[k]++] = src[i];
        }
        std::swap(src, dst);
    };

    do_pass(cnt0, 2048, 0, 0x7FFu);
    do_pass(cnt1, 2048, 11, 0x7FFu);
    do_pass(cnt2, 512, 22, 0x1FFu);

    if (src != arr) {
        std::memcpy(arr, src, n * sizeof(ShardClk));
    }

    size_t i = 0;
    while (i + 1 < n) {
        if (__builtin_expect(arr[i].ts_rel == arr[i + 1].ts_rel, 0)) {
            size_t j = i + 2;
            while (j < n && arr[j].ts_rel == arr[i].ts_rel) ++j;
            std::sort(arr + i, arr + j, [&](const ShardClk& a, const ShardClk& b) {
                return ClickId64::less(b.click_id, a.click_id, ca_ptrs);
            });
            i = j;
        } else {
            ++i;
        }
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

[[gnu::always_inline]] inline uint32_t lookup_product_fast(
    std::string_view sv, const std::vector<uint8_t>& fast_prod_by_num,
    const FlatStrMap<uint32_t>& product_map, uint32_t unknown_cat_id) noexcept {
    const size_t n = sv.size();
    const char* s = sv.data();
    if (__builtin_expect(n >= 2 && s[0] == 'p' && static_cast<uint8_t>(s[1] - '1') <= 8, 1)) {
        uint64_t num = static_cast<uint64_t>(s[1] - '0');
        bool is_canon = true;
        for (size_t i = 2; i < n; ++i) {
            const uint8_t d = static_cast<uint8_t>(s[i] - '0');
            if (__builtin_expect(d > 9 || num > 1000000ULL, 0)) {
                is_canon = false;
                break;
            }
            num = num * 10 + d;
        }
        if (__builtin_expect(is_canon, 1)) {
            if (num < fast_prod_by_num.size()) {
                const uint8_t c = fast_prod_by_num[num];
                return (c != 0xFFu) ? static_cast<uint32_t>(c) : unknown_cat_id;
            }
            return unknown_cat_id;
        }
    }
    const uint32_t* pc = product_map.find(sv);
    return pc ? *pc : unknown_cat_id;
}

[[gnu::always_inline]] inline uint32_t lookup_campaign_fast(
    std::string_view sv, const std::vector<uint32_t>& fast_camp_by_num,
    const FlatStrMap<uint32_t>& campaign_map) noexcept {
    if (__builtin_expect(sv.size() >= 3 && sv[0] == 'c' && sv[1] == '_' &&
                             static_cast<uint8_t>(sv[2] - '1') <= 8,
                         1)) {
        uint64_t num = static_cast<uint64_t>(sv[2] - '0');
        bool is_canon = true;
        for (size_t i = 3; i < sv.size(); ++i) {
            const uint8_t d = static_cast<uint8_t>(sv[i] - '0');
            if (__builtin_expect(d > 9 || num > 1000000ULL, 0)) {
                is_canon = false;
                break;
            }
            num = num * 10 + d;
        }
        if (__builtin_expect(is_canon, 1)) {
            return (num < fast_camp_by_num.size()) ? fast_camp_by_num[num]
                                                   : std::numeric_limits<uint32_t>::max();
        }
    }
    const uint32_t* p = campaign_map.find(sv);
    return p ? *p : std::numeric_limits<uint32_t>::max();
}

template <typename TSlice, size_t BlockCap>
[[gnu::noinline]] TSlice* alloc_bucket_block(ThreadBlockArena& arena, TSlice* cur, TSlice*& endp,
                                             std::vector<BlockSlice<TSlice>>& out) {
    if (cur != nullptr) {
        out.push_back(BlockSlice<TSlice>{cur - BlockCap, static_cast<uint32_t>(BlockCap)});
    }
    TSlice* blk = static_cast<TSlice*>(arena.alloc());
    endp = blk + BlockCap;
    return blk;
}

template <bool FastPack, typename PushFn>
void process_events_chunk(bool use_avx512, const PreadView& view, uint16_t aid, OverflowArena& ea,
                          OverflowArena& ua, const OverflowArena* const* ea_ptrs,
                          const FastDateLut& ev_date_lut, int64_t event_win_start,
                          int64_t event_win_end, const std::vector<uint8_t>& fast_prod_by_num,
                          const FlatStrMap<uint32_t>& product_map, uint32_t unknown_cat_id,
                          size_t dedup_shift, PushFn&& push_ev) {
    using T = Traits<FastPack>;
    using RawEv = typename T::RawEvent;

    auto handle_row_impl = [&](const std::string_view (&row)[7],
                               auto is_padded_tag) __attribute__((always_inline)) {
        constexpr bool is_padded = decltype(is_padded_tag)::value;
        if (__builtin_expect(row[0].empty(), 0)) return;

        int64_t ingest_ms = 0;
        int32_t ingest_rel_sec = 0;
        uint64_t ihi8 = 0;
        uint16_t ilo2 = 0;
        int64_t imid = 0;
        bool ilut = false;
        if (__builtin_expect(!parse_ts_f1_sec(row[3], ev_date_lut, event_win_start, ingest_ms,
                                              ingest_rel_sec, ihi8, ilo2, imid, ilut),
                             0)) {
            return;
        }

        const EventId64 eid = is_padded ? EventId64::make_str_padded(row[0], aid, ea)
                                        : EventId64::make_str_safe(row[0], aid, ea);
        const uint64_t eh = eid.hash(ea_ptrs);
        const size_t b = static_cast<size_t>(eh >> dedup_shift);

        bool valid = !row[1].empty() && !row[5].empty();
        int64_t ts_ms = 0;
        if (__builtin_expect(valid, 1)) {
            valid = parse_ts_any_with_hint(row[2], ev_date_lut, ilut, ihi8, ilo2, imid, ts_ms) &&
                    ts_ms >= event_win_start && ts_ms < event_win_end &&
                    (ingest_ms - ts_ms) <= kLateCutoffMs;
        }

        uint32_t price_cents = 0;
        uint32_t cat_id = 0;
        if (__builtin_expect(valid, 1)) {
            const std::string_view et = row[4];
            if (__builtin_expect(et.size() == 4, 1)) {
                if (__builtin_expect(std::memcmp(et.data(), "view", 4) != 0 &&
                                         std::memcmp(et.data(), "cart", 4) != 0,
                                     0)) {
                    valid = false;
                }
            } else if (et.size() == 8 && std::memcmp(et.data(), "purchase", 8) == 0) {
                valid = parse_price_cents(row[6], price_cents);
                if (valid) {
                    cat_id =
                        lookup_product_fast(row[5], fast_prod_by_num, product_map, unknown_cat_id);
                }
            } else {
                valid = false;
            }
        }

        RawEv re{};
        re.event_id = eid;
        re.eid_hash32 = static_cast<uint32_t>(eh);
        if constexpr (FastPack) {
            re.ingest_rel_sec = ingest_rel_sec;
            if (__builtin_expect(valid, 1)) {
                re.user_id = is_padded ? UserId64::make_str_padded(row[1], aid, ua)
                                       : UserId64::make_str_safe(row[1], aid, ua);
                re.ts_rel = static_cast<uint32_t>(ts_ms - event_win_start);
                re.price_cat = (cat_id << 24) | price_cents;
            } else {
                re.ts_rel = kFastTombstone;
            }
        } else {
            re.ingest_ms = ingest_ms;
            if (valid) {
                re.user_id = is_padded ? UserId64::make_str_padded(row[1], aid, ua)
                                       : UserId64::make_str_safe(row[1], aid, ua);
                re.ts_ms = ts_ms;
                re.price_cents = price_cents;
                re.cat_id = cat_id;
            } else {
                re.ts_ms = kSlowTombstone;
            }
        }
        push_ev(b, re);
    };

    auto handle_fast = [&](const std::string_view (&row)[7]) __attribute__((always_inline)) {
        handle_row_impl(row, std::true_type{});
    };
    auto handle_slow = [&](const std::string_view (&row)[7]) {
        handle_row_impl(row, std::false_type{});
    };

    if (use_avx512) {
        csv_simd::scan_pread_buf_avx512<7>(view, handle_fast, handle_slow);
    } else {
        csv_simd::scan_pread_buf_avx2<7>(view, handle_fast, handle_slow);
    }
}

template <typename RowFn>
void process_clicks_chunk(bool use_avx512, const PreadView& view, const FastDateLut& clk_date_lut,
                          int64_t click_win_start, int64_t click_win_end,
                          const std::vector<uint32_t>& fast_camp_by_num,
                          const FlatStrMap<uint32_t>& campaign_map, RowFn&& on_valid_click) {
    auto handle_row_impl = [&](const std::string_view (&row)[4],
                               auto is_padded_tag) __attribute__((always_inline)) {
        if (__builtin_expect(row[0].empty() || row[1].empty(), 0)) return;
        const uint32_t cid = lookup_campaign_fast(row[2], fast_camp_by_num, campaign_map);
        if (__builtin_expect(cid == std::numeric_limits<uint32_t>::max(), 0)) return;
        int64_t ts_ms = 0;
        if (__builtin_expect(!parse_ts_any_swar(row[3], clk_date_lut, ts_ms) ||
                                 ts_ms < click_win_start || ts_ms >= click_win_end,
                             0)) {
            return;
        }
        on_valid_click(row[0], row[1], cid, ts_ms, is_padded_tag);
    };

    auto handle_fast = [&](const std::string_view (&row)[4]) __attribute__((always_inline)) {
        handle_row_impl(row, std::true_type{});
    };
    auto handle_slow = [&](const std::string_view (&row)[4]) {
        handle_row_impl(row, std::false_type{});
    };

    if (use_avx512) {
        csv_simd::scan_pread_buf_avx512<4>(view, handle_fast, handle_slow);
    } else {
        csv_simd::scan_pread_buf_avx2<4>(view, handle_fast, handle_slow);
    }
}

template <bool FastPack>
void run_pipeline(const std::string& data_dir, const std::string& output_dir, ThreadPool& pool,
                  int64_t event_win_start, int64_t event_win_end, int64_t click_win_start,
                  int64_t click_win_end, int32_t day_start, int32_t day_end,
                  const FastDateLut& ev_date_lut, const FastDateLut& clk_date_lut,
                  const std::vector<std::string>& cat_names, uint32_t unknown_cat_id,
                  const std::vector<uint8_t>& fast_prod_by_num, FlatStrMap<uint32_t>& product_map,
                  const std::vector<CampaignInfo>& campaigns,
                  const std::vector<uint32_t>& fast_camp_by_num,
                  FlatStrMap<uint32_t>& campaign_map, std::vector<PreadFile>& ev_files,
                  std::vector<PreadFile>& clk_files, size_t total_ev_bytes,
                  std::chrono::steady_clock::time_point t0) {
    using T = Traits<FastPack>;
    using RawEv = typename T::RawEvent;
    using SurvEv = typename T::SurvivingEvent;
    using SurvClk = typename T::SurvivingClick;
    using ShardEv = typename T::ShardEvent;
    using ShardClk = typename T::ShardClick;

    const size_t ncpu = pool.size();
    const bool use_avx512 = cpu_has_avx512bw();

    auto ev_chunks = build_pread_chunks(ev_files, size_t(256) << 10);
    auto clk_chunks = build_pread_chunks(clk_files, size_t(256) << 10);

    const size_t est_events = total_ev_bytes / 72;
    size_t dedup_buckets = 64;
    if (est_events >= 35000000ULL) dedup_buckets = 512;
    else if (est_events >= 2000000ULL) dedup_buckets = 256;

    size_t person_shards = 64;
    if (est_events >= 35000000ULL) person_shards = 512;
    else if (est_events >= 2000000ULL) person_shards = 256;

    const unsigned dedup_shift = 64u - log2_pow2(dedup_buckets);
    const unsigned person_shift = 64u - log2_pow2(person_shards);

    std::vector<ThreadBlockArena> block_arenas(ncpu);
    std::vector<OverflowArena> event_arenas(ncpu);
    std::vector<OverflowArena> user_arenas(ncpu * 2);
    std::vector<OverflowArena> click_arenas(ncpu);

    std::vector<const OverflowArena*> event_arena_ptrs(ncpu);
    std::vector<const OverflowArena*> click_arena_ptrs(ncpu);
    std::vector<const OverflowArena*> all_user_arenas(ncpu * 2 + LinkTable::kShards);
    for (size_t t = 0; t < ncpu; ++t) {
        event_arena_ptrs[t] = &event_arenas[t];
        click_arena_ptrs[t] = &click_arenas[t];
        all_user_arenas[t] = &user_arenas[t];
        all_user_arenas[ncpu + t] = &user_arenas[ncpu + t];
    }

    LinkTable links;
    const uint16_t link_arena_base = static_cast<uint16_t>(ncpu * 2);
    for (size_t s = 0; s < LinkTable::kShards; ++s) {
        all_user_arenas[link_arena_base + s] = &links.node_arena(s);
    }

    std::vector<std::vector<std::vector<BlockSlice<RawEv>>>> raw_ev_buckets(
        ncpu, std::vector<std::vector<BlockSlice<RawEv>>>(dedup_buckets));

    // Phase 1: Parallel 64-shard LinkTable build
    const auto t_pre_links = std::chrono::steady_clock::now();
    links.build(join_path(data_dir, "links"), pool, block_arenas, all_user_arenas.data(),
                link_arena_base);
    const auto t_links = std::chrono::steady_clock::now();

    // Phase 2: Parallel 256 KB pread + SIMD events scan with L1-resident bucket pointers
    if (!ev_chunks.empty()) {
        std::atomic<size_t> next_ev_chunk{0};
        pool.run([&](size_t tid) {
            ThreadBlockArena& my_arena = block_arenas[tid];
            OverflowArena& ea = event_arenas[tid];
            OverflowArena& ua = user_arenas[tid];
            const OverflowArena* const* ea_ptrs = event_arena_ptrs.data();
            auto& buckets = raw_ev_buckets[tid];
            const uint16_t aid = static_cast<uint16_t>(tid);

            std::vector<char> chunk_buf;
            PreadView view;

            RawEv* cur[512]{};
            RawEv* endp[512]{};
            auto push_ev = [&](size_t b, const RawEv& re) __attribute__((always_inline)) {
                if (__builtin_expect(cur[b] == endp[b], 0)) {
                    cur[b] = alloc_bucket_block<RawEv, T::kEvBlockCap>(my_arena, cur[b], endp[b],
                                                                       buckets[b]);
                }
                *cur[b]++ = re;
            };

            for (;;) {
                const size_t ci = next_ev_chunk.fetch_add(1, std::memory_order_relaxed);
                if (ci >= ev_chunks.size()) break;
                const PreadChunk& pc = ev_chunks[ci];
                if (!read_chunk_into(ev_files[pc.file_idx], pc, chunk_buf, view)) continue;
                process_events_chunk<FastPack>(use_avx512, view, aid, ea, ua, ea_ptrs,
                                               ev_date_lut, event_win_start, event_win_end,
                                               fast_prod_by_num, product_map, unknown_cat_id,
                                               dedup_shift, push_ev);
            }

            for (size_t b = 0; b < dedup_buckets; ++b) {
                if (cur[b] != nullptr) {
                    RawEv* base = endp[b] - T::kEvBlockCap;
                    const uint32_t len = static_cast<uint32_t>(cur[b] - base);
                    if (len > 0) buckets[b].push_back(BlockSlice<RawEv>{base, len});
                }
            }
        });
    }
    ev_chunks.clear();
    ev_files.clear();
    product_map.clear_and_free();
    const auto t_phase_a = std::chrono::steady_clock::now();

    // Phase B: Overlap dedup + Bloom/LinkTable resolution with clicks/*.csv scan
    std::vector<std::vector<std::vector<BlockSlice<SurvEv>>>> ev_person_buckets(
        ncpu, std::vector<std::vector<BlockSlice<SurvEv>>>(person_shards));
    std::vector<std::vector<std::vector<BlockSlice<SurvClk>>>> clk_person_buckets(
        ncpu, std::vector<std::vector<BlockSlice<SurvClk>>>(person_shards));

    const size_t clk_starter_threads = clk_chunks.empty() ? 0 : (ncpu >= 4 ? 2 : 1);
    const bool has_links = !links.empty();

    {
        std::atomic<size_t> next_dedup_b{0};
        std::atomic<size_t> next_clk_chunk{0};

        pool.run([&](size_t tid) {
            ThreadBlockArena& my_arena = block_arenas[tid];
            const OverflowArena* const* ea_ptrs = event_arena_ptrs.data();
            const OverflowArena* const* ua_ptrs = all_user_arenas.data();
            auto& out_ev_buckets = ev_person_buckets[tid];
            auto& out_clk_buckets = clk_person_buckets[tid];
            OverflowArena& ua = user_arenas[ncpu + tid];
            OverflowArena& ca = click_arenas[tid];
            const uint16_t clk_uid_aid = static_cast<uint16_t>(ncpu + tid);
            const uint16_t clk_kid_aid = static_cast<uint16_t>(tid);

            auto run_dedup = [&]() {
                std::vector<BlockSlice<RawEv>> block_refs;
                std::vector<DedupIdx> idx_table;
                SurvEv* sev_cur[512]{};
                SurvEv* sev_end[512]{};
                auto push_sev = [&](size_t s, const SurvEv& se) {
                    if (__builtin_expect(sev_cur[s] == sev_end[s], 0)) {
                        if (sev_cur[s] != nullptr) {
                            out_ev_buckets[s].push_back(BlockSlice<SurvEv>{
                                sev_cur[s] - T::kSurvEvBlockCap, T::kSurvEvBlockCap});
                        }
                        SurvEv* blk = static_cast<SurvEv*>(my_arena.alloc());
                        sev_cur[s] = blk;
                        sev_end[s] = blk + T::kSurvEvBlockCap;
                    }
                    *sev_cur[s]++ = se;
                };

                for (;;) {
                    const size_t b = next_dedup_b.fetch_add(1, std::memory_order_relaxed);
                    if (b >= dedup_buckets) break;

                    block_refs.clear();
                    size_t total = 0;
                    for (size_t t = 0; t < ncpu; ++t) {
                        for (const auto& sl : raw_ev_buckets[t][b]) {
                            if (sl.len > 0) {
                                block_refs.push_back(sl);
                                total += sl.len;
                            }
                        }
                    }
                    if (total == 0) continue;

                    size_t cap = 64;
                    while (cap * 13 < total * 20) cap <<= 1;
                    idx_table.resize(cap);
                    std::memset(static_cast<void*>(idx_table.data()), 0, cap * sizeof(DedupIdx));
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
                            const uint32_t code = (bi << T::kEvBlockShift) | i;
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
                                    RawEv& best =
                                        block_refs[bcode >> T::kEvBlockShift]
                                            .ptr[bcode & T::kEvBlockMask];
                                    if (EventId64::equals(best.event_id, re.event_id, ea_ptrs)) {
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

                    // 2-stage Bloom + LinkTable software prefetch pipeline
                    constexpr size_t batch_size = 32;
                    for (const auto& block : block_refs) {
                        for (size_t start = 0; start < block.len; start += batch_size) {
                            const RawEv* surv[batch_size];
                            uint64_t hashes[batch_size];
                            UserId64 uids[batch_size];
                            size_t count = 0;
                            const size_t end = std::min(start + batch_size, size_t(block.len));

                            for (size_t i = start; i < end; ++i) {
                                const RawEv& re = block.ptr[i];
                                if (re.is_tombstone()) continue;
                                const uint64_t h = re.user_id.hash(ua_ptrs);
                                if (has_links) links.prefetch_bloom(h);
                                surv[count] = &re;
                                uids[count] = re.user_id;
                                hashes[count++] = h;
                            }

                            if (has_links) {
                                uint8_t hit_idx[batch_size];
                                size_t nhits = 0;
                                for (size_t i = 0; i < count; ++i) {
                                    if (links.may_contain(hashes[i])) {
                                        links.prefetch_slot(hashes[i]);
                                        hit_idx[nhits++] = static_cast<uint8_t>(i);
                                    }
                                }
                                for (size_t j = 0; j < nhits; ++j) {
                                    const size_t i = hit_idx[j];
                                    UserId64 canon;
                                    if (links.resolve_with_hash(uids[i], hashes[i], canon,
                                                                ua_ptrs)) {
                                        uids[i] = canon;
                                        hashes[i] = canon.hash(ua_ptrs);
                                    }
                                }
                            }

                            for (size_t i = 0; i < count; ++i) {
                                const RawEv& re = *surv[i];
                                const size_t s = static_cast<size_t>(hashes[i] >> person_shift);
                                if constexpr (FastPack) {
                                    const uint64_t pk =
                                        (static_cast<uint64_t>(re.ts_rel) << 32) | re.price_cat;
                                    push_sev(s, SurvEv{uids[i], pk});
                                } else {
                                    push_sev(s, SurvEv{uids[i], re.ts_ms, re.price_cents, re.cat_id});
                                }
                            }
                        }
                    }

                    for (size_t t = 0; t < ncpu; ++t) {
                        for (const auto& sl : raw_ev_buckets[t][b]) {
                            my_arena.recycle(sl.ptr);
                        }
                        std::vector<BlockSlice<RawEv>>().swap(raw_ev_buckets[t][b]);
                    }
                }

                for (size_t s = 0; s < person_shards; ++s) {
                    if (sev_cur[s] != nullptr) {
                        SurvEv* base = sev_end[s] - T::kSurvEvBlockCap;
                        const uint32_t len = static_cast<uint32_t>(sev_cur[s] - base);
                        if (len > 0) out_ev_buckets[s].push_back(BlockSlice<SurvEv>{base, len});
                    }
                }
            };

            auto run_clicks = [&]() {
                if (clk_chunks.empty()) return;

                SurvClk* sclk_cur[512]{};
                SurvClk* sclk_end[512]{};
                auto push_sclk = [&](size_t s, const SurvClk& sc) {
                    if (__builtin_expect(sclk_cur[s] == sclk_end[s], 0)) {
                        if (sclk_cur[s] != nullptr) {
                            out_clk_buckets[s].push_back(BlockSlice<SurvClk>{
                                sclk_cur[s] - T::kSurvClkBlockCap, T::kSurvClkBlockCap});
                        }
                        SurvClk* blk = static_cast<SurvClk*>(my_arena.alloc());
                        sclk_cur[s] = blk;
                        sclk_end[s] = blk + T::kSurvClkBlockCap;
                    }
                    *sclk_cur[s]++ = sc;
                };

                struct PendingClk {
                    int64_t ts_ms;
                    uint64_t uh;
                    UserId64 uid_try;
                    ClickId64 kid;
                    std::string_view uid_sv;
                    uint32_t camp_id;
                };
                PendingClk batch[32];
                size_t bsz = 0;

                auto flush_batch = [&]() {
                    if (bsz == 0) return;
                    if (has_links) {
                        uint8_t hit_idx[32];
                        size_t nhits = 0;
                        for (size_t i = 0; i < bsz; ++i) {
                            if (links.may_contain(batch[i].uh)) {
                                links.prefetch_slot(batch[i].uh);
                                hit_idx[nhits++] = static_cast<uint8_t>(i);
                            }
                        }
                        for (size_t j = 0; j < nhits; ++j) {
                            PendingClk& pc = batch[hit_idx[j]];
                            UserId64 canon;
                            const bool found =
                                (pc.uid_try.raw != 0)
                                    ? links.resolve_with_hash(pc.uid_try, pc.uh, canon, ua_ptrs)
                                    : links.resolve_sv_with_hash(pc.uid_sv, pc.uh, canon, ua_ptrs);
                            if (found) {
                                pc.uid_try = canon;
                                pc.uh = canon.hash(ua_ptrs);
                            }
                        }
                    }

                    for (size_t i = 0; i < bsz; ++i) {
                        const PendingClk& pc = batch[i];
                        UserId64 uid = pc.uid_try;
                        if (__builtin_expect(uid.raw == 0, 0)) {
                            uid = UserId64::make_overflow(pc.uid_sv, clk_uid_aid, ua);
                        }
                        const size_t s = static_cast<size_t>(pc.uh >> person_shift);
                        if constexpr (FastPack) {
                            push_sclk(s, SurvClk{uid, pc.kid,
                                                 static_cast<uint32_t>(pc.ts_ms - click_win_start),
                                                 pc.camp_id});
                        } else {
                            push_sclk(s, SurvClk{pc.ts_ms, uid, pc.kid, pc.camp_id, 0});
                        }
                    }
                    bsz = 0;
                };

                auto on_valid_click = [&](std::string_view kid_sv, std::string_view uid_sv,
                                          uint32_t cid, int64_t ts_ms,
                                          auto is_padded_tag) __attribute__((always_inline)) {
                    constexpr bool is_padded = decltype(is_padded_tag)::value;
                    UserId64 uid_try =
                        is_padded ? UserId64::try_inline_padded(uid_sv.data(), uid_sv.size())
                                  : UserId64::make_str_safe(uid_sv, clk_uid_aid, ua);
                    const uint64_t uh = (uid_try.raw != 0)
                                            ? uid_try.hash(ua_ptrs)
                                            : UserId64::hash_sv(uid_sv);
                    if (has_links) links.prefetch_bloom(uh);
                    const ClickId64 kid = is_padded ? ClickId64::make_str_padded(kid_sv, clk_kid_aid, ca)
                                                    : ClickId64::make_str_safe(kid_sv, clk_kid_aid, ca);
                    batch[bsz++] = PendingClk{ts_ms, uh, uid_try, kid, uid_sv, cid};
                    if (bsz == 32) flush_batch();
                };

                std::vector<char> chunk_buf;
                PreadView view;
                for (;;) {
                    const size_t ci = next_clk_chunk.fetch_add(1, std::memory_order_relaxed);
                    if (ci >= clk_chunks.size()) break;
                    const PreadChunk& pc = clk_chunks[ci];
                    if (!read_chunk_into(clk_files[pc.file_idx], pc, chunk_buf, view)) continue;
                    process_clicks_chunk(use_avx512, view, clk_date_lut, click_win_start,
                                         click_win_end, fast_camp_by_num, campaign_map,
                                         on_valid_click);
                    flush_batch();
                }

                for (size_t s = 0; s < person_shards; ++s) {
                    if (sclk_cur[s] != nullptr) {
                        SurvClk* base = sclk_end[s] - T::kSurvClkBlockCap;
                        const uint32_t len = static_cast<uint32_t>(sclk_cur[s] - base);
                        if (len > 0) out_clk_buckets[s].push_back(BlockSlice<SurvClk>{base, len});
                    }
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
    if (std::getenv("SS_PGO_RUN")) {
        for (auto& ea : event_arenas) ea.free_all();
    }
    campaign_map.clear_and_free();
    const auto t_phase_b = std::chrono::steady_clock::now();

    // Phase C: Parallel processing of person shards (LPT descending)
    std::vector<size_t> shard_order(person_shards);
    std::iota(shard_order.begin(), shard_order.end(), 0);
    std::vector<size_t> shard_load(person_shards, 0);
    for (size_t s = 0; s < person_shards; ++s) {
        size_t load = 0;
        for (size_t t = 0; t < ncpu; ++t) {
            for (const auto& sl : ev_person_buckets[t][s]) load += sl.len;
            for (const auto& sl : clk_person_buckets[t][s]) load += sl.len;
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

    // Transposed [day_idx * num_camps + camp_id] keeps active days L1/L2 resident!
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

    const int64_t first_next_day_rel =
        static_cast<int64_t>(day_start + 1) * 86400000LL - event_win_start;

    {
        std::atomic<size_t> next_order_idx{0};
        pool.run([&](size_t tid) {
            ThreadBlockArena& my_arena = block_arenas[tid];
            const OverflowArena* const* ua_ptrs = all_user_arenas.data();
            const OverflowArena* const* ca_ptrs = click_arena_ptrs.data();
            auto& agg = thread_aggs[tid];

            const std::string out_path =
                join_path(join_path(output_dir, "user_daily"),
                          "part-" + std::to_string(tid) + ".parquet");
            UserDailyParquetWriter writer(out_path);

            auto add_credit = [&](uint32_t camp_id, int32_t day, uint8_t has_purch, uint64_t rev) {
                if (use_flat_camp) {
                    CampCreditAgg& cd =
                        agg.flat_credits[static_cast<size_t>(day - day_start) * num_camps +
                                         static_cast<size_t>(camp_id)];
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
            auto add_assist_n = [&](uint32_t camp_id, int32_t day, int32_t delta) {
                if (use_flat_camp) {
                    agg.flat_assists[static_cast<size_t>(day - day_start) * num_camps +
                                     static_cast<size_t>(camp_id)] += delta;
                } else {
                    agg.map.get(camp_id, day).assisted_sessions += delta;
                }
            };

            std::vector<PersonMapSlot> pmap;
            std::vector<UserId64> pid_user;
            std::vector<uint32_t> ev_off, clk_cnt, clk_off, ev_cur, clk_cur;
            std::vector<uint32_t> tmp_ev_pids, tmp_clk_pids;
            std::vector<ShardEv> tmp_evs, shard_events;
            std::vector<ShardClk> tmp_clks, shard_clicks, clk_radix_scratch;
            std::vector<uint64_t> radix_scratch, packed_clk_keys, radix_scratch2;
            std::vector<SessionInfo> sessions;
            std::vector<uint64_t> cat_rev(num_cats, 0);
            std::vector<uint32_t> touched_cats;
            std::vector<CampState> camp_state(num_camps);
            std::vector<uint32_t> active_camps;

            for (;;) {
                const size_t oi = next_order_idx.fetch_add(1, std::memory_order_relaxed);
                if (oi >= person_shards) break;
                const size_t s = shard_order[oi];

                size_t total_ev = 0;
                for (size_t t = 0; t < ncpu; ++t) {
                    for (const auto& sl : ev_person_buckets[t][s]) total_ev += sl.len;
                }
                if (total_ev == 0) {
                    for (size_t t = 0; t < ncpu; ++t) {
                        for (const auto& sl : clk_person_buckets[t][s]) my_arena.recycle(sl.ptr);
                        std::vector<BlockSlice<SurvClk>>().swap(clk_person_buckets[t][s]);
                    }
                    continue;
                }

                size_t pcap = 64;
                while (pcap < total_ev / 14) pcap <<= 1;
                if (pcap < 128) pcap = 128;
                pmap.resize(pcap);
                std::memset(static_cast<void*>(pmap.data()), 0, pcap * sizeof(PersonMapSlot));
                size_t pmask = pcap - 1;
                pid_user.clear();
                tmp_ev_pids.resize(total_ev);
                tmp_evs.resize(total_ev);
                uint32_t* ev_pid_out = tmp_ev_pids.data();
                ShardEv* ev_out = tmp_evs.data();
                size_t out_ev_i = 0;

                auto rehash_pmap = [&]() {
                    const size_t ncap = pmap.size() * 2;
                    std::vector<PersonMapSlot> nxt(ncap);
                    const size_t nmask = ncap - 1;
                    for (const auto& sl : pmap) {
                        if (sl.user_id.empty()) continue;
                        const uint64_t h = sl.user_id.hash(ua_ptrs);
                        size_t idx = static_cast<size_t>(h) & nmask;
                        while (!nxt[idx].user_id.empty()) idx = (idx + 1) & nmask;
                        nxt[idx] = sl;
                    }
                    pmap.swap(nxt);
                    pmask = nmask;
                };

                auto find_pid = [&](uint64_t h, UserId64 uid) -> uint32_t {
                    size_t idx = static_cast<size_t>(h) & pmask;
                    for (;;) {
                        const PersonMapSlot& sl = pmap[idx];
                        if (sl.user_id.empty()) return std::numeric_limits<uint32_t>::max();
                        if (UserId64::equals(sl.user_id, uid, ua_ptrs)) return sl.local_pid;
                        idx = (idx + 1) & pmask;
                    }
                };

                for (size_t t = 0; t < ncpu; ++t) {
                    for (const BlockSlice<SurvEv>& blk : ev_person_buckets[t][s]) {
                        const SurvEv* ptr = blk.ptr;
                        const size_t n = blk.len;
                        for (size_t i = 0; i < n; ++i) {
                            if (i + 8 < n) {
                                const uint64_t nh = ptr[i + 8].user_id.hash(ua_ptrs);
                                __builtin_prefetch(&pmap[static_cast<size_t>(nh) & pmask], 1, 1);
                            }
                            const SurvEv& se = ptr[i];
                            if (__builtin_expect((pid_user.size() + 1) * 20 >= pmap.size() * 11,
                                                 0)) {
                                rehash_pmap();
                            }
                            const uint64_t h = se.user_id.hash(ua_ptrs);
                            size_t idx = static_cast<size_t>(h) & pmask;
                            uint32_t pid = 0;
                            for (;;) {
                                PersonMapSlot& sl = pmap[idx];
                                if (sl.user_id.empty()) {
                                    pid = static_cast<uint32_t>(pid_user.size());
                                    sl.user_id = se.user_id;
                                    sl.local_pid = pid;
                                    sl.ev_cnt = 1;
                                    pid_user.push_back(se.user_id);
                                    break;
                                }
                                if (UserId64::equals(sl.user_id, se.user_id, ua_ptrs)) {
                                    pid = sl.local_pid;
                                    ++sl.ev_cnt;
                                    break;
                                }
                                idx = (idx + 1) & pmask;
                            }
                            ev_pid_out[out_ev_i] = pid;
                            if constexpr (FastPack) {
                                ev_out[out_ev_i] = ShardEv{se.ts_price_cat};
                            } else {
                                ev_out[out_ev_i] = ShardEv{se.ts_ms, se.price_cents, se.cat_id};
                            }
                            ++out_ev_i;
                        }
                        my_arena.recycle(blk.ptr);
                    }
                    std::vector<BlockSlice<SurvEv>>().swap(ev_person_buckets[t][s]);
                }

                const size_t num_persons = pid_user.size();
                ev_off.resize(num_persons + 1);
                ev_off[0] = 0;
                for (const PersonMapSlot& sl : pmap) {
                    if (!sl.user_id.empty()) {
                        ev_off[sl.local_pid + 1] = sl.ev_cnt;
                    }
                }
                for (size_t p = 0; p < num_persons; ++p) ev_off[p + 1] += ev_off[p];
                ev_cur.assign(ev_off.begin(), ev_off.begin() + num_persons);
                shard_events.resize(total_ev);
                for (size_t i = 0; i < total_ev; ++i) {
                    shard_events[ev_cur[ev_pid_out[i]]++] = ev_out[i];
                }

                clk_cnt.assign(num_persons, 0);
                tmp_clk_pids.clear();
                tmp_clks.clear();
                for (size_t t = 0; t < ncpu; ++t) {
                    for (const BlockSlice<SurvClk>& blk : clk_person_buckets[t][s]) {
                        const SurvClk* ptr = blk.ptr;
                        const size_t n = blk.len;
                        for (size_t i = 0; i < n; ++i) {
                            if (i + 8 < n) {
                                const uint64_t nh = ptr[i + 8].user_id.hash(ua_ptrs);
                                __builtin_prefetch(&pmap[static_cast<size_t>(nh) & pmask], 0, 1);
                            }
                            const SurvClk& sc = ptr[i];
                            const uint64_t h = sc.user_id.hash(ua_ptrs);
                            const uint32_t pid = find_pid(h, sc.user_id);
                            if (pid != std::numeric_limits<uint32_t>::max()) {
                                ++clk_cnt[pid];
                                tmp_clk_pids.push_back(pid);
                                if constexpr (FastPack) {
                                    tmp_clks.push_back(ShardClk{sc.ts_rel, sc.camp_id, sc.click_id});
                                } else {
                                    tmp_clks.push_back(ShardClk{sc.ts_ms, sc.click_id, sc.camp_id, 0});
                                }
                            }
                        }
                        my_arena.recycle(blk.ptr);
                    }
                    std::vector<BlockSlice<SurvClk>>().swap(clk_person_buckets[t][s]);
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

                auto run_attr_window = [&](size_t nclk, auto&& clk_ts_fn, auto&& clk_camp_fn) {
                    if (nclk <= 16) {
                        struct SmallAct {
                            uint32_t camp;
                            uint32_t cnt;
                            uint32_t enter_s;
                        };
                        SmallAct act[16];
                        size_t nact = 0;
                        size_t l = 0, r = 0;
                        const size_t nsess = sessions.size();
                        size_t si = 0;
                        while (si < nsess) {
                            const int32_t day = sessions[si].day;
                            size_t day_end_idx = si + 1;
                            while (day_end_idx < nsess && sessions[day_end_idx].day == day) {
                                ++day_end_idx;
                            }
                            const uint32_t K = static_cast<uint32_t>(day_end_idx - si);

                            for (uint32_t k = 0; k < K; ++k) {
                                const SessionInfo& sess = sessions[si + k];
                                int64_t win_lo, win_hi;
                                if constexpr (FastPack) {
                                    win_lo = sess.start_key;
                                    win_hi = sess.start_key + kClickReachMs;
                                } else {
                                    win_lo = sess.start_key - kClickReachMs;
                                    win_hi = sess.start_key;
                                }

                                if (l == r) {
                                    while (r < nclk && clk_ts_fn(r) < win_lo) ++r;
                                    l = r;
                                }
                                while (r < nclk && clk_ts_fn(r) <= win_hi) {
                                    const uint32_t c = clk_camp_fn(r);
                                    size_t idx = 0;
                                    while (idx < nact && act[idx].camp != c) ++idx;
                                    if (idx < nact) {
                                        ++act[idx].cnt;
                                    } else {
                                        act[nact++] = SmallAct{c, 1, k};
                                    }
                                    ++r;
                                }
                                while (l < r && clk_ts_fn(l) < win_lo) {
                                    const uint32_t c = clk_camp_fn(l);
                                    size_t idx = 0;
                                    while (act[idx].camp != c) ++idx;
                                    if (--act[idx].cnt == 0) {
                                        if (k > act[idx].enter_s) {
                                            add_assist_n(c, day,
                                                         static_cast<int32_t>(k - act[idx].enter_s));
                                        }
                                        act[idx] = act[--nact];
                                    }
                                    ++l;
                                }
                                const uint32_t credit_camp = (l < r) ? clk_camp_fn(r - 1) : 0u;
                                add_credit(credit_camp, day, sess.has_purch, sess.rev);
                            }

                            for (size_t idx = 0; idx < nact; ++idx) {
                                if (K > act[idx].enter_s) {
                                    add_assist_n(act[idx].camp, day,
                                                 static_cast<int32_t>(K - act[idx].enter_s));
                                }
                                act[idx].enter_s = 0;
                            }
                            si = day_end_idx;
                        }
                    } else {
                        size_t l = 0, r = 0;
                        const size_t nsess = sessions.size();
                        size_t si = 0;
                        while (si < nsess) {
                            const int32_t day = sessions[si].day;
                            size_t day_end_idx = si + 1;
                            while (day_end_idx < nsess && sessions[day_end_idx].day == day) {
                                ++day_end_idx;
                            }
                            const uint32_t K = static_cast<uint32_t>(day_end_idx - si);

                            for (uint32_t k = 0; k < K; ++k) {
                                const SessionInfo& sess = sessions[si + k];
                                int64_t win_lo, win_hi;
                                if constexpr (FastPack) {
                                    win_lo = sess.start_key;
                                    win_hi = sess.start_key + kClickReachMs;
                                } else {
                                    win_lo = sess.start_key - kClickReachMs;
                                    win_hi = sess.start_key;
                                }

                                if (l == r) {
                                    while (r < nclk && clk_ts_fn(r) < win_lo) ++r;
                                    l = r;
                                }
                                while (r < nclk && clk_ts_fn(r) <= win_hi) {
                                    const uint32_t c = clk_camp_fn(r);
                                    CampState& st = camp_state[c];
                                    if (st.count++ == 0) {
                                        st.set_enter_pos(k,
                                                         static_cast<uint32_t>(active_camps.size()));
                                        active_camps.push_back(c);
                                    }
                                    ++r;
                                }
                                while (l < r && clk_ts_fn(l) < win_lo) {
                                    const uint32_t c = clk_camp_fn(l);
                                    CampState& st = camp_state[c];
                                    if (--st.count == 0) {
                                        const uint32_t es = st.enter();
                                        if (k > es) {
                                            add_assist_n(c, day, static_cast<int32_t>(k - es));
                                        }
                                        const uint32_t pos = st.pos();
                                        const uint32_t last = active_camps.back();
                                        active_camps[pos] = last;
                                        camp_state[last].set_pos(pos);
                                        active_camps.pop_back();
                                        st.enter_and_pos = 0x00FFFFFFu;
                                    }
                                    ++l;
                                }
                                const uint32_t credit_camp = (l < r) ? clk_camp_fn(r - 1) : 0u;
                                add_credit(credit_camp, day, sess.has_purch, sess.rev);
                            }

                            for (uint32_t ac : active_camps) {
                                CampState& st = camp_state[ac];
                                const uint32_t es = st.enter();
                                if (K > es) {
                                    add_assist_n(ac, day, static_cast<int32_t>(K - es));
                                }
                                st.clear_enter();
                            }
                            si = day_end_idx;
                        }

                        for (uint32_t ac : active_camps) {
                            camp_state[ac].reset();
                        }
                        active_camps.clear();
                    }
                };

                for (size_t p = 0; p < num_persons; ++p) {
                    ShardEv* pev = shard_events.data() + ev_off[p];
                    const size_t nev = ev_off[p + 1] - ev_off[p];
                    if constexpr (FastPack) {
                        sort_u64_hi31(reinterpret_cast<uint64_t*>(pev), nev, radix_scratch);
                    } else {
                        std::sort(pev, pev + nev, [](const ShardEv& a, const ShardEv& b) {
                            return a.ts_ms < b.ts_ms;
                        });
                    }

                    const UserId64 uid = pid_user[p];

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

                    sessions.clear();
                    int64_t first_key = 0;
                    uint32_t first_price = 0, first_cat = 0;
                    get_ev(0, first_key, first_price, first_cat);

                    int64_t sess_start = first_key;
                    int64_t prev_ts = first_key;
                    int32_t sess_day;
                    int64_t next_day_rel = first_next_day_rel;
                    if constexpr (FastPack) {
                        sess_day = day_start;
                        while (sess_start >= next_day_rel) {
                            ++sess_day;
                            next_day_rel += 86400000LL;
                        }
                    } else {
                        sess_day = floor_div_day(sess_start);
                    }
                    uint64_t sess_rev = first_price;
                    uint8_t sess_purch = static_cast<uint8_t>(first_price > 0);

                    int32_t cur_day = sess_day;
                    int32_t day_sessions = 1;
                    int32_t day_events = 1;
                    uint64_t day_rev = first_price;
                    touched_cats.clear();
                    if (first_price > 0) {
                        touched_cats.push_back(first_cat);
                        cat_rev[first_cat] = first_price;
                    }

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
                        writer.add_row(uid, ua_ptrs, cur_day, day_sessions, day_events,
                                       static_cast<int64_t>(day_rev), has_top,
                                       has_top ? std::string_view(cat_names[best_cat])
                                               : std::string_view{});
                        agg.total_sessions += day_sessions;
                        agg.total_revenue += static_cast<int64_t>(day_rev);
                    };

                    for (size_t i = 1; i < nev; ++i) {
                        int64_t ts_key = 0;
                        uint32_t price = 0, cat = 0;
                        get_ev(i, ts_key, price, cat);

                        if (ts_key - prev_ts > kSessionGapMs) {
                            sessions.push_back(SessionInfo{sess_start, sess_rev, sess_day, sess_purch});
                            sess_start = ts_key;
                            if constexpr (FastPack) {
                                while (sess_start >= next_day_rel) {
                                    ++sess_day;
                                    next_day_rel += 86400000LL;
                                }
                            } else {
                                sess_day = floor_div_day(sess_start);
                            }
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
                    } else if constexpr (FastPack) {
                        ShardClk* pclk = shard_clicks.data() + clk_off[p];
                        if (nclk <= 65535 && num_camps <= 65535) {
                            uint64_t small_pkeys[32];
                            uint64_t* pkeys = small_pkeys;
                            if (nclk > 32) {
                                packed_clk_keys.resize(nclk);
                                pkeys = packed_clk_keys.data();
                            }
                            for (size_t ci = 0; ci < nclk; ++ci) {
                                pkeys[ci] = (static_cast<uint64_t>(pclk[ci].ts_rel) << 32) |
                                            (static_cast<uint64_t>(pclk[ci].camp_id) << 16) |
                                            static_cast<uint64_t>(ci);
                            }
                            sort_u64_hi31(pkeys, nclk, radix_scratch2);
                            size_t ci = 0;
                            while (ci + 1 < nclk) {
                                if (__builtin_expect((pkeys[ci] >> 32) == (pkeys[ci + 1] >> 32), 0)) {
                                    size_t cj = ci + 2;
                                    while (cj < nclk && (pkeys[cj] >> 32) == (pkeys[ci] >> 32)) ++cj;
                                    std::sort(pkeys + ci, pkeys + cj, [&](uint64_t a, uint64_t b) {
                                        return ClickId64::less(pclk[b & 0xFFFFu].click_id,
                                                               pclk[a & 0xFFFFu].click_id, ca_ptrs);
                                    });
                                    ci = cj;
                                } else {
                                    ++ci;
                                }
                            }
                            run_attr_window(
                                nclk,
                                [&](size_t idx) -> int64_t {
                                    return static_cast<int64_t>(pkeys[idx] >> 32);
                                },
                                [&](size_t idx) -> uint32_t {
                                    return static_cast<uint32_t>(pkeys[idx]) >> 16;
                                });
                        } else {
                            sort_shard_clicks_fast(pclk, nclk, clk_radix_scratch, ca_ptrs);
                            run_attr_window(
                                nclk,
                                [&](size_t idx) -> int64_t {
                                    return static_cast<int64_t>(pclk[idx].ts_rel);
                                },
                                [&](size_t idx) -> uint32_t { return pclk[idx].camp_id; });
                        }
                    } else {
                        ShardClk* pclk = shard_clicks.data() + clk_off[p];
                        std::sort(pclk, pclk + nclk, [&](const ShardClk& a, const ShardClk& b) {
                            if (a.ts_ms != b.ts_ms) return a.ts_ms < b.ts_ms;
                            return ClickId64::less(b.click_id, a.click_id, ca_ptrs);
                        });
                        run_attr_window(
                            nclk, [&](size_t idx) -> int64_t { return pclk[idx].ts_ms; },
                            [&](size_t idx) -> uint32_t { return pclk[idx].camp_id; });
                    }
                }
            }

            writer.finish();
        });
    }
    const auto t_phase_c = std::chrono::steady_clock::now();

    // Phase D: Reduce transposed campaign_daily across threads & write campaign_daily/part-0.parquet
    CampaignDailyParquetWriter camp_writer(
        join_path(join_path(output_dir, "campaign_daily"), "part-0.parquet"));
    int64_t camp_sessions_sum = 0;
    int64_t camp_revenue_sum = 0;
    int64_t camp_assists_sum = 0;
    int64_t non_organic_sessions_sum = 0;

    if (use_flat_camp) {
        for (int64_t d = 0; d < num_days_i64; ++d) {
            const int32_t day_val = static_cast<int32_t>(day_start + d);
            const size_t base = static_cast<size_t>(d) * num_camps;
            for (size_t c = 0; c < num_camps; ++c) {
                int32_t sess = 0, purch = 0, assist = 0;
                int64_t rev = 0;
                const size_t idx = base + c;
                for (size_t t = 0; t < ncpu; ++t) {
                    const CampCreditAgg& cr = thread_aggs[t].flat_credits[idx];
                    sess += cr.sessions;
                    purch += cr.purchasing_sessions;
                    rev += cr.revenue_cents;
                    assist += thread_aggs[t].flat_assists[idx];
                }
                if (sess > 0 || assist > 0) {
                    camp_writer.add_row(campaigns[c].id, campaigns[c].channel, day_val, sess, purch,
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
                 "done in %.1f ms (init %.1f, links %.1f, events %.1f, dedup+clk %.1f, shards %.1f, "
                 "camp %.1f) | sessions=%lld/%lld rev=%lld/%lld assists=%lld>=%lld\n",
                 ms(t0, t_end), ms(t0, t_pre_links), ms(t_pre_links, t_links),
                 ms(t_links, t_phase_a), ms(t_phase_a, t_phase_b), ms(t_phase_b, t_phase_c),
                 ms(t_phase_c, t_end), static_cast<long long>(user_sessions_sum),
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

    FastDateLut ev_date_lut;
    ev_date_lut.build(event_win_start - 86400000LL, 24, event_win_start);
    FastDateLut clk_date_lut;
    clk_date_lut.build(click_win_start - 86400000LL, 25, click_win_start);

    auto ev_paths = list_csv_files(join_path(data_dir, "events"));
    auto clk_paths = list_csv_files(join_path(data_dir, "clicks"));
    std::vector<PreadFile> ev_files(ev_paths.size());
    std::vector<PreadFile> clk_files(clk_paths.size());

    std::vector<std::string> cat_names;
    FlatStrMap<uint32_t> product_map;
    std::vector<uint8_t> fast_prod_by_num;
    uint32_t unknown_cat_id = 0;

    std::vector<CampaignInfo> campaigns;
    campaigns.push_back({"organic", "organic"});
    FlatStrMap<uint32_t> campaign_map;
    std::vector<uint32_t> fast_camp_by_num;

    auto load_products = [&]() {
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
        const bool can_fast_prod = (cat_names.size() <= 254);
        if (can_fast_prod) {
            fast_prod_by_num.reserve(rows.size() + 2);
        }
        product_map.reserve(rows.size());
        for (const auto& r : rows) {
            const uint32_t* cid_p = sorted_cat_map.find(r.cat);
            const uint32_t cid = cid_p ? *cid_p : unknown_cat_id;
            if (can_fast_prod && r.pid.size() >= 2 && r.pid[0] == 'p' &&
                static_cast<uint8_t>(r.pid[1] - '1') <= 8) {
                uint64_t num = static_cast<uint64_t>(r.pid[1] - '0');
                bool is_canon = true;
                for (size_t i = 2; i < r.pid.size(); ++i) {
                    const uint8_t d = static_cast<uint8_t>(r.pid[i] - '0');
                    if (d > 9 || num > 1000000ULL) {
                        is_canon = false;
                        break;
                    }
                    num = num * 10 + d;
                }
                if (is_canon && num <= 200000ULL) {
                    if (num >= fast_prod_by_num.size()) {
                        fast_prod_by_num.resize(num + 1, 0xFFu);
                    }
                    fast_prod_by_num[num] = static_cast<uint8_t>(cid);
                    continue;
                }
            }
            product_map.emplace(r.pid, cid);
        }
    };

    auto load_campaigns = [&]() {
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
    };

    std::atomic<size_t> total_ev_bytes{0};
    const size_t total_files = ev_paths.size() + clk_paths.size();
    if (total_files >= 8 && ncpu >= 2) {
        std::atomic<size_t> next_file{0};
        std::mutex err_mu;
        std::string err_msg;
        pool.run([&](size_t tid) {
            if (tid == 0) {
                try {
                    load_products();
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lk(err_mu);
                    if (err_msg.empty()) err_msg = e.what();
                }
            } else if (tid == 1) {
                try {
                    load_campaigns();
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lk(err_mu);
                    if (err_msg.empty()) err_msg = e.what();
                }
            }
            size_t loc_ev = 0;
            for (;;) {
                const size_t fi = next_file.fetch_add(1, std::memory_order_relaxed);
                if (fi >= total_files) break;
                try {
                    if (fi < ev_paths.size()) {
                        ev_files[fi] = open_one_pread_file(ev_paths[fi]);
                        loc_ev += ev_files[fi].size;
                    } else {
                        const size_t ci = fi - ev_paths.size();
                        clk_files[ci] = open_one_pread_file(clk_paths[ci]);
                    }
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lk(err_mu);
                    if (err_msg.empty()) err_msg = e.what();
                }
            }
            total_ev_bytes.fetch_add(loc_ev, std::memory_order_relaxed);
        });
        if (!err_msg.empty()) throw std::runtime_error(err_msg);
    } else {
        load_products();
        load_campaigns();
        size_t loc_ev = 0;
        for (size_t i = 0; i < ev_paths.size(); ++i) {
            ev_files[i] = open_one_pread_file(ev_paths[i]);
            loc_ev += ev_files[i].size;
        }
        for (size_t i = 0; i < clk_paths.size(); ++i) {
            clk_files[i] = open_one_pread_file(clk_paths[i]);
        }
        total_ev_bytes.store(loc_ev, std::memory_order_relaxed);
    }

    const bool can_fast_pack =
        !std::getenv("SS_FORCE_SLOW_PACK") && (event_win_end > event_win_start) &&
        (click_win_end > click_win_start) &&
        (static_cast<uint64_t>(event_win_end - event_win_start) <= 0x7FFFFFFEULL) &&
        (static_cast<uint64_t>(click_win_end - click_win_start) <= 0x7FFFFFFEULL) &&
        (cat_names.size() <= 254);

    if (can_fast_pack) {
        run_pipeline<true>(data_dir, output_dir, pool, event_win_start, event_win_end,
                           click_win_start, click_win_end, day_start, day_end, ev_date_lut,
                           clk_date_lut, cat_names, unknown_cat_id, fast_prod_by_num, product_map,
                           campaigns, fast_camp_by_num, campaign_map, ev_files, clk_files,
                           total_ev_bytes.load(std::memory_order_relaxed), t0);
    } else {
        run_pipeline<false>(data_dir, output_dir, pool, event_win_start, event_win_end,
                            click_win_start, click_win_end, day_start, day_end, ev_date_lut,
                            clk_date_lut, cat_names, unknown_cat_id, fast_prod_by_num, product_map,
                            campaigns, fast_camp_by_num, campaign_map, ev_files, clk_files,
                            total_ev_bytes.load(std::memory_order_relaxed), t0);
    }

    return 0;
}
