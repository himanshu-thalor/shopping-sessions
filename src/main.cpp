#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <thread>
#include <unistd.h>
#include <utility>
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

using EventId64 = FastId64<'e'>;
using UserId64 = FastId64<'u'>;
using ClickId64 = FastId64<'k'>;

template <bool FastPack>
struct Traits;

template <>
struct Traits<true> {
    static constexpr size_t kSurvEvBlockCap = 512;  // 1024 * 16B = 16 KiB
    static constexpr size_t kSurvClkBlockCap = 256;  // 512 * 24B = 12 KiB

    struct SurvivingEvent {
        UserId64 user_id;
        uint64_t ts_price_cat;  // (uint64_t(ts_rel) << 32) | (cat_id << 24) | price_cents
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
        uint64_t packed;  // (uint64_t(ts_rel) << 32) | (cat_id << 24) | price_cents
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
    static constexpr size_t kSurvEvBlockCap = 256;   // 256 * 24B = 12 KiB
    static constexpr size_t kSurvClkBlockCap = 512;  // 512 * 32B = 16 KiB

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

// Every timestamp in a bin of width (kSessionGapMs + 1) is within kSessionGapMs
// of every other timestamp in that bin; only bin extrema can affect adjacent gaps.
struct SessionBins {
    static constexpr uint32_t width = static_cast<uint32_t>(kSessionGapMs) + 1;
    static constexpr size_t capacity = 1 + 0x7FFFFFFEu / width;
    static constexpr size_t max_words = (capacity + 63) / 64;

    struct Bin {
        uint32_t first;
        uint32_t last;
        uint32_t count;
        uint32_t purchases;
    };
    struct Purchase {
        uint32_t price_cat;
        uint32_t next;
    };

    Bin bins[capacity];
    uint64_t used[max_words]{};
    std::vector<Purchase> purchases;

    void init_empty() noexcept {
        for (size_t i = 0; i < capacity; ++i) {
            bins[i] = Bin{UINT32_MAX, 0, 0, UINT32_MAX};
        }
    }

    template <typename E>
    [[gnu::always_inline]] void add(const E* events, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            const uint64_t packed = events[i].packed;
            const uint32_t ts = static_cast<uint32_t>(packed >> 32);
            const uint32_t ix = ts / width;
            Bin& bin = bins[ix];
            used[ix >> 6] |= (uint64_t(1) << (ix & 63u));
            if (ts < bin.first) bin.first = ts;
            if (ts > bin.last) bin.last = ts;
            ++bin.count;
            const uint32_t pc = static_cast<uint32_t>(packed);
            if (__builtin_expect(pc != 0, 0)) {
                purchases.push_back(Purchase{pc, bin.purchases});
                bin.purchases = static_cast<uint32_t>(purchases.size() - 1);
            }
        }
    }

    [[gnu::always_inline]] uint32_t first_ts(size_t active_words) const noexcept {
        for (size_t w = 0; w < active_words; ++w) {
            if (used[w]) return bins[(w << 6) + static_cast<size_t>(__builtin_ctzll(used[w]))].first;
        }
        return 0;
    }

    template <class Fn>
    [[gnu::always_inline]] void consume(size_t active_words, Fn&& fn) {
        for (size_t w = 0; w < active_words; ++w) {
            uint64_t bits = std::exchange(used[w], 0);
            while (bits) {
                Bin& b = bins[(w << 6) + static_cast<size_t>(__builtin_ctzll(bits))];
                fn(b);
                b = Bin{UINT32_MAX, 0, 0, UINT32_MAX};
                bits &= bits - 1;
            }
        }
    }
};

struct SessionInfo {
    int64_t start_key;
    uint64_t rev;
    int32_t day;
    uint8_t has_purch;
};

struct CampSessAssist {
    int32_t sessions = 0;
    int32_t assists = 0;
};
static_assert(sizeof(CampSessAssist) == 8);

struct CampPurchRev {
    int32_t purchasing_sessions = 0;
    int64_t revenue_cents = 0;
};
static_assert(sizeof(CampPurchRev) == 16);

struct SmallCampState {
    uint16_t count = 0;
    uint8_t enter = 0;
    uint8_t in_list = 0;
};
static_assert(sizeof(SmallCampState) == 4);

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

template <bool Fast>
struct EventKey {
    EventId64 id;
    std::conditional_t<Fast, int32_t, int64_t> ingest;
    uint32_t payload;
};
static_assert(sizeof(EventKey<true>) == 16);

struct alignas(64) DedupGroup {
    uint32_t hashes[8];
    uint32_t rows[8];
};
static_assert(sizeof(DedupGroup) == 64);

struct CampState {
    uint32_t count = 0;
    uint16_t enter = 0;
    uint8_t in_list = 0;
    uint8_t pad = 0;
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
    if (__builtin_expect(!fast_prod_by_num.empty(), 1)) {
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
                    return static_cast<uint32_t>(fast_prod_by_num[num]);
                }
                if (num <= 200000ULL) return unknown_cat_id;
            }
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

template <typename PushFn>
void process_events_chunk(bool use_avx512, const PreadView& view, uint16_t aid, OverflowArena& ea,
                          OverflowArena& ua, const OverflowArena* const* ea_ptrs,
                          const FastDateLut& ev_date_lut, int64_t event_win_start,
                          int64_t event_win_end, const std::vector<uint8_t>& fast_prod_by_num,
                          const FlatStrMap<uint32_t>& product_map, uint32_t unknown_cat_id,
                          size_t dedup_shift, PushFn&& push_ev) {
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
                uint32_t w4;
                std::memcpy(&w4, et.data(), 4);
                static constexpr uint32_t kView = 0x77656976u;  // "view"
                static constexpr uint32_t kCart = 0x74726163u;  // "cart"
                if (__builtin_expect(w4 != kView && w4 != kCart, 0)) {
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

        UserId64 uid{};
        if (__builtin_expect(valid, 1)) {
            uid = is_padded ? UserId64::make_str_padded(row[1], aid, ua)
                            : UserId64::make_str_safe(row[1], aid, ua);
        }
        push_ev(b, eid, uid, ingest_ms, ingest_rel_sec, ts_ms, price_cents, cat_id);
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
void run_pipeline(const std::string& output_dir, ThreadPool& pool,
                  int64_t event_win_start, int64_t event_win_end, int64_t click_win_start,
                  int64_t click_win_end, int32_t day_start, int32_t day_end,
                  const FastDateLut& ev_date_lut, const FastDateLut& clk_date_lut,
                  const std::vector<std::string>& cat_names, uint32_t unknown_cat_id,
                  const std::vector<uint8_t>& fast_prod_by_num, FlatStrMap<uint32_t>& product_map,
                  const std::vector<CampaignInfo>& campaigns,
                  const std::vector<uint32_t>& fast_camp_by_num,
                  FlatStrMap<uint32_t>& campaign_map, std::vector<PreadFile>& ev_files,
                  std::vector<PreadFile>& clk_files, std::vector<PreadFile>& link_files,
                  size_t total_ev_bytes, std::chrono::steady_clock::time_point t0) {
    using T = Traits<FastPack>;
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
    if (est_events >= 500000ULL) dedup_buckets = 512;

    size_t person_shards = 64;
    if (est_events >= 500000ULL) person_shards = 512;

    const unsigned dedup_shift = 64u - log2_pow2(dedup_buckets);
    const unsigned person_shift = 64u - log2_pow2(person_shards);

    std::vector<ThreadBlockArena> block_arenas(ncpu);
    std::vector<ThreadBlockArena> key_arenas(ncpu);
    std::vector<OverflowArena> event_arenas(ncpu);
    std::vector<OverflowArena> user_arenas(ncpu * 2);
    std::vector<OverflowArena> click_arenas(ncpu);

    const size_t max_link_threads = std::min<size_t>(ncpu, 4);
    LinkTable links(person_shards, max_link_threads);
    const uint16_t link_aid_base = static_cast<uint16_t>(ncpu * 2);

    std::vector<const OverflowArena*> event_arena_ptrs(ncpu);
    std::vector<const OverflowArena*> click_arena_ptrs(ncpu);
    std::vector<const OverflowArena*> all_user_arenas(ncpu * 2 + links.num_arenas());
    for (size_t t = 0; t < ncpu; ++t) {
        event_arena_ptrs[t] = &event_arenas[t];
        click_arena_ptrs[t] = &click_arenas[t];
        all_user_arenas[t] = &user_arenas[t];
        all_user_arenas[ncpu + t] = &user_arenas[ncpu + t];
    }
    for (size_t t = 0; t < links.num_arenas(); ++t) {
        all_user_arenas[link_aid_base + t] = &links.arena(t);
    }

    using Key = EventKey<FastPack>;
    constexpr int32_t indirect_ingest = 1 << 30;
    std::vector<std::vector<int64_t>> long_ingests(ncpu);
    constexpr unsigned key_shift = FastPack ? 9 : 8;
    constexpr uint32_t key_cap = 1u << key_shift;
    constexpr unsigned payload_shift = FastPack ? 9 : 8;
    static_assert((1u << payload_shift) == T::kSurvEvBlockCap);

    std::vector<std::vector<std::vector<BlockSlice<Key>>>> key_buckets(
        ncpu, std::vector<std::vector<BlockSlice<Key>>>(dedup_buckets));
    std::vector<std::vector<SurvEv*>> payload_blocks(ncpu);
    std::vector<std::vector<std::vector<BlockSlice<SurvEv>>>> ev_person_buckets(
        ncpu, std::vector<std::vector<BlockSlice<SurvEv>>>(person_shards));

    const size_t n_link_workers = links.prepare_overlapped(link_files, ncpu);
    const auto t_pre_links = std::chrono::steady_clock::now();

    // Phase A: tid 0..n_link_workers-1 build LinkTable in parallel (Stage 1+2 ~15ms, then tid
    // 1..n_link-1 immediately join events; tid 0 finishes flat UF ~20ms then joins events),
    // while tid n_link_workers..ncpu-1 scan events/*.csv from t=0!
    {
        std::atomic<size_t> next_ev_chunk{0};
        pool.run([&](size_t tid) {
            ThreadBlockArena& my_arena = block_arenas[tid];
            ThreadBlockArena& my_key_arena = key_arenas[tid];
            if (tid < n_link_workers) {
                links.run_overlapped_worker(tid, my_arena, all_user_arenas.data(), link_aid_base,
                                            use_avx512);
            }
            if (ev_chunks.empty()) return;

            OverflowArena& ea = event_arenas[tid];
            OverflowArena& ua = user_arenas[tid];
            const OverflowArena* const* ea_ptrs = event_arena_ptrs.data();
            const OverflowArena* const* ua_ptrs = all_user_arenas.data();
            auto& buckets = key_buckets[tid];
            auto& blocks = payload_blocks[tid];
            auto& payloads = ev_person_buckets[tid];
            const uint16_t aid = static_cast<uint16_t>(tid);

            RawBuf chunk_buf;
            PreadView view;

            struct KeyOut {
                Key* cur = nullptr;
                Key* end = nullptr;
            };
            struct EvOut {
                SurvEv* cur = nullptr;
                SurvEv* end = nullptr;
                uint32_t next_code = 0;
                uint32_t pad = 0;
            };
            KeyOut kout[512]{};
            EvOut eout[512]{};

            auto push_ev = [&](size_t b, EventId64 eid, UserId64 uid, int64_t ingest_ms,
                               int32_t ingest_sec, int64_t ts, uint32_t price, uint32_t cat)
                               __attribute__((always_inline)) {
                uint32_t code = UINT32_MAX;
                if (__builtin_expect(!uid.empty(), 1)) {
                    const size_t s = static_cast<size_t>(uid.hash(ua_ptrs) >> person_shift);
                    EvOut& eo = eout[s];
                    if (__builtin_expect(eo.cur == eo.end, 0)) {
                        eo.cur = alloc_bucket_block<SurvEv, T::kSurvEvBlockCap>(my_arena, eo.cur,
                                                                               eo.end, payloads[s]);
                        if (blocks.size() >= (UINT32_MAX >> payload_shift)) {
                            throw std::runtime_error("too many event blocks for one reader");
                        }
                        eo.next_code = static_cast<uint32_t>(blocks.size()) << payload_shift;
                        blocks.push_back(eo.cur);
                    }
                    code = eo.next_code++;
                    if constexpr (FastPack) {
                        const auto rel = static_cast<uint32_t>(ts - event_win_start);
                        *eo.cur++ = SurvEv{uid, (uint64_t(rel) << 32) | (cat << 24) | price};
                    } else {
                        *eo.cur++ = SurvEv{uid, ts, price, cat};
                    }
                }
                KeyOut& ko = kout[b];
                if (__builtin_expect(ko.cur == ko.end, 0)) {
                    ko.cur =
                        alloc_bucket_block<Key, key_cap>(my_key_arena, ko.cur, ko.end, buckets[b]);
                }
                if constexpr (FastPack) {
                    if (__builtin_expect(
                            ingest_sec == INT32_MIN || ingest_sec >= indirect_ingest, 0)) {
                        auto& extra = long_ingests[tid];
                        if (extra.size() >= static_cast<size_t>(indirect_ingest)) {
                            throw std::runtime_error("too many distant ingest dates");
                        }
                        ingest_sec = indirect_ingest + static_cast<int32_t>(extra.size());
                        extra.push_back(ingest_ms);
                    }
                    *ko.cur++ = Key{eid, ingest_sec, code};
                } else {
                    *ko.cur++ = Key{eid, ingest_ms, code};
                }
            };

            for (;;) {
                const size_t ci = next_ev_chunk.fetch_add(1, std::memory_order_relaxed);
                if (ci >= ev_chunks.size()) break;
                const PreadChunk& pc = ev_chunks[ci];
                if (!read_chunk_into(ev_files[pc.file_idx], pc, chunk_buf, view)) continue;
                process_events_chunk(use_avx512, view, aid, ea, ua, ea_ptrs, ev_date_lut,
                                     event_win_start, event_win_end, fast_prod_by_num, product_map,
                                     unknown_cat_id, dedup_shift, push_ev);
            }

            for (size_t b = 0; b < dedup_buckets; ++b) {
                if (kout[b].cur) {
                    Key* base = kout[b].end - key_cap;
                    buckets[b].push_back(
                        BlockSlice<Key>{base, static_cast<uint32_t>(kout[b].cur - base)});
                }
            }
            for (size_t s = 0; s < person_shards; ++s) {
                if (eout[s].cur) {
                    SurvEv* base = eout[s].end - T::kSurvEvBlockCap;
                    payloads[s].push_back(
                        BlockSlice<SurvEv>{base, static_cast<uint32_t>(eout[s].cur - base)});
                }
            }
        });
    }
    ev_chunks.clear();
    ev_files.clear();
    link_files.clear();
    product_map.clear_and_free();
    const auto t_phase_a = std::chrono::steady_clock::now();

    // Phase B: Concurrent AVX2 DedupGroup deduplication (8-ahead prefetched) + clicks scan.
    std::vector<std::vector<std::vector<BlockSlice<SurvClk>>>> clk_person_buckets(
        ncpu, std::vector<std::vector<BlockSlice<SurvClk>>>(person_shards));

    const size_t clk_starter_threads =
        clk_chunks.empty() ? 0 : (ncpu >= 8 ? 4 : (ncpu >= 6 ? 3 : (ncpu >= 4 ? 2 : 1)));

    {
        std::atomic<size_t> next_dedup_b{0};
        std::atomic<size_t> next_clk_chunk{0};

        pool.run([&](size_t tid) {
            ThreadBlockArena& my_arena = block_arenas[tid];
            const OverflowArena* const* ea_ptrs = event_arena_ptrs.data();
            const OverflowArena* const* ua_ptrs = all_user_arenas.data();
            auto& out_clk_buckets = clk_person_buckets[tid];
            OverflowArena& ua = user_arenas[ncpu + tid];
            OverflowArena& ca = click_arenas[tid];
            const uint16_t clk_uid_aid = static_cast<uint16_t>(ncpu + tid);
            const uint16_t clk_kid_aid = static_cast<uint16_t>(tid);

            auto run_dedup = [&]() {
                struct Ref {
                    BlockSlice<Key> block;
                    size_t reader;
                };
                auto earlier = [&](const Key& a, size_t ar, const Key& b, size_t br) {
                    if constexpr (FastPack) {
                        if (__builtin_expect(
                                a.ingest >= indirect_ingest || b.ingest >= indirect_ingest, 0)) {
                            auto exact = [&](int32_t k, size_t reader) {
                                return k >= indirect_ingest
                                           ? long_ingests[reader][k - indirect_ingest]
                                           : event_win_start + int64_t(k) * 1000LL;
                            };
                            return exact(a.ingest, ar) < exact(b.ingest, br);
                        }
                    }
                    return a.ingest < b.ingest;
                };

                std::vector<Ref> refs;
                std::vector<DedupGroup> table;
                auto discard = [&](const Ref& ref, uint32_t code) {
                    if (code != UINT32_MAX) {
                        auto* block = payload_blocks[ref.reader][code >> payload_shift];
                        block[code & (T::kSurvEvBlockCap - 1)].user_id.raw = 0;
                    }
                };

                for (;;) {
                    const size_t b = next_dedup_b.fetch_add(1, std::memory_order_relaxed);
                    if (b >= dedup_buckets) break;
                    refs.clear();
                    size_t total = 0;
                    for (size_t t = 0; t < ncpu; ++t) {
                        for (auto sl : key_buckets[t][b]) {
                            refs.push_back(Ref{sl, t});
                            total += sl.len;
                        }
                    }
                    if (!total) continue;
                    const size_t capacity = std::bit_ceil(total / 6 + 1);
                    table.resize(capacity);
                    std::memset(static_cast<void*>(table.data()), 0,
                                capacity * sizeof(DedupGroup));
                    const size_t mask = capacity - 1;

                    for (uint32_t bi = 0; bi < refs.size(); ++bi) {
                        const Ref& ref = refs[bi];
                        const Key* const kptr = ref.block.ptr;
                        const uint32_t klen = ref.block.len;
                        uint32_t hring[8];
                        const uint32_t pre = std::min<uint32_t>(8u, klen);
                        for (uint32_t i = 0; i < pre; ++i) {
                            hring[i] = static_cast<uint32_t>(kptr[i].id.hash(ea_ptrs));
                        }
                        for (uint32_t i = 0; i < klen; ++i) {
                            const uint32_t hash = hring[i & 7u];
                            if (i + 8u < klen) {
                                const uint32_t nh =
                                    static_cast<uint32_t>(kptr[i + 8u].id.hash(ea_ptrs));
                                hring[i & 7u] = nh;
                                __builtin_prefetch(&table[nh & mask], 1, 1);
                            }
                            const Key& key = kptr[i];
                            const uint32_t tag = hash | 1u;
                            const uint32_t code = (bi << key_shift) | i;
                            size_t g = hash & mask;
                            const __m256i vtag = _mm256_set1_epi32(static_cast<int>(tag));
                            const __m256i vzero = _mm256_setzero_si256();
                            bool found = false;
                            for (;;) {
                                DedupGroup& bucket = table[g];
                                const __m256i tags = _mm256_load_si256(
                                    reinterpret_cast<const __m256i*>(bucket.hashes));
                                unsigned same = static_cast<unsigned>(_mm256_movemask_ps(
                                    _mm256_castsi256_ps(_mm256_cmpeq_epi32(tags, vtag))));
                                while (__builtin_expect(same != 0, 0)) {
                                    const unsigned lane = static_cast<unsigned>(__builtin_ctz(same));
                                    const uint32_t old = bucket.rows[lane];
                                    const Ref& old_ref = refs[old >> key_shift];
                                    const Key& best = old_ref.block.ptr[old & (key_cap - 1)];
                                    if (EventId64::equals(key.id, best.id, ea_ptrs)) {
                                        if (earlier(key, ref.reader, best, old_ref.reader)) {
                                            discard(old_ref, best.payload);
                                            bucket.rows[lane] = code;
                                        } else {
                                            discard(ref, key.payload);
                                        }
                                        found = true;
                                        break;
                                    }
                                    same &= same - 1;
                                }
                                if (found) break;
                                const unsigned empty = static_cast<unsigned>(_mm256_movemask_ps(
                                    _mm256_castsi256_ps(_mm256_cmpeq_epi32(tags, vzero))));
                                if (__builtin_expect(empty != 0, 1)) {
                                    const unsigned lane =
                                        static_cast<unsigned>(__builtin_ctz(empty));
                                    bucket.hashes[lane] = tag;
                                    bucket.rows[lane] = code;
                                    break;
                                }
                                g = (g + 1) & mask;
                            }
                        }
                    }
                    // Key blocks live in key_arenas and are reclaimed in bulk via reset_to_bump() in C1.
                }
            };

            auto run_clicks = [&]() {
                if (clk_chunks.empty()) return;

                struct ClkOut {
                    SurvClk* cur = nullptr;
                    SurvClk* end = nullptr;
                };
                ClkOut cout[512]{};
                auto push_sclk = [&](size_t s, const SurvClk& sc) __attribute__((always_inline)) {
                    ClkOut& co = cout[s];
                    if (__builtin_expect(co.cur == co.end, 0)) {
                        co.cur = alloc_bucket_block<SurvClk, T::kSurvClkBlockCap>(
                            my_arena, co.cur, co.end, out_clk_buckets[s]);
                    }
                    *co.cur++ = sc;
                };

                auto on_valid_click = [&](std::string_view kid_sv, std::string_view uid_sv,
                                          uint32_t cid, int64_t ts_ms,
                                          auto padded) __attribute__((always_inline)) {
                    constexpr bool pad = decltype(padded)::value;
                    const UserId64 uid = pad ? UserId64::make_str_padded(uid_sv, clk_uid_aid, ua)
                                             : UserId64::make_str_safe(uid_sv, clk_uid_aid, ua);
                    const ClickId64 kid = pad ? ClickId64::make_str_padded(kid_sv, clk_kid_aid, ca)
                                              : ClickId64::make_str_safe(kid_sv, clk_kid_aid, ca);
                    const size_t s = static_cast<size_t>(uid.hash(ua_ptrs) >> person_shift);
                    if constexpr (FastPack) {
                        push_sclk(s, SurvClk{uid, kid,
                                             static_cast<uint32_t>(ts_ms - click_win_start), cid});
                    } else {
                        push_sclk(s, SurvClk{ts_ms, uid, kid, cid, 0});
                    }
                };

                RawBuf chunk_buf;
                PreadView view;
                for (;;) {
                    const size_t ci = next_clk_chunk.fetch_add(1, std::memory_order_relaxed);
                    if (ci >= clk_chunks.size()) break;
                    const PreadChunk& pc = clk_chunks[ci];
                    if (!read_chunk_into(clk_files[pc.file_idx], pc, chunk_buf, view)) continue;
                    process_clicks_chunk(use_avx512, view, clk_date_lut, click_win_start,
                                         click_win_end, fast_camp_by_num, campaign_map,
                                         on_valid_click);
                }

                for (size_t s = 0; s < person_shards; ++s) {
                    if (cout[s].cur != nullptr) {
                        SurvClk* base = cout[s].end - T::kSurvClkBlockCap;
                        const uint32_t len = static_cast<uint32_t>(cout[s].cur - base);
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

    // Phase C:
    //   C1: Group each shard with 16-ahead prefetched interning; process ALL unlinked persons
    //       immediately while 100% L2-hot, drop click-only unlinked clicks, and pack linked
    //       aliases into a single flat 32-byte AliasSlot array.
    //   C2: Process ONLY linked component roots in 16-root batched prefetched wavefronts,
    //       feeding multi-alias event spans straight into SessionBins without copying!
    const size_t num_cats = cat_names.size();
    const size_t num_camps = campaigns.size();
    const int64_t num_days_i64 =
        static_cast<int64_t>(day_end) - static_cast<int64_t>(day_start) + 1;
    const bool use_flat_camp =
        (num_days_i64 > 0) && (num_camps <= (2000000ULL / static_cast<size_t>(num_days_i64)));
    const size_t flat_camp_size =
        use_flat_camp ? (num_camps * static_cast<size_t>(num_days_i64)) : 0;

    struct ThreadAgg {
        std::vector<int32_t> flat_sess;
        std::vector<int32_t> flat_assists;
        std::vector<CampPurchRev> flat_pr;
        CampDayHashMap map;
        int64_t total_sessions = 0;
        int64_t total_revenue = 0;
    };
    std::vector<ThreadAgg> thread_aggs(ncpu);
    std::vector<std::unique_ptr<UserDailyParquetWriter>> writers(ncpu);

    const int64_t day0_off = event_win_start - static_cast<int64_t>(day_start) * 86400000LL;
    const int64_t first_next_day_rel =
        static_cast<int64_t>(day_start + 1) * 86400000LL - event_win_start;
    const size_t active_bin_words = [&]() -> size_t {
        if constexpr (!FastPack) return 0;
        const uint64_t span = static_cast<uint64_t>(event_win_end - event_win_start);
        const size_t max_bin = static_cast<size_t>((span > 0 ? span - 1 : 0) / SessionBins::width);
        return std::min(SessionBins::max_words, (max_bin >> 6) + 1);
    }();

    struct alignas(32) AliasSlot {
        ShardEv* ev_ptr;
        ShardClk* clk_ptr;
        uint32_t ev_len;
        uint32_t clk_len;
        uint32_t next;
        uint32_t pad;
    };
    static_assert(sizeof(AliasSlot) == 32);

    const uint32_t total_linked = links.total_linked();
    HugeBuffer<AliasSlot> alias_slots;
    if (total_linked > 0) {
        alias_slots.assign_zero(total_linked);
    }

    // Per-thread scratch for sessionization, sorting, and attribution (shared across C1 and C2).
    struct WorkerCtx {
        std::vector<ShardEv> merged_events;
        std::vector<ShardClk> merged_clicks, clk_radix_scratch;
        std::vector<uint64_t> radix_scratch, packed_clk_keys, radix_scratch2;
        SessionBins bins;
        std::vector<SessionInfo> sessions;
        std::vector<uint64_t> cat_rev;
        std::vector<uint32_t> touched_cats;
        std::vector<SmallCampState> small_camp_state;
        std::vector<CampState> camp_state;
        std::vector<uint32_t> camp_epoch;
        uint32_t seen_epoch = 0;
        std::vector<uint32_t> active_camps;

        void init(size_t ncats, size_t ncamps) {
            cat_rev.assign(ncats, 0);
            small_camp_state.resize(ncamps);
            camp_state.resize(ncamps);
            camp_epoch.assign(ncamps, 0);
            bins.init_empty();
        }
    };
    std::vector<WorkerCtx> ctxs(ncpu);

    auto run_person_common = [&](size_t tid, UserId64 uid, bool is_binned, int64_t first_key,
                                 ShardEv* pev, size_t nev, ShardClk* pclk, size_t nclk) {
        auto& ctx = ctxs[tid];
        auto& agg = thread_aggs[tid];
        auto& writer = *writers[tid];
        const OverflowArena* const* ua_ptrs = all_user_arenas.data();
        const OverflowArena* const* ca_ptrs = click_arena_ptrs.data();

        auto add_credit = [&](uint32_t camp_id, int32_t day, uint8_t has_purch, uint64_t rev) {
            if (use_flat_camp) {
                const size_t idx =
                    static_cast<size_t>(day - day_start) * num_camps + static_cast<size_t>(camp_id);
                ++agg.flat_sess[idx];
                if (__builtin_expect(has_purch != 0, 0)) {
                    ++agg.flat_pr[idx].purchasing_sessions;
                    agg.flat_pr[idx].revenue_cents += static_cast<int64_t>(rev);
                }
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

        auto& sessions = ctx.sessions;
        auto& cat_rev = ctx.cat_rev;
        auto& touched_cats = ctx.touched_cats;

        sessions.clear();
        int64_t sess_start = first_key;
        int64_t prev_ts = first_key;
        int32_t sess_day;
        int64_t next_day_rel = first_next_day_rel;
        if constexpr (FastPack) {
            const int64_t d_off = static_cast<int64_t>(
                static_cast<uint64_t>(sess_start + day0_off) / 86400000ULL);
            sess_day = day_start + static_cast<int32_t>(d_off);
            next_day_rel = first_next_day_rel + d_off * 86400000LL;
        } else {
            sess_day = floor_div_day(sess_start);
        }
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
            writer.add_row(uid, ua_ptrs, cur_day, day_sessions, day_events,
                           static_cast<int64_t>(day_rev), has_top,
                           has_top ? std::string_view(cat_names[best_cat]) : std::string_view{});
            agg.total_sessions += day_sessions;
            agg.total_revenue += static_cast<int64_t>(day_rev);
        };

        auto add_group = [&](int64_t ts_key, int64_t last_key, uint32_t count) {
            if (ts_key - prev_ts > kSessionGapMs) {
                sessions.push_back(SessionInfo{sess_start, sess_rev, sess_day, sess_purch});
                sess_start = ts_key;
                sess_rev = 0;
                sess_purch = 0;
                if constexpr (FastPack) {
                    if (sess_start >= next_day_rel) {
                        flush_day();
                        if (__builtin_expect(sess_start < next_day_rel + 86400000LL, 1)) {
                            ++sess_day;
                            next_day_rel += 86400000LL;
                        } else {
                            const int64_t d_off = static_cast<int64_t>(
                                static_cast<uint64_t>(sess_start + day0_off) / 86400000ULL);
                            sess_day = day_start + static_cast<int32_t>(d_off);
                            next_day_rel = first_next_day_rel + d_off * 86400000LL;
                        }
                        cur_day = sess_day;
                        day_sessions = 0;
                        day_events = 0;
                        day_rev = 0;
                    }
                } else {
                    sess_day = floor_div_day(sess_start);
                    if (sess_day != cur_day) {
                        flush_day();
                        cur_day = sess_day;
                        day_sessions = 0;
                        day_events = 0;
                        day_rev = 0;
                    }
                }
                ++day_sessions;
            }
            prev_ts = last_key;
            day_events += static_cast<int32_t>(count);
        };

        auto purchase = [&](uint32_t price, uint32_t cat) {
            sess_rev += price;
            sess_purch = 1;
            day_rev += price;
            if (cat_rev[cat] == 0) touched_cats.push_back(cat);
            cat_rev[cat] += price;
        };

        if constexpr (FastPack) {
            if (is_binned) {
                auto& bins = ctx.bins;
                bins.consume(active_bin_words, [&](const SessionBins::Bin& bin) {
                    add_group(bin.first, bin.last, bin.count);
                    for (uint32_t i = bin.purchases; i != UINT32_MAX; i = bins.purchases[i].next) {
                        const uint32_t pc = bins.purchases[i].price_cat;
                        purchase(pc & 0xFFFFFFu, pc >> 24);
                    }
                });
            } else {
                const uint32_t pc0 = static_cast<uint32_t>(pev[0].packed);
                day_events = 1;
                if (__builtin_expect(pc0 != 0, 0)) purchase(pc0 & 0xFFFFFFu, pc0 >> 24);
                for (size_t i = 1; i < nev; ++i) {
                    const uint64_t pk = pev[i].packed;
                    const uint32_t ts = static_cast<uint32_t>(pk >> 32);
                    const uint32_t pc = static_cast<uint32_t>(pk);
                    add_group(ts, ts, 1);
                    if (__builtin_expect(pc != 0, 0)) purchase(pc & 0xFFFFFFu, pc >> 24);
                }
            }
        } else {
            day_events = 1;
            if (pev[0].price_cents) purchase(pev[0].price_cents, pev[0].cat_id);
            for (size_t i = 1; i < nev; ++i) {
                const auto& e = pev[i];
                add_group(e.ts_ms, e.ts_ms, 1);
                if (e.price_cents) purchase(e.price_cents, e.cat_id);
            }
        }
        sessions.push_back(SessionInfo{sess_start, sess_rev, sess_day, sess_purch});
        flush_day();

        if (nclk == 0) {
            for (const auto& sess : sessions) {
                add_credit(0, sess.day, sess.has_purch, sess.rev);
            }
            return;
        }

        // Single-session fast path: zero sort, O(1) epoch array (zero cleanup loop!).
        if (sessions.size() == 1) {
            const auto& sess = sessions[0];
            const int64_t lo = FastPack ? sess.start_key : (sess.start_key - kClickReachMs);
            const int64_t hi = lo + kClickReachMs;
            int64_t best_ts = INT64_MIN;
            ClickId64 best_id{};
            uint32_t winner = 0;
            if (++ctx.seen_epoch == 0) {
                std::fill(ctx.camp_epoch.begin(), ctx.camp_epoch.end(), 0);
                ctx.seen_epoch = 1;
            }
            const uint32_t ep = ctx.seen_epoch;
            uint32_t* const cep = ctx.camp_epoch.data();
            for (size_t i = 0; i < nclk; ++i) {
                const auto& click = pclk[i];
                int64_t ts;
                if constexpr (FastPack) ts = click.ts_rel;
                else ts = click.ts_ms;
                if (ts < lo || ts > hi) continue;
                if (ts > best_ts ||
                    (ts == best_ts && ClickId64::less(click.click_id, best_id, ca_ptrs))) {
                    best_ts = ts;
                    best_id = click.click_id;
                    winner = click.camp_id;
                }
                const uint32_t c = click.camp_id;
                if (cep[c] != ep) {
                    cep[c] = ep;
                    add_assist_n(c, sess.day, 1);
                }
            }
            add_credit(winner, sess.day, sess.has_purch, sess.rev);
            return;
        }

        // Single-click fast path: zero sort, zero sliding window!
        if (nclk == 1) {
            int64_t cts;
            if constexpr (FastPack) cts = pclk[0].ts_rel;
            else cts = pclk[0].ts_ms;
            const uint32_t cid = pclk[0].camp_id;
            for (const auto& sess : sessions) {
                const int64_t lo = FastPack ? sess.start_key : (sess.start_key - kClickReachMs);
                if (cts >= lo && cts <= lo + kClickReachMs) {
                    add_credit(cid, sess.day, sess.has_purch, sess.rev);
                    add_assist_n(cid, sess.day, 1);
                } else {
                    add_credit(0, sess.day, sess.has_purch, sess.rev);
                }
            }
            return;
        }

        // General multi-session, multi-click attribution using O(1) L1-resident CampState array.
        auto run_attr_window = [&](auto* cstate, size_t mclk, auto&& clk_ts_fn,
                                   auto&& clk_camp_fn) {
            auto& active_camps = ctx.active_camps;
            size_t l = 0, r = 0;
            const size_t nsess = sessions.size();
            size_t si = 0;
            if (__builtin_expect(use_flat_camp, 1)) {
                int32_t* const base_sess = agg.flat_sess.data();
                int32_t* const base_assists = agg.flat_assists.data();
                CampPurchRev* const base_pr = agg.flat_pr.data();
                while (si < nsess) {
                    const int32_t day = sessions[si].day;
                    size_t day_end_idx = si + 1;
                    while (day_end_idx < nsess && sessions[day_end_idx].day == day) {
                        ++day_end_idx;
                    }
                    const uint32_t K = static_cast<uint32_t>(day_end_idx - si);
                    const size_t day_off = static_cast<size_t>(day - day_start) * num_camps;
                    int32_t* const day_sess = base_sess + day_off;
                    int32_t* const day_assists = base_assists + day_off;
                    CampPurchRev* const day_pr = base_pr + day_off;

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
                            while (r < mclk && clk_ts_fn(r) < win_lo) ++r;
                            l = r;
                        }
                        while (r < mclk && clk_ts_fn(r) <= win_hi) {
                            const uint32_t c = clk_camp_fn(r);
                            auto& st = cstate[c];
                            if (st.count++ == 0) {
                                st.enter = static_cast<decltype(st.enter)>(k);
                                if (!st.in_list) {
                                    st.in_list = 1;
                                    active_camps.push_back(c);
                                }
                            }
                            ++r;
                        }
                        while (l < r && clk_ts_fn(l) < win_lo) {
                            const uint32_t c = clk_camp_fn(l);
                            auto& st = cstate[c];
                            if (--st.count == 0) {
                                if (k > st.enter) {
                                    day_assists[c] += static_cast<int32_t>(k - st.enter);
                                }
                            }
                            ++l;
                        }
                        const uint32_t credit_camp = (l < r) ? clk_camp_fn(r - 1) : 0u;
                        ++day_sess[credit_camp];
                        if (__builtin_expect(sess.has_purch != 0, 0)) {
                            ++day_pr[credit_camp].purchasing_sessions;
                            day_pr[credit_camp].revenue_cents += static_cast<int64_t>(sess.rev);
                        }
                    }

                    const size_t nact = active_camps.size();
                    size_t aw = 0;
                    for (size_t ai = 0; ai < nact; ++ai) {
                        if (ai + 4 < nact) {
                            __builtin_prefetch(&day_assists[active_camps[ai + 4]], 1, 1);
                        }
                        const uint32_t ac = active_camps[ai];
                        auto& st = cstate[ac];
                        if (st.count > 0) {
                            day_assists[ac] += static_cast<int32_t>(K - st.enter);
                            st.enter = 0;
                            active_camps[aw++] = ac;
                        } else {
                            st.in_list = 0;
                        }
                    }
                    active_camps.resize(aw);
                    si = day_end_idx;
                }
            } else {
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
                            while (r < mclk && clk_ts_fn(r) < win_lo) ++r;
                            l = r;
                        }
                        while (r < mclk && clk_ts_fn(r) <= win_hi) {
                            const uint32_t c = clk_camp_fn(r);
                            auto& st = cstate[c];
                            if (st.count++ == 0) {
                                st.enter = static_cast<decltype(st.enter)>(k);
                                if (!st.in_list) {
                                    st.in_list = 1;
                                    active_camps.push_back(c);
                                }
                            }
                            ++r;
                        }
                        while (l < r && clk_ts_fn(l) < win_lo) {
                            const uint32_t c = clk_camp_fn(l);
                            auto& st = cstate[c];
                            if (--st.count == 0) {
                                if (k > st.enter) {
                                    add_assist_n(c, day, static_cast<int32_t>(k - st.enter));
                                }
                            }
                            ++l;
                        }
                        const uint32_t credit_camp = (l < r) ? clk_camp_fn(r - 1) : 0u;
                        add_credit(credit_camp, day, sess.has_purch, sess.rev);
                    }

                    size_t aw = 0;
                    for (size_t ai = 0; ai < active_camps.size(); ++ai) {
                        const uint32_t ac = active_camps[ai];
                        auto& st = cstate[ac];
                        if (st.count > 0) {
                            if (K > st.enter) {
                                add_assist_n(ac, day, static_cast<int32_t>(K - st.enter));
                            }
                            st.enter = 0;
                            active_camps[aw++] = ac;
                        } else {
                            st.in_list = 0;
                        }
                    }
                    active_camps.resize(aw);
                    si = day_end_idx;
                }
            }

            for (uint32_t ac : active_camps) {
                cstate[ac].count = 0;
                cstate[ac].in_list = 0;
            }
            active_camps.clear();
        };

        if constexpr (FastPack) {
            if (nclk <= 65535 && num_camps <= 65535) {
                const uint32_t min_ts = static_cast<uint32_t>(sessions.front().start_key);
                const uint32_t max_ts =
                    static_cast<uint32_t>(sessions.back().start_key + kClickReachMs);
                uint64_t small_pkeys[32];
                uint64_t* pkeys = small_pkeys;
                if (nclk > 32) {
                    ctx.packed_clk_keys.resize(nclk);
                    pkeys = ctx.packed_clk_keys.data();
                }
                size_t mclk = 0;
                for (size_t ci = 0; ci < nclk; ++ci) {
                    const uint32_t ts = pclk[ci].ts_rel;
                    if (ts >= min_ts && ts <= max_ts) {
                        pkeys[mclk++] = (static_cast<uint64_t>(ts) << 32) |
                                        (static_cast<uint64_t>(pclk[ci].camp_id) << 16) |
                                        static_cast<uint64_t>(ci);
                    }
                }
                if (mclk == 0) {
                    for (const auto& sess : sessions) {
                        add_credit(0, sess.day, sess.has_purch, sess.rev);
                    }
                    return;
                }
                if (mclk == 1) {
                    const int64_t cts = static_cast<int64_t>(pkeys[0] >> 32);
                    const uint32_t cid = static_cast<uint32_t>(pkeys[0]) >> 16;
                    for (const auto& sess : sessions) {
                        const int64_t lo = sess.start_key;
                        if (cts >= lo && cts <= lo + kClickReachMs) {
                            add_credit(cid, sess.day, sess.has_purch, sess.rev);
                            add_assist_n(cid, sess.day, 1);
                        } else {
                            add_credit(0, sess.day, sess.has_purch, sess.rev);
                        }
                    }
                    return;
                }
                sort_u64_hi31(pkeys, mclk, ctx.radix_scratch2);
                size_t ci = 0;
                while (ci + 1 < mclk) {
                    if (__builtin_expect((pkeys[ci] >> 32) == (pkeys[ci + 1] >> 32), 0)) {
                        size_t cj = ci + 2;
                        while (cj < mclk && (pkeys[cj] >> 32) == (pkeys[ci] >> 32)) ++cj;
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
                    ctx.small_camp_state.data(), mclk,
                    [&](size_t idx) -> int64_t { return static_cast<int64_t>(pkeys[idx] >> 32); },
                    [&](size_t idx) -> uint32_t {
                        return static_cast<uint32_t>(pkeys[idx]) >> 16;
                    });
            } else {
                sort_shard_clicks_fast(pclk, nclk, ctx.clk_radix_scratch, ca_ptrs);
                run_attr_window(
                    ctx.camp_state.data(), nclk,
                    [&](size_t idx) -> int64_t { return static_cast<int64_t>(pclk[idx].ts_rel); },
                    [&](size_t idx) -> uint32_t { return pclk[idx].camp_id; });
            }
        } else {
            std::sort(pclk, pclk + nclk, [&](const ShardClk& a, const ShardClk& b) {
                if (a.ts_ms != b.ts_ms) return a.ts_ms < b.ts_ms;
                return ClickId64::less(b.click_id, a.click_id, ca_ptrs);
            });
            run_attr_window(
                ctx.camp_state.data(), nclk, [&](size_t idx) -> int64_t { return pclk[idx].ts_ms; },
                [&](size_t idx) -> uint32_t { return pclk[idx].camp_id; });
        }
    };

    auto process_single_span = [&](size_t tid, UserId64 uid, ShardEv* pev, size_t nev,
                                   ShardClk* pclk, size_t nclk) {
        auto& ctx = ctxs[tid];
        if constexpr (FastPack) {
            if (nev >= 96) {
                ctx.bins.purchases.clear();
                ctx.bins.add(pev, nev);
                const int64_t first_key = ctx.bins.first_ts(active_bin_words);
                run_person_common(tid, uid, true, first_key, nullptr, nev, pclk, nclk);
            } else {
                sort_u64_hi31(reinterpret_cast<uint64_t*>(pev), nev, ctx.radix_scratch);
                const int64_t first_key = static_cast<int64_t>(pev[0].packed >> 32);
                run_person_common(tid, uid, false, first_key, pev, nev, pclk, nclk);
            }
        } else {
            std::sort(pev, pev + nev,
                      [](const ShardEv& a, const ShardEv& b) { return a.ts_ms < b.ts_ms; });
            run_person_common(tid, uid, false, pev[0].ts_ms, pev, nev, pclk, nclk);
        }
    };

    // Pass C1: Zero-fault key_arenas reuse + branchless unified scatter + immediate unlinked!
    {
        std::atomic<size_t> next_shard{0};
        pool.run([&](size_t tid) {
            if (use_flat_camp) {
                thread_aggs[tid].flat_sess.resize(flat_camp_size);
                thread_aggs[tid].flat_assists.resize(flat_camp_size);
                thread_aggs[tid].flat_pr.resize(flat_camp_size);
            }
            writers[tid] = std::make_unique<UserDailyParquetWriter>(join_path(
                join_path(output_dir, "user_daily"), "part-" + std::to_string(tid) + ".parquet"));
            ctxs[tid].init(num_cats, num_camps);

            const auto arenas = all_user_arenas.data();
            ThreadBlockArena& my_arena = block_arenas[tid];
            ThreadBlockArena& slab_arena = key_arenas[tid];
            slab_arena.reset_to_bump();
            AliasSlot* const slots_out = alias_slots.data();

            std::vector<uint32_t> ev_off, clk_off, cur;

            for (;;) {
                const size_t s = next_shard.fetch_add(1, std::memory_order_relaxed);
                if (s >= person_shards) break;

                auto& part = links.part(s);
                const uint32_t num_linked_s = part.num_linked;
                const uint32_t base_lid = part.linked_base;

                size_t total_ev = 0;
                for (size_t w = 0; w < ncpu; ++w) {
                    for (const auto& sl : ev_person_buckets[w][s]) total_ev += sl.len;
                }

                if (total_ev == 0 && num_linked_s == 0) {
                    for (size_t w = 0; w < ncpu; ++w) {
                        for (auto sl : clk_person_buckets[w][s]) my_arena.recycle(sl.ptr);
                        std::vector<BlockSlice<SurvClk>>().swap(clk_person_buckets[w][s]);
                    }
                    part.release_table();
                    continue;
                }

                if (total_ev > 0) {
                    if (__builtin_expect(part.table.empty(), 0)) part.reserve(total_ev / 32 + 64, arenas);
                    size_t mask = part.table.size() - 1;
                    for (size_t w = 0; w < ncpu; ++w) {
                        for (auto sl : ev_person_buckets[w][s]) {
                            SurvEv* const ptr = sl.ptr;
                            const uint32_t len = sl.len;
                            for (uint32_t i = 0; i < len; ++i) {
                                if (i + 16 < len) {
                                    const UserId64 nu = ptr[i + 16].user_id;
                                    if (!nu.empty()) {
                                        __builtin_prefetch(&part.table[nu.hash(arenas) & mask], 1,
                                                           1);
                                    }
                                }
                                UserId64& uid = ptr[i].user_id;
                                if (__builtin_expect(uid.empty(), 0)) {
                                    uid.raw = UINT64_MAX;
                                    continue;
                                }
                                auto& entry = part.intern(uid, uid.hash(arenas), arenas);
                                mask = part.table.size() - 1;
                                ++entry.count;
                                uid.raw = entry.local;
                            }
                        }
                    }
                }

                const uint32_t num_total_s = static_cast<uint32_t>(part.names.size());
                ev_off.assign(num_total_s + 1, 0);
                if (total_ev > 0) {
                    for (auto& e : part.table) {
                        if (e.key.empty()) continue;
                        ev_off[e.local + 1] = std::exchange(e.count, 0);
                    }
                    for (uint32_t i = 0; i < num_total_s; ++i) ev_off[i + 1] += ev_off[i];
                }

                // Count clicks ONLY for existing nodes (in-cache-line entry.count!).
                const size_t mask = part.table.empty() ? 0 : (part.table.size() - 1);
                bool any_clk = false;
                if (!part.table.empty()) {
                    for (size_t w = 0; w < ncpu; ++w) {
                        for (auto sl : clk_person_buckets[w][s]) {
                            SurvClk* const ptr = sl.ptr;
                            const uint32_t len = sl.len;
                            for (uint32_t i = 0; i < len; ++i) {
                                if (i + 16 < len) {
                                    const uint64_t nh = ptr[i + 16].user_id.hash(arenas);
                                    __builtin_prefetch(&part.table[nh & mask], 1, 1);
                                }
                                UserId64& uid = ptr[i].user_id;
                                auto* ep = part.find_existing(uid, uid.hash(arenas), arenas);
                                if (!ep) {
                                    uid.raw = UINT64_MAX;
                                } else {
                                    ++ep->count;
                                    uid.raw = ep->local;
                                    any_clk = true;
                                }
                            }
                        }
                    }
                }

                clk_off.assign(num_total_s + 1, 0);
                if (any_clk) {
                    for (auto& e : part.table) {
                        if (e.key.empty()) continue;
                        clk_off[e.local + 1] = e.count;
                    }
                    for (uint32_t i = 0; i < num_total_s; ++i) clk_off[i + 1] += clk_off[i];
                }

                // Branchless unified scatter into already-faulted 2-MiB THP key_arenas slabs!
                const uint32_t tot_ev_all = ev_off[num_total_s];
                ShardEv* const ev_dst = slab_arena.bump_alloc<ShardEv>(tot_ev_all);
                cur.assign(ev_off.begin(), ev_off.begin() + num_total_s);
                for (size_t w = 0; w < ncpu; ++w) {
                    for (auto sl : ev_person_buckets[w][s]) {
                        const SurvEv* const ptr = sl.ptr;
                        const uint32_t len = sl.len;
                        for (uint32_t i = 0; i < len; ++i) {
                            const uint64_t loc = ptr[i].user_id.raw;
                            if (__builtin_expect(loc == UINT64_MAX, 0)) continue;
                            if constexpr (FastPack) {
                                ev_dst[cur[loc]++] = ShardEv{ptr[i].ts_price_cat};
                            } else {
                                ev_dst[cur[loc]++] =
                                    ShardEv{ptr[i].ts_ms, ptr[i].price_cents, ptr[i].cat_id};
                            }
                        }
                        my_arena.recycle(sl.ptr);
                    }
                    std::vector<BlockSlice<SurvEv>>().swap(ev_person_buckets[w][s]);
                }

                const uint32_t tot_clk_all = clk_off[num_total_s];
                ShardClk* const clk_dst = slab_arena.bump_alloc<ShardClk>(tot_clk_all);
                cur.assign(clk_off.begin(), clk_off.begin() + num_total_s);
                for (size_t w = 0; w < ncpu; ++w) {
                    for (auto sl : clk_person_buckets[w][s]) {
                        if (any_clk) {
                            const SurvClk* const ptr = sl.ptr;
                            const uint32_t len = sl.len;
                            for (uint32_t i = 0; i < len; ++i) {
                                const uint64_t loc = ptr[i].user_id.raw;
                                if (loc == UINT64_MAX) continue;
                                if constexpr (FastPack) {
                                    clk_dst[cur[loc]++] =
                                        ShardClk{ptr[i].ts_rel, ptr[i].camp_id, ptr[i].click_id};
                                } else {
                                    clk_dst[cur[loc]++] =
                                        ShardClk{ptr[i].ts_ms, ptr[i].click_id, ptr[i].camp_id, 0};
                                }
                            }
                        }
                        my_arena.recycle(sl.ptr);
                    }
                    std::vector<BlockSlice<SurvClk>>().swap(clk_person_buckets[w][s]);
                }

                // Publish dense 32-byte AliasSlots for linked nodes in this shard.
                for (uint32_t i = 0; i < num_linked_s; ++i) {
                    const uint32_t lid = base_lid + i;
                    slots_out[lid] = AliasSlot{
                        ev_dst + ev_off[i],
                        clk_dst + clk_off[i],
                        ev_off[i + 1] - ev_off[i],
                        clk_off[i + 1] - clk_off[i],
                        links.next_alias(lid),
                        0,
                    };
                }

                // Process ALL unlinked persons immediately while L2-hot!
                for (uint32_t u = num_linked_s; u < num_total_s; ++u) {
                    const size_t nev = ev_off[u + 1] - ev_off[u];
                    if (nev == 0) continue;
                    const size_t nclk = clk_off[u + 1] - clk_off[u];
                    process_single_span(tid, part.names[u], ev_dst + ev_off[u], nev,
                                        clk_dst + clk_off[u], nclk);
                }

                part.release_table();
            }

            if (links.linked_roots().empty()) {
                writers[tid]->finish();
            }
        });
    }

    // Pass C2: Process ONLY linked component roots in 16-root prefetched wavefront batches!
    const auto& roots = links.linked_roots();
    if (!roots.empty()) {
        std::atomic<size_t> next_root{0};
        pool.run([&](size_t tid) {
            auto& ctx = ctxs[tid];
            const AliasSlot* const slots = alias_slots.data();
            const size_t nroots = roots.size();

            struct RootWork {
                UserId64 name;
                size_t nev;
                size_t nclk;
                uint32_t span_begin;
                uint32_t span_end;
                uint32_t cur_lid;
            };
            struct SpanRef {
                ShardEv* ev_ptr;
                ShardClk* clk_ptr;
                uint32_t ev_len;
                uint32_t clk_len;
            };
            static constexpr size_t kBatch = 16;
            static constexpr size_t kChunk = 256;
            std::vector<SpanRef> batch_spans;
            batch_spans.reserve(128);
            RootWork rwork[kBatch];

            for (;;) {
                const size_t r0 = next_root.fetch_add(kChunk, std::memory_order_relaxed);
                if (r0 >= nroots) break;
                const size_t r_lim = std::min(nroots, r0 + kChunk);

                for (size_t b0 = r0; b0 < r_lim; b0 += kBatch) {
                    const size_t B = std::min(kBatch, r_lim - b0);
                    // Wavefront prefetch initial root AliasSlots.
                    for (size_t j = 0; j < B; ++j) {
                        const uint32_t rlid = roots[b0 + j].root_lid;
                        __builtin_prefetch(&slots[rlid], 0, 1);
                        rwork[j] = RootWork{roots[b0 + j].name, 0, 0, 0, 0, rlid};
                    }

                    // Collect spans per root while prefetching next alias slots in lockstep.
                    batch_spans.clear();
                    for (size_t j = 0; j < B; ++j) {
                        if (j + 2 < B) {
                            const uint32_t nl = slots[rwork[j + 2].cur_lid].next;
                            if (nl != LinkTable::kNone) __builtin_prefetch(&slots[nl], 0, 1);
                        }
                        rwork[j].span_begin = static_cast<uint32_t>(batch_spans.size());
                        uint32_t lid = rwork[j].cur_lid;
                        size_t nev = 0, nclk = 0;
                        while (lid != LinkTable::kNone) {
                            const AliasSlot& sl = slots[lid];
                            const uint32_t nxt = sl.next;
                            if (nxt != LinkTable::kNone) __builtin_prefetch(&slots[nxt], 0, 1);
                            if (sl.ev_len || sl.clk_len) {
                                batch_spans.push_back(
                                    SpanRef{sl.ev_ptr, sl.clk_ptr, sl.ev_len, sl.clk_len});
                                nev += sl.ev_len;
                                nclk += sl.clk_len;
                            }
                            lid = nxt;
                        }
                        rwork[j].nev = nev;
                        rwork[j].nclk = nclk;
                        rwork[j].span_end = static_cast<uint32_t>(batch_spans.size());
                    }

                    // Prefetch payload cache lines across the entire batch!
                    for (size_t j = 0; j < B; ++j) {
                        if (rwork[j].nev == 0) continue;
                        for (uint32_t si = rwork[j].span_begin; si < rwork[j].span_end; ++si) {
                            const SpanRef& sp = batch_spans[si];
                            if (sp.ev_len) __builtin_prefetch(sp.ev_ptr, 0, 1);
                            if (sp.clk_len) __builtin_prefetch(sp.clk_ptr, 0, 1);
                        }
                    }

                    // Process each root in the batch.
                    for (size_t j = 0; j < B; ++j) {
                        const size_t nev = rwork[j].nev;
                        if (nev == 0) continue;
                        const size_t nclk = rwork[j].nclk;
                        const uint32_t sb = rwork[j].span_begin;
                        const uint32_t se = rwork[j].span_end;

                        // Gather clicks (zero copy when a single span holds all clicks!).
                        ShardClk* pclk = nullptr;
                        if (nclk > 0) {
                            size_t wclk = 0;
                            for (uint32_t si = sb; si < se; ++si) {
                                const SpanRef& sp = batch_spans[si];
                                if (!sp.clk_len) continue;
                                if (sp.clk_len == nclk) {
                                    pclk = sp.clk_ptr;
                                    break;
                                }
                                if (wclk == 0) {
                                    ctx.merged_clicks.resize(nclk);
                                    pclk = ctx.merged_clicks.data();
                                }
                                std::memcpy(pclk + wclk, sp.clk_ptr, sp.clk_len * sizeof(ShardClk));
                                wclk += sp.clk_len;
                            }
                        }

                        if constexpr (FastPack) {
                            if (nev >= 96) {
                                // Zero-copy, zero-sort direct streaming into SessionBins!
                                ctx.bins.purchases.clear();
                                for (uint32_t si = sb; si < se; ++si) {
                                    const SpanRef& sp = batch_spans[si];
                                    if (!sp.ev_len) continue;
                                    ctx.bins.add(sp.ev_ptr, sp.ev_len);
                                }
                                const uint32_t first_ts = ctx.bins.first_ts(active_bin_words);
                                run_person_common(tid, rwork[j].name, true,
                                                  static_cast<int64_t>(first_ts), nullptr, nev,
                                                  pclk, nclk);
                                continue;
                            }
                        }

                        ShardEv* pev = nullptr;
                        size_t wev = 0;
                        for (uint32_t si = sb; si < se; ++si) {
                            const SpanRef& sp = batch_spans[si];
                            if (!sp.ev_len) continue;
                            if (sp.ev_len == nev) {
                                pev = sp.ev_ptr;
                                break;
                            }
                            if (wev == 0) {
                                ctx.merged_events.resize(nev);
                                pev = ctx.merged_events.data();
                            }
                            std::memcpy(pev + wev, sp.ev_ptr, sp.ev_len * sizeof(ShardEv));
                            wev += sp.ev_len;
                        }
                        process_single_span(tid, rwork[j].name, pev, nev, pclk, nclk);
                    }
                }
            }

            writers[tid]->finish();
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
                    const CampPurchRev& pr = thread_aggs[t].flat_pr[idx];
                    sess += thread_aggs[t].flat_sess[idx];
                    assist += thread_aggs[t].flat_assists[idx];
                    purch += pr.purchasing_sessions;
                    rev += pr.revenue_cents;
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
                 "done in %.1f ms (init %.1f, events+links %.1f, dedup+clk %.1f, shards %.1f, "
                 "camp %.1f) | sessions=%lld/%lld rev=%lld/%lld assists=%lld>=%lld\n",
                 ms(t0, t_end), ms(t0, t_pre_links), ms(t_pre_links, t_phase_a),
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

    // Open all input CSV descriptors sequentially BEFORE spawning worker threads (0.2 ms vs 25 ms!).
    size_t total_ev_bytes = 0;
    auto ev_files = open_all_pread_files_seq(join_path(data_dir, "events"), &total_ev_bytes);
    auto clk_files = open_all_pread_files_seq(join_path(data_dir, "clicks"));
    auto link_files = open_all_pread_files_seq(join_path(data_dir, "links"));

    ThreadPool pool(ncpu);

    std::vector<std::string> cat_names;
    FlatStrMap<uint32_t> product_map;
    std::vector<uint8_t> fast_prod_by_num;
    uint32_t unknown_cat_id = 0;

    std::vector<CampaignInfo> campaigns;
    campaigns.push_back({"organic", "organic"});
    FlatStrMap<uint32_t> campaign_map;
    std::vector<uint32_t> fast_camp_by_num;

    // Fast 1-pass products.csv loader with 32-entry inline category cache + pre-sorted cat_names.
    auto load_products = [&]() {
        auto pf = read_whole_file(join_path(data_dir, "products.csv"));
        FlatStrMap<uint32_t> categories;
        categories.emplace("unknown", 0);
        cat_names.emplace_back("unknown");

        struct CatCache {
            std::string_view sv;
            uint32_t id;
        };
        CatCache cache[32]{};
        size_t ncache = 1;
        cache[0] = CatCache{"unknown", 0};

        std::vector<uint32_t> raw_fast;
        raw_fast.reserve(std::min<size_t>(200001, pf.size() / 10 + 1));
        struct OvProd {
            std::string_view id;
            uint32_t raw_cat;
        };
        std::vector<OvProd> ov_prods;

        if (!pf.empty()) {
            const char* p = static_cast<const char*>(std::memchr(pf.data(), '\n', pf.size()));
            if (p) {
                ++p;
                const char* const end = pf.data() + pf.size();
                std::string_view f[2];
                while (p < end) {
                    const char* nl =
                        static_cast<const char*>(std::memchr(p, '\n', static_cast<size_t>(end - p)));
                    const char* line_end = nl ? nl : end;
                    if (split_csv_line<2>(p, static_cast<size_t>(line_end - p), f) &&
                        !f[0].empty()) {
                        uint32_t cid = UINT32_MAX;
                        const std::string_view c_sv = f[1];
                        for (size_t k = 0; k < ncache; ++k) {
                            if (cache[k].sv.size() == c_sv.size() &&
                                (c_sv.empty() ||
                                 std::memcmp(cache[k].sv.data(), c_sv.data(), c_sv.size()) == 0)) {
                                cid = cache[k].id;
                                break;
                            }
                        }
                        if (__builtin_expect(cid == UINT32_MAX, 0)) {
                            const auto [ptr, added] =
                                categories.emplace(c_sv, static_cast<uint32_t>(cat_names.size()));
                            cid = *ptr;
                            if (added) {
                                cat_names.emplace_back(c_sv);
                                if (ncache < 32) cache[ncache++] = CatCache{c_sv, cid};
                            }
                        }

                        const auto id = f[0];
                        uint32_t n = 0;
                        bool indexed = id.size() >= 2 && id.size() <= 7 && id[0] == 'p' &&
                                       static_cast<unsigned char>(id[1] - '1') <= 8;
                        for (size_t j = 1; indexed && j < id.size(); ++j) {
                            const unsigned char digit = static_cast<unsigned char>(id[j] - '0');
                            indexed = digit <= 9;
                            n = n * 10 + digit;
                        }
                        if (__builtin_expect(indexed && n <= 200000u, 1)) {
                            if (n >= raw_fast.size()) raw_fast.resize(n + 1, UINT32_MAX);
                            raw_fast[n] = cid;
                        } else {
                            ov_prods.push_back(OvProd{id, cid});
                        }
                    }
                    p = nl ? nl + 1 : end;
                }
            }
        }

        // Pre-sort cat_names so Phase C tie-breaks with integer (c < best_cat)!
        const size_t ncats = cat_names.size();
        std::vector<uint32_t> order(ncats);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return byte_less(cat_names[a], cat_names[b]);
        });
        std::vector<std::string> sorted_cats(ncats);
        std::vector<uint32_t> remap(ncats);
        for (uint32_t new_id = 0; new_id < ncats; ++new_id) {
            sorted_cats[new_id] = std::move(cat_names[order[new_id]]);
            remap[order[new_id]] = new_id;
        }
        cat_names.swap(sorted_cats);
        unknown_cat_id = remap[0];

        if (ncats <= 254) {
            fast_prod_by_num.resize(raw_fast.size(), static_cast<uint8_t>(unknown_cat_id));
            for (size_t i = 0; i < raw_fast.size(); ++i) {
                if (raw_fast[i] != UINT32_MAX) {
                    fast_prod_by_num[i] = static_cast<uint8_t>(remap[raw_fast[i]]);
                }
            }
            if (!ov_prods.empty()) {
                product_map.reserve(ov_prods.size());
                for (const auto& ov : ov_prods) product_map.emplace(ov.id, remap[ov.raw_cat]);
            }
        } else {
            product_map.reserve(raw_fast.size() + ov_prods.size());
            char buf[16];
            for (size_t i = 0; i < raw_fast.size(); ++i) {
                if (raw_fast[i] != UINT32_MAX) {
                    int n = std::snprintf(buf, sizeof(buf), "p%zu", i);
                    product_map.emplace(std::string_view(buf, static_cast<size_t>(n)),
                                        remap[raw_fast[i]]);
                }
            }
            for (const auto& ov : ov_prods) product_map.emplace(ov.id, remap[ov.raw_cat]);
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

    if (ncpu >= 2) {
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
        });
        if (!err_msg.empty()) throw std::runtime_error(err_msg);
    } else {
        load_products();
        load_campaigns();
    }

    const bool can_fast_pack =
        !std::getenv("SS_FORCE_SLOW_PACK") && (event_win_end > event_win_start) &&
        (click_win_end > click_win_start) &&
        (static_cast<uint64_t>(event_win_end - event_win_start) <= 0x7FFFFFFEULL) &&
        (static_cast<uint64_t>(click_win_end - click_win_start) <= 0x7FFFFFFEULL) &&
        (cat_names.size() <= 254);

    if (can_fast_pack) {
        run_pipeline<true>(output_dir, pool, event_win_start, event_win_end, click_win_start,
                           click_win_end, day_start, day_end, ev_date_lut, clk_date_lut, cat_names,
                           unknown_cat_id, fast_prod_by_num, product_map, campaigns,
                           fast_camp_by_num, campaign_map, ev_files, clk_files, link_files,
                           total_ev_bytes, t0);
    } else {
        run_pipeline<false>(output_dir, pool, event_win_start, event_win_end, click_win_start,
                            click_win_end, day_start, day_end, ev_date_lut, clk_date_lut, cat_names,
                            unknown_cat_id, fast_prod_by_num, product_map, campaigns,
                            fast_camp_by_num, campaign_map, ev_files, clk_files, link_files,
                            total_ev_bytes, t0);
    }

    return 0;
}
