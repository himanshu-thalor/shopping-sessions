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
#include <vector>

#include "csv_parse.hpp"
#include "hash_util.hpp"
#include "io_util.hpp"
#include "parquet_writer.hpp"
#include "union_find.hpp"

namespace {

constexpr size_t kNumBuckets = 256;
constexpr int64_t kDayMs = 86400000LL;
constexpr int64_t kLateCutoffMs = 86400000LL;
constexpr int64_t kSessionGapMs = 1800000LL;
constexpr int64_t kClickReachMs = 7LL * 86400000LL;
constexpr int64_t kTombstoneTs = std::numeric_limits<int64_t>::min();

struct RawEvent {
    CompactStr event_id;
    CompactStr user_id;
    uint32_t price_cents;
    uint32_t cat_id;
    int64_t ts_ms;
    int64_t ingest_ms;
    uint64_t eid_hash;
};
static_assert(sizeof(RawEvent) == 56);
static_assert(std::is_trivially_copyable_v<RawEvent>);

struct SurvivingEvent {
    CompactStr user_id;
    uint32_t price_cents;
    int64_t ts_ms;
    uint64_t person_hash;
    uint32_t cat_id;
    uint32_t pad;
};
static_assert(sizeof(SurvivingEvent) == 40);
static_assert(std::is_trivially_copyable_v<SurvivingEvent>);

struct SurvivingClick {
    int64_t ts_ms;
    uint64_t person_hash;
    CompactStr user_id;
    CompactStr click_id;
    uint32_t camp_id;
    uint32_t pad;
};
static_assert(sizeof(SurvivingClick) == 48);
static_assert(std::is_trivially_copyable_v<SurvivingClick>);

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

struct SessionInfo {
    int64_t start_ms;
    uint64_t rev;
    int32_t day;
    uint8_t has_purch;
};

struct CampDayAgg {
    int64_t sessions = 0;
    int64_t purchasing_sessions = 0;
    int64_t revenue_cents = 0;
    int64_t assisted_sessions = 0;
};

struct CampDayHashMap {
    struct Slot {
        uint64_t key = 0;  // UINT64_MAX == empty
        CampDayAgg agg{};
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

    CampDayAgg& get(uint32_t camp_id, int32_t day) {
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

struct DedupSlot {
    uint64_t eid_hash = 0;  // 0 == empty
    CompactStr event_id{};
    CompactStr user_id{};
    uint32_t price_cents = 0;
    uint32_t cat_id = 0;
    int64_t ts_ms = 0;
    int64_t ingest_ms = 0;
};

struct PersonMapSlot {
    uint64_t hash = 0;  // 0 == empty
    CompactStr user_id{};
    uint32_t local_pid = 0;
};

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

    // Phase 0: manifest, products, campaigns
    int64_t event_win_start = 0, event_win_end = 0;
    parse_manifest(join_path(data_dir, "manifest.json"), event_win_start, event_win_end);
    const int64_t click_win_start = event_win_start - kClickReachMs;
    const int64_t click_win_end = event_win_end;
    const int32_t day_start = floor_div_day(event_win_start);
    const int32_t day_end = floor_div_day(event_win_end - 1);

    // Load products.csv -> sort unique categories + "unknown" byte-wise
    std::vector<std::string> cat_names;
    FlatStrMap<uint32_t> product_map;
    uint32_t unknown_cat_id = 0;
    {
        MappedFile pf = MappedFile::open_readonly(join_path(data_dir, "products.csv"));
        struct ProdTmp {
            std::string_view pid;
            std::string_view cat;
        };
        std::vector<ProdTmp> rows;
        FlatStrMap<uint32_t> tmp_cats;
        tmp_cats.emplace("unknown", 0);
        cat_names.emplace_back("unknown");

        if (pf.size > 0) {
            const char* nl = static_cast<const char*>(std::memchr(pf.data, '\n', pf.size));
            if (nl) {
                const char* p = nl + 1;
                const char* end = pf.data + pf.size;
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

    // Load campaigns.csv -> index 0 reserved for ("organic", "organic")
    struct CampaignInfo {
        std::string id;
        std::string channel;
    };
    std::vector<CampaignInfo> campaigns;
    campaigns.push_back({"organic", "organic"});
    FlatStrMap<uint32_t> campaign_map;
    {
        MappedFile cf = MappedFile::open_readonly(join_path(data_dir, "campaigns.csv"));
        if (cf.size > 0) {
            const char* nl = static_cast<const char*>(std::memchr(cf.data, '\n', cf.size));
            if (nl) {
                const char* p = nl + 1;
                const char* end = cf.data + cf.size;
                std::string_view f[2];
                while (p < end) {
                    const char* lnl = static_cast<const char*>(
                        std::memchr(p, '\n', static_cast<size_t>(end - p)));
                    const char* lend = lnl ? lnl : end;
                    if (split_csv_line<2>(p, static_cast<size_t>(lend - p), f) && !f[0].empty()) {
                        const uint32_t cid = static_cast<uint32_t>(campaigns.size());
                        if (campaign_map.emplace(f[0], cid).second) {
                            campaigns.push_back({std::string(f[0]), std::string(f[1])});
                        }
                    }
                    p = lnl ? (lnl + 1) : end;
                }
            }
        }
    }

    // Phase 1: LinkTable
    LinkTable links;
    links.build(join_path(data_dir, "links"), ncpu);
    const auto t1 = std::chrono::steady_clock::now();

    // Thread-owned arenas (user_arenas stay alive through Phase 5; event_arenas freed after Phase 3)
    std::vector<OverflowArena> event_arenas(ncpu);
    std::vector<OverflowArena> user_arenas(ncpu);
    std::vector<OverflowArena> click_arenas(ncpu);

    // Phase 2: Parallel scan of events/*.csv into 256 event_id hash buckets per thread
    std::vector<std::vector<ChunkedVec<RawEvent>>> raw_ev_buckets(ncpu);
    for (size_t t = 0; t < ncpu; ++t) raw_ev_buckets[t].resize(kNumBuckets);
    {
        auto ev_files = map_all_csvs(join_path(data_dir, "events"));
        auto ev_slices = build_slices(ev_files);
        const size_t nworkers = std::min(ncpu, std::max<size_t>(1, ev_slices.size()));
        std::atomic<size_t> next_slice{0};

        auto worker = [&](size_t tid) {
            auto& ea = event_arenas[tid];
            auto& ua = user_arenas[tid];
            auto& buckets = raw_ev_buckets[tid];
            const uint16_t aid = static_cast<uint16_t>(tid);
            std::string_view f[7];

            for (;;) {
                const size_t si = next_slice.fetch_add(1, std::memory_order_relaxed);
                if (si >= ev_slices.size()) break;
                const FileSlice& s = ev_slices[si];
                const char* p = s.base + s.start;
                const char* end = s.base + s.end;
                while (p < end) {
                    const char* line_start = p;
                    const char* nl =
                        static_cast<const char*>(std::memchr(p, '\n', static_cast<size_t>(end - p)));
                    const char* lend = nl ? nl : end;
                    const size_t len = static_cast<size_t>(lend - line_start);
                    p = nl ? (nl + 1) : end;

                    if (!split_csv_line<7>(line_start, len, f)) {
                        continue;
                    }
                    if (f[0].empty()) continue;

                    int64_t ingest_ms = 0;
                    if (!parse_ts_f1(f[3], ingest_ms)) continue;

                    uint64_t eh = wyhash64(f[0]);
                    if (eh == 0) eh = 1;
                    const size_t b = static_cast<size_t>(eh >> 56);

                    bool valid = !f[1].empty() && !f[5].empty();
                    int64_t ts_ms = 0;
                    if (valid) {
                        valid = parse_ts_any(f[2], ts_ms) && ts_ms >= event_win_start &&
                                ts_ms < event_win_end && (ingest_ms - ts_ms) <= kLateCutoffMs;
                    }

                    uint32_t price_cents = 0;
                    uint32_t cat_id = 0;
                    if (valid) {
                        const std::string_view et = f[4];
                        if (et == "view" || et == "cart") {
                            price_cents = 0;
                        } else if (et == "purchase") {
                            valid = parse_price_cents(f[6], price_cents);
                            if (valid) {
                                const uint32_t* pc = product_map.find(f[5]);
                                cat_id = pc ? *pc : unknown_cat_id;
                            }
                        } else {
                            valid = false;
                        }
                    }

                    RawEvent re{};
                    re.event_id = CompactStr::make_str(f[0], aid, ea);
                    re.ingest_ms = ingest_ms;
                    re.eid_hash = eh;
                    if (valid) {
                        if (auto canon = links.resolve_user(f[1])) {
                            re.user_id = CompactStr::make_linked(*canon);
                        } else {
                            re.user_id = CompactStr::make_str(f[1], aid, ua);
                        }
                        re.ts_ms = ts_ms;
                        re.price_cents = price_cents;
                        re.cat_id = cat_id;
                    } else {
                        re.ts_ms = kTombstoneTs;
                    }
                    buckets[b].push_back(re);
                }
            }
        };

        if (nworkers == 1) {
            worker(0);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(nworkers);
            for (size_t t = 0; t < nworkers; ++t) threads.emplace_back(worker, t);
            for (auto& th : threads) th.join();
        }
    }
    product_map.clear_and_free();
    const auto t2 = std::chrono::steady_clock::now();

    // Phase 3: Parallel event_id dedup across 256 buckets -> 256 person shards
    std::vector<std::vector<ChunkedVec<SurvivingEvent>>> ev_person_buckets(ncpu);
    for (size_t t = 0; t < ncpu; ++t) ev_person_buckets[t].resize(kNumBuckets);
    {
        std::atomic<size_t> next_bucket{0};
        auto worker = [&](size_t tid) {
            const OverflowArena* ea_ptr = event_arenas.data();
            const OverflowArena* ua_ptr = user_arenas.data();
            auto& out_buckets = ev_person_buckets[tid];
            std::vector<DedupSlot> table;

            for (;;) {
                const size_t b = next_bucket.fetch_add(1, std::memory_order_relaxed);
                if (b >= kNumBuckets) break;

                size_t total = 0;
                for (size_t t = 0; t < ncpu; ++t) total += raw_ev_buckets[t][b].size();
                if (total == 0) continue;

                size_t cap = 64;
                while (cap * 11 < total * 20) cap <<= 1;
                table.assign(cap, DedupSlot{});
                const size_t mask = cap - 1;

                for (size_t t = 0; t < ncpu; ++t) {
                    raw_ev_buckets[t][b].for_each_block([&](const RawEvent* ptr, size_t n) {
                        for (size_t i = 0; i < n; ++i) {
                            const RawEvent& re = ptr[i];
                            size_t idx = static_cast<size_t>(re.eid_hash) & mask;
                            for (;;) {
                                DedupSlot& sl = table[idx];
                                if (sl.eid_hash == 0) {
                                    sl.eid_hash = re.eid_hash;
                                    sl.event_id = re.event_id;
                                    sl.user_id = re.user_id;
                                    sl.price_cents = re.price_cents;
                                    sl.cat_id = re.cat_id;
                                    sl.ts_ms = re.ts_ms;
                                    sl.ingest_ms = re.ingest_ms;
                                    break;
                                }
                                if (sl.eid_hash == re.eid_hash &&
                                    CompactStr::raw_equals(sl.event_id, re.event_id, ea_ptr)) {
                                    if (re.ingest_ms < sl.ingest_ms) {
                                        sl.user_id = re.user_id;
                                        sl.price_cents = re.price_cents;
                                        sl.cat_id = re.cat_id;
                                        sl.ts_ms = re.ts_ms;
                                        sl.ingest_ms = re.ingest_ms;
                                    }
                                    break;
                                }
                                idx = (idx + 1) & mask;
                            }
                        }
                    });
                    raw_ev_buckets[t][b].clear_and_free();
                }

                for (const DedupSlot& sl : table) {
                    if (sl.eid_hash == 0 || sl.ts_ms == kTombstoneTs) continue;
                    uint64_t ph = sl.user_id.is_linked()
                                      ? links.canonical_hash(sl.user_id.linked_id())
                                      : wyhash64(sl.user_id.raw_view(ua_ptr));
                    if (ph == 0) ph = 1;
                    out_buckets[static_cast<size_t>(ph >> 56)].push_back(SurvivingEvent{
                        sl.user_id, sl.price_cents, sl.ts_ms, ph, sl.cat_id, 0});
                }
            }
        };

        if (ncpu == 1) {
            worker(0);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(ncpu);
            for (size_t t = 0; t < ncpu; ++t) threads.emplace_back(worker, t);
            for (auto& th : threads) th.join();
        }
    }
    std::vector<std::vector<ChunkedVec<RawEvent>>>().swap(raw_ev_buckets);
    for (auto& ea : event_arenas) ea.free_all();
    std::vector<OverflowArena>().swap(event_arenas);
    const auto t3 = std::chrono::steady_clock::now();

    // Phase 4: Parallel scan of clicks/*.csv into 256 person shards per thread
    std::vector<std::vector<ChunkedVec<SurvivingClick>>> clk_person_buckets(ncpu);
    for (size_t t = 0; t < ncpu; ++t) clk_person_buckets[t].resize(kNumBuckets);
    {
        auto clk_files = map_all_csvs(join_path(data_dir, "clicks"));
        auto clk_slices = build_slices(clk_files);
        const size_t nworkers = std::min(ncpu, std::max<size_t>(1, clk_slices.size()));
        std::atomic<size_t> next_slice{0};

        auto worker = [&](size_t tid) {
            auto& ua = user_arenas[tid];
            auto& ca = click_arenas[tid];
            auto& buckets = clk_person_buckets[tid];
            const uint16_t aid = static_cast<uint16_t>(tid);
            std::string_view f[4];

            for (;;) {
                const size_t si = next_slice.fetch_add(1, std::memory_order_relaxed);
                if (si >= clk_slices.size()) break;
                const FileSlice& s = clk_slices[si];
                const char* p = s.base + s.start;
                const char* end = s.base + s.end;
                while (p < end) {
                    const char* line_start = p;
                    const char* nl =
                        static_cast<const char*>(std::memchr(p, '\n', static_cast<size_t>(end - p)));
                    const char* lend = nl ? nl : end;
                    const size_t len = static_cast<size_t>(lend - line_start);
                    p = nl ? (nl + 1) : end;

                    if (!split_csv_line<4>(line_start, len, f)) continue;
                    if (f[0].empty() || f[1].empty()) continue;

                    const uint32_t* cid = campaign_map.find(f[2]);
                    if (!cid) continue;

                    int64_t ts_ms = 0;
                    if (!parse_ts_any(f[3], ts_ms) || ts_ms < click_win_start ||
                        ts_ms >= click_win_end) {
                        continue;
                    }

                    CompactStr uid{};
                    uint64_t ph = 0;
                    if (auto canon = links.resolve_user(f[1])) {
                        uid = CompactStr::make_linked(*canon);
                        ph = links.canonical_hash(*canon);
                    } else {
                        uid = CompactStr::make_str(f[1], aid, ua);
                        ph = wyhash64(f[1]);
                    }
                    if (ph == 0) ph = 1;

                    const CompactStr kid = CompactStr::make_str(f[0], aid, ca);
                    buckets[static_cast<size_t>(ph >> 56)].push_back(
                        SurvivingClick{ts_ms, ph, uid, kid, *cid, 0});
                }
            }
        };

        if (nworkers == 1) {
            worker(0);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(nworkers);
            for (size_t t = 0; t < nworkers; ++t) threads.emplace_back(worker, t);
            for (auto& th : threads) th.join();
        }
    }
    campaign_map.clear_and_free();
    const auto t4 = std::chrono::steady_clock::now();

    // Phase 5: Parallel processing of 256 person shards (LPT descending order)
    std::vector<size_t> shard_order(kNumBuckets);
    std::iota(shard_order.begin(), shard_order.end(), 0);
    std::vector<size_t> shard_load(kNumBuckets, 0);
    for (size_t s = 0; s < kNumBuckets; ++s) {
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
    const int64_t num_days_i64 = static_cast<int64_t>(day_end) - static_cast<int64_t>(day_start) + 1;
    const bool use_flat_camp =
        (num_days_i64 > 0) &&
        (num_camps <= (2000000ULL / static_cast<size_t>(num_days_i64)));
    const size_t flat_camp_size = use_flat_camp ? (num_camps * static_cast<size_t>(num_days_i64)) : 0;

    struct ThreadAgg {
        std::vector<CampDayAgg> flat;
        CampDayHashMap map;
        int64_t total_sessions = 0;
        int64_t total_revenue = 0;
    };
    std::vector<ThreadAgg> thread_aggs(ncpu);
    for (size_t t = 0; t < ncpu; ++t) {
        if (use_flat_camp) thread_aggs[t].flat.resize(flat_camp_size);
    }

    {
        std::atomic<size_t> next_order_idx{0};
        auto worker = [&](size_t tid) {
            const OverflowArena* ua_ptr = user_arenas.data();
            const OverflowArena* ca_ptr = click_arenas.data();
            auto& agg = thread_aggs[tid];

            const std::string out_path =
                join_path(join_path(output_dir, "user_daily"),
                          "part-" + std::to_string(tid) + ".parquet");
            UserDailyParquetWriter writer(out_path);

            auto add_credit = [&](uint32_t camp_id, int32_t day, uint8_t has_purch, uint64_t rev) {
                CampDayAgg& cd =
                    use_flat_camp
                        ? agg.flat[static_cast<size_t>(camp_id) * static_cast<size_t>(num_days_i64) +
                                   static_cast<size_t>(day - day_start)]
                        : agg.map.get(camp_id, day);
                ++cd.sessions;
                cd.purchasing_sessions += has_purch;
                cd.revenue_cents += static_cast<int64_t>(rev);
            };
            auto add_assist = [&](uint32_t camp_id, int32_t day) {
                CampDayAgg& cd =
                    use_flat_camp
                        ? agg.flat[static_cast<size_t>(camp_id) * static_cast<size_t>(num_days_i64) +
                                   static_cast<size_t>(day - day_start)]
                        : agg.map.get(camp_id, day);
                ++cd.assisted_sessions;
            };

            std::vector<PersonMapSlot> pmap;
            std::vector<CompactStr> pid_user;
            std::vector<uint32_t> ev_cnt, clk_cnt, ev_off, clk_off, ev_cur, clk_cur;
            std::vector<ShardEvent> shard_events;
            std::vector<ShardClick> shard_clicks;
            std::vector<SessionInfo> sessions;
            std::vector<uint64_t> cat_rev(num_cats, 0);
            std::vector<uint32_t> touched_cats;
            std::vector<uint32_t> camp_count(num_camps, 0);
            std::vector<uint32_t> active_camps;
            std::vector<uint32_t> active_pos(num_camps, std::numeric_limits<uint32_t>::max());

            for (;;) {
                const size_t oi = next_order_idx.fetch_add(1, std::memory_order_relaxed);
                if (oi >= kNumBuckets) break;
                const size_t s = shard_order[oi];

                size_t total_ev = 0;
                for (size_t t = 0; t < ncpu; ++t) total_ev += ev_person_buckets[t][s].size();
                if (total_ev == 0) {
                    for (size_t t = 0; t < ncpu; ++t) clk_person_buckets[t][s].clear_and_free();
                    continue;
                }

                // 1. Intern persons in shard s from surviving events
                size_t pcap = 64;
                while (pcap * 11 < total_ev * 10) pcap <<= 1;  // ~2x unique persons (<= total_ev)
                if (pcap < 128) pcap = 128;
                pmap.assign(pcap, PersonMapSlot{});
                size_t pmask = pcap - 1;
                pid_user.clear();
                ev_cnt.clear();

                auto rehash_pmap = [&]() {
                    const size_t ncap = pmap.size() * 2;
                    std::vector<PersonMapSlot> nxt(ncap);
                    const size_t nmask = ncap - 1;
                    for (const auto& sl : pmap) {
                        if (sl.hash == 0) continue;
                        size_t idx = static_cast<size_t>(sl.hash) & nmask;
                        while (nxt[idx].hash != 0) idx = (idx + 1) & nmask;
                        nxt[idx] = sl;
                    }
                    pmap.swap(nxt);
                    pmask = nmask;
                };

                auto find_pid = [&](uint64_t h, const CompactStr& uid) -> uint32_t {
                    size_t idx = static_cast<size_t>(h) & pmask;
                    for (;;) {
                        const PersonMapSlot& sl = pmap[idx];
                        if (sl.hash == 0) return std::numeric_limits<uint32_t>::max();
                        if (sl.hash == h && CompactStr::raw_equals(sl.user_id, uid, ua_ptr)) {
                            return sl.local_pid;
                        }
                        idx = (idx + 1) & pmask;
                    }
                };

                for (size_t t = 0; t < ncpu; ++t) {
                    ev_person_buckets[t][s].for_each_block([&](const SurvivingEvent* ptr, size_t n) {
                        for (size_t i = 0; i < n; ++i) {
                            const SurvivingEvent& se = ptr[i];
                            if ((pid_user.size() + 1) * 20 >= pmap.size() * 11) rehash_pmap();
                            size_t idx = static_cast<size_t>(se.person_hash) & pmask;
                            for (;;) {
                                PersonMapSlot& sl = pmap[idx];
                                if (sl.hash == 0) {
                                    const uint32_t pid = static_cast<uint32_t>(pid_user.size());
                                    sl.hash = se.person_hash;
                                    sl.user_id = se.user_id;
                                    sl.local_pid = pid;
                                    pid_user.push_back(se.user_id);
                                    ev_cnt.push_back(1);
                                    break;
                                }
                                if (sl.hash == se.person_hash &&
                                    CompactStr::raw_equals(sl.user_id, se.user_id, ua_ptr)) {
                                    ++ev_cnt[sl.local_pid];
                                    break;
                                }
                                idx = (idx + 1) & pmask;
                            }
                        }
                    });
                }

                const size_t num_persons = pid_user.size();
                ev_off.resize(num_persons + 1);
                ev_off[0] = 0;
                for (size_t p = 0; p < num_persons; ++p) ev_off[p + 1] = ev_off[p] + ev_cnt[p];
                ev_cur.assign(ev_off.begin(), ev_off.begin() + num_persons);
                shard_events.resize(total_ev);

                for (size_t t = 0; t < ncpu; ++t) {
                    ev_person_buckets[t][s].for_each_block([&](const SurvivingEvent* ptr, size_t n) {
                        for (size_t i = 0; i < n; ++i) {
                            const SurvivingEvent& se = ptr[i];
                            const uint32_t pid = find_pid(se.person_hash, se.user_id);
                            shard_events[ev_cur[pid]++] =
                                ShardEvent{se.ts_ms, se.price_cents, se.cat_id};
                        }
                    });
                    ev_person_buckets[t][s].clear_and_free();
                }

                // 2. Count & scatter clicks matching persons in shard s
                clk_cnt.assign(num_persons, 0);
                size_t total_clk = 0;
                for (size_t t = 0; t < ncpu; ++t) {
                    clk_person_buckets[t][s].for_each_block([&](const SurvivingClick* ptr, size_t n) {
                        for (size_t i = 0; i < n; ++i) {
                            const uint32_t pid = find_pid(ptr[i].person_hash, ptr[i].user_id);
                            if (pid != std::numeric_limits<uint32_t>::max()) {
                                ++clk_cnt[pid];
                                ++total_clk;
                            }
                        }
                    });
                }

                clk_off.resize(num_persons + 1);
                clk_off[0] = 0;
                for (size_t p = 0; p < num_persons; ++p) clk_off[p + 1] = clk_off[p] + clk_cnt[p];
                clk_cur.assign(clk_off.begin(), clk_off.begin() + num_persons);
                shard_clicks.resize(total_clk);

                for (size_t t = 0; t < ncpu; ++t) {
                    if (total_clk > 0) {
                        clk_person_buckets[t][s].for_each_block(
                            [&](const SurvivingClick* ptr, size_t n) {
                                for (size_t i = 0; i < n; ++i) {
                                    const SurvivingClick& sc = ptr[i];
                                    const uint32_t pid = find_pid(sc.person_hash, sc.user_id);
                                    if (pid != std::numeric_limits<uint32_t>::max()) {
                                        shard_clicks[clk_cur[pid]++] =
                                            ShardClick{sc.ts_ms, sc.camp_id, sc.click_id};
                                    }
                                }
                            });
                    }
                    clk_person_buckets[t][s].clear_and_free();
                }

                // 3. Per-person sessionization, user_daily, and campaign attribution/assists
                for (size_t p = 0; p < num_persons; ++p) {
                    ShardEvent* pev = shard_events.data() + ev_off[p];
                    const size_t nev = ev_off[p + 1] - ev_off[p];
                    std::sort(pev, pev + nev, [](const ShardEvent& a, const ShardEvent& b) {
                        return a.ts_ms < b.ts_ms;
                    });

                    const CompactStr& uid = pid_user[p];
                    const std::string_view user_sv =
                        uid.is_linked() ? links.node_sv(uid.linked_id()) : uid.raw_view(ua_ptr);

                    sessions.clear();
                    int64_t sess_start = pev[0].ts_ms;
                    int64_t prev_ts = pev[0].ts_ms;
                    int32_t sess_day = floor_div_day(sess_start);
                    uint64_t sess_rev = 0;
                    uint8_t sess_purch = 0;

                    int32_t cur_day = sess_day;
                    int64_t day_sessions = 1;
                    int64_t day_events = 0;
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
                        if (i > 0 && pev[i].ts_ms - prev_ts > kSessionGapMs) {
                            sessions.push_back(SessionInfo{sess_start, sess_rev, sess_day, sess_purch});
                            sess_start = pev[i].ts_ms;
                            sess_day = floor_div_day(sess_start);
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
                        prev_ts = pev[i].ts_ms;
                        ++day_events;
                        if (pev[i].price_cents > 0) {
                            sess_rev += pev[i].price_cents;
                            sess_purch = 1;
                            day_rev += pev[i].price_cents;
                            if (cat_rev[pev[i].cat_id] == 0) touched_cats.push_back(pev[i].cat_id);
                            cat_rev[pev[i].cat_id] += pev[i].price_cents;
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
                        ShardClick* pclk = shard_clicks.data() + clk_off[p];
                        std::sort(pclk, pclk + nclk, [&](const ShardClick& a, const ShardClick& b) {
                            if (a.ts_ms != b.ts_ms) return a.ts_ms < b.ts_ms;
                            return byte_less(b.click_id.raw_view(ca_ptr),
                                             a.click_id.raw_view(ca_ptr));
                        });

                        size_t l = 0, r = 0;
                        for (const auto& sess : sessions) {
                            const int64_t win_lo = sess.start_ms - kClickReachMs;
                            while (r < nclk && pclk[r].ts_ms <= sess.start_ms) {
                                const uint32_t c = pclk[r].camp_id;
                                if (camp_count[c]++ == 0) {
                                    active_pos[c] = static_cast<uint32_t>(active_camps.size());
                                    active_camps.push_back(c);
                                }
                                ++r;
                            }
                            while (l < r && pclk[l].ts_ms < win_lo) {
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
        };

        if (ncpu == 1) {
            worker(0);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(ncpu);
            for (size_t t = 0; t < ncpu; ++t) threads.emplace_back(worker, t);
            for (auto& th : threads) th.join();
        }
    }
    const auto t5 = std::chrono::steady_clock::now();

    // Phase 6: Reduce campaign_daily across threads & write campaign_daily/part-0.parquet
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
                CampDayAgg sum{};
                for (size_t t = 0; t < ncpu; ++t) {
                    const CampDayAgg& a = thread_aggs[t].flat[base + static_cast<size_t>(d)];
                    sum.sessions += a.sessions;
                    sum.purchasing_sessions += a.purchasing_sessions;
                    sum.revenue_cents += a.revenue_cents;
                    sum.assisted_sessions += a.assisted_sessions;
                }
                if (sum.sessions > 0 || sum.assisted_sessions > 0) {
                    camp_writer.add_row(cid, ch, static_cast<int32_t>(day_start + d), sum.sessions,
                                        sum.purchasing_sessions, sum.revenue_cents,
                                        sum.assisted_sessions);
                    camp_sessions_sum += sum.sessions;
                    camp_revenue_sum += sum.revenue_cents;
                    camp_assists_sum += sum.assisted_sessions;
                    if (c != 0) non_organic_sessions_sum += sum.sessions;
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
                CampDayAgg& m = merged.get(c, d);
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

    const auto t6 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::fprintf(stderr,
                 "done in %.1f ms (links %.1f, events %.1f, dedup %.1f, clicks %.1f, shards %.1f, "
                 "camp %.1f) | sessions=%lld/%lld rev=%lld/%lld assists=%lld>=%lld\n",
                 ms(t0, t6), ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4), ms(t4, t5),
                 ms(t5, t6), static_cast<long long>(user_sessions_sum),
                 static_cast<long long>(camp_sessions_sum),
                 static_cast<long long>(user_revenue_sum),
                 static_cast<long long>(camp_revenue_sum),
                 static_cast<long long>(camp_assists_sum),
                 static_cast<long long>(non_organic_sessions_sum));
    return 0;
}
