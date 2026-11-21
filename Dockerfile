# Shopping Sessions: native C++20 zero-dependency pipeline with representative PGO.
# Builds for linux/amd64 natively on the Intel Xeon Cascade Lake+ scorers.
FROM gcc:13-bookworm AS builder

WORKDIR /build
COPY src/ /build/src/

# 1. Build PGO-instrumented binary (thread-safe atomic profile updates; normal exit flushes .gcda)
RUN g++ -O3 -std=c++20 -march=x86-64-v3 -mtune=cascadelake -flto=auto -DNDEBUG \
    -fprofile-generate=/tmp/pgo -fprofile-update=prefer-atomic \
    -pthread -static-libstdc++ -static-libgcc \
    /build/src/main.cpp -o /build/shopping_sessions

# 2. Generate representative PGO training dataset (realistic 9-byte base36 IDs, whales, chains, kiosks, dirt)
RUN g++ -O2 -std=c++20 -x c++ - -o /tmp/pgo_gen << 'EOF' && \
    /tmp/pgo_gen && \
    SS_PGO_RUN=1 NCPU=8 /build/shopping_sessions /tmp/pgo_in /tmp/pgo_out && \
    rm -rf /tmp/pgo_in /tmp/pgo_out /tmp/pgo_gen /build/shopping_sessions
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static uint64_t rng_state = 123456789ULL;
static uint64_t next_u64() {
    rng_state += 0x9e3779b97f4a7c15ULL;
    uint64_t z = rng_state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static uint64_t below(uint64_t n) { return next_u64() % n; }

static std::string make_id(char prefix, uint64_t v) {
    static constexpr char kDigits[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    uint64_t m = (v * 0x9E3779B97F4BULL) & ((1ULL << 40) - 1);
    m ^= m >> 17;
    m = (m * 0xD6E8FEB86659ULL) & ((1ULL << 40) - 1);
    m |= (1ULL << 39); // ensure 8 base36 chars -> 9-byte realistic IDs
    char buf[10];
    buf[0] = prefix;
    int pos = 9;
    while (pos > 1) {
        buf[--pos] = kDigits[m % 36];
        m /= 36;
    }
    return std::string(buf, 9);
}

static std::string maybe_quote(const std::string& s) {
    return (below(5) == 0) ? ("\"" + s + "\"") : s;
}

int main() {
    std::string d = "/tmp/pgo_in";
    for (auto sub : {"events", "clicks", "links"}) fs::create_directories(d + "/" + sub);
    {
        std::ofstream mf(d + "/manifest.json");
        mf << "{\"window_start\":\"2026-10-01T00:00:00Z\",\"window_end\":\"2026-10-15T00:00:00Z\"}\n";
    }
    {
        std::ofstream pf(d + "/products.csv");
        pf << "product_id,category\n";
        for (int i = 1; i <= 50000; ++i) pf << "p" << i << ",cat" << (i % 20) << "\n";
    }
    {
        std::ofstream cf(d + "/campaigns.csv");
        cf << "campaign_id,channel\n";
        const char* ch[5] = {"affiliate", "display", "email", "search", "social"};
        for (int i = 1; i <= 2000; ++i) cf << "c_" << i << "," << ch[i % 5] << "\n";
    }
    const int n_users = 27000;
    std::vector<std::string> uids(n_users);
    for (int i = 0; i < n_users; ++i) uids[i] = make_id('u', static_cast<uint64_t>(i) * 512ULL);

    // Links: normal trees, 300-node chains, 3 shared kiosks, repeats, and dirt
    {
        std::vector<std::ofstream> lf;
        for (int p = 0; p < 8; ++p) {
            lf.emplace_back(d + "/links/part-" + std::to_string(p) + ".csv");
            lf.back() << "user_id_a,user_id_b\n";
        }
        for (int i = 0; i < n_users; ++i) {
            int aliases = (i < 5) ? 4 : (below(100) < 45 ? 1 : (below(100) < 60 ? 2 : 3));
            for (int a = 1; a < aliases; ++a) {
                std::string ua = make_id('u', static_cast<uint64_t>(i) * 512ULL + a);
                std::string ub = make_id('u', static_cast<uint64_t>(i) * 512ULL + below(a));
                lf[i & 7] << maybe_quote(ua) << "," << maybe_quote(ub) << "\n";
                if (below(12) == 0) lf[i & 7] << ub << "," << ua << "\n";
            }
            if (below(250) == 0) {
                lf[i & 7] << uids[i] << "," << make_id('u', (1ULL << 39) + below(3)) << "\n";
            }
        }
        for (int c = 0; c < 300; ++c) {
            lf[0] << make_id('u', 90000000ULL + c) << "," << make_id('u', 90000001ULL + c) << "\n";
        }
        lf[2] << "bad_link_only_one_col\n,empty_left\n";
    }
    const int64_t base_ms = 1790812800000LL;
    auto fmt_iso = [&](int64_t ms, bool f2) -> std::string {
        int64_t off = f2 ? -240 : 0;
        int64_t loc = ms + off * 60000LL;
        int64_t day = (loc - base_ms) / 86400000LL + 1;
        if (day < 1) day = 1;
        if (day > 14) day = 14;
        int64_t sec = ((loc % 86400000LL) + 86400000LL) % 86400000LL / 1000LL;
        char b[40];
        if (!f2) {
            std::snprintf(b, sizeof(b), "2026-10-%02lldT%02lld:%02lld:%02lldZ",
                          (long long)day, (long long)(sec / 3600),
                          (long long)((sec / 60) % 60), (long long)(sec % 60));
        } else {
            std::snprintf(b, sizeof(b), "2026-10-%02lldT%02lld:%02lld:%02lld-04:00",
                          (long long)day, (long long)(sec / 3600),
                          (long long)((sec / 60) % 60), (long long)(sec % 60));
        }
        return std::string(b);
    };
    {
        std::vector<std::ofstream> ef;
        for (int p = 0; p < 16; ++p) {
            ef.emplace_back(d + "/events/part-" + std::to_string(p) + ".csv");
            ef.back() << "event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n";
        }
        for (int i = 0; i < 800000; ++i) {
            int fidx = (below(10) < 4) ? 0 : static_cast<int>(below(16));
            uint64_t uidx = (i < 60000) ? (i % 5) : (5 + below(n_users - 5));
            std::string u = make_id('u', uidx * 512ULL + below(2));
            int64_t ts = base_ms + static_cast<int64_t>(below(14 * 86400)) * 1000LL +
                         (below(30) == 0 ? static_cast<int64_t>(below(999)) + 1 : 0);
            int64_t ing = ts + (below(100) == 0 ? 90000000LL : (below(200) == 0 ? -2000LL : 60000LL));
            int r = static_cast<int>(below(10));
            std::string ts_s = (ts % 1000 != 0 || r >= 7) ? std::to_string(ts)
                                                          : fmt_iso(ts, r == 6);
            if (below(200) == 0) ts_s = "bad_ts";
            std::string ing_s = fmt_iso(ing, false);
            int et = static_cast<int>(below(100));
            const char* et_s = (et < 3) ? "purchase" : (et < 13 ? "cart" : "view");
            std::string pr = (et < 3) ? std::to_string(100 + below(40000)) : "";
            if (et < 3 && below(20) == 0) pr = "19.99";
            int eid_num = (below(30) == 0 && i > 0) ? (i - 1) : i;
            std::string eid_s = make_id('e', static_cast<uint64_t>(eid_num));
            std::string prod = (below(200) == 0) ? "p99999" : ("p" + std::to_string(1 + below(50000)));
            ef[fidx] << maybe_quote(eid_s) << "," << maybe_quote(u) << "," << maybe_quote(ts_s) << ","
                     << maybe_quote(ing_s) << "," << maybe_quote(et_s) << "," << maybe_quote(prod)
                     << "," << maybe_quote(pr) << "\n";
            if (below(2000) == 0) ef[fidx] << "malformed,line\n";
        }
    }
    {
        std::vector<std::ofstream> kf;
        for (int p = 0; p < 8; ++p) {
            kf.emplace_back(d + "/clicks/part-" + std::to_string(p) + ".csv");
            kf.back() << "click_id,user_id,campaign_id,ts\n";
        }
        for (int i = 0; i < 200000; ++i) {
            uint64_t uidx = (i < 10000) ? (i % 5) : (5 + below(n_users - 5));
            std::string u = make_id('u', uidx * 512ULL + below(2));
            int64_t ts = base_ms - 3 * 86400000LL + static_cast<int64_t>(below(17 * 86400)) * 1000LL;
            std::string ts_s = (below(3) == 0) ? std::to_string(ts) : fmt_iso(ts, below(6) == 0);
            std::string camp = (below(200) == 0) ? "c_9999" : ("c_" + std::to_string(1 + below(2000)));
            kf[i & 7] << maybe_quote(make_id('k', static_cast<uint64_t>(i))) << ","
                      << maybe_quote(u) << "," << maybe_quote(camp) << "," << maybe_quote(ts_s) << "\n";
        }
    }
    return 0;
}
EOF

# 3. Rebuild final binary using collected PGO profile
RUN g++ -O3 -std=c++20 -march=x86-64-v3 -mtune=cascadelake -flto=auto -DNDEBUG \
    -fprofile-use=/tmp/pgo -fprofile-correction -Wall -Wextra -Werror \
    -pthread -static-libstdc++ -static-libgcc \
    /build/src/main.cpp -o /build/shopping_sessions && \
    strip --strip-all /build/shopping_sessions && \
    rm -rf /tmp/pgo

FROM debian:bookworm-slim

RUN useradd --uid 1000 --no-create-home --home-dir /scratch app && \
    mkdir -p /app && chown 1000:1000 /app

COPY --from=builder /build/shopping_sessions /app/shopping_sessions

USER 1000:1000
WORKDIR /app
CMD ["/app/shopping_sessions"]
