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

# 2. Generate representative PGO training dataset (whales >4096 events, chains, kiosks, dirt, F1/F2/F3, multi-part)
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
        for (int i = 1; i <= 5000; ++i) pf << "p" << i << ",cat" << (i % 20) << "\n";
    }
    {
        std::ofstream cf(d + "/campaigns.csv");
        cf << "campaign_id,channel\n";
        const char* ch[5] = {"affiliate", "display", "email", "search", "social"};
        for (int i = 1; i <= 500; ++i) cf << "c_" << i << "," << ch[i % 5] << "\n";
    }
    // Links: normal trees, 250-node chains, and a shared kiosk
    {
        std::vector<std::ofstream> lf;
        for (int p = 0; p < 4; ++p) {
            lf.emplace_back(d + "/links/part-" + std::to_string(p) + ".csv");
            lf.back() << "user_id_a,user_id_b\n";
        }
        for (int i = 1; i <= 15000; ++i) {
            lf[i & 3] << "u" << i << ",ua" << i << "\n";
            if (i % 4 == 0) lf[i & 3] << "\"u" << i << "\",\"ub" << i << "\"\n";
        }
        for (int c = 0; c < 250; ++c) lf[0] << "chain_" << c << ",chain_" << (c + 1) << "\n";
        for (int k = 1; k <= 500; ++k) lf[1] << "u" << k << ",kiosk_shared_device_01\n";
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
        for (int p = 0; p < 8; ++p) {
            ef.emplace_back(d + "/events/part-" + std::to_string(p) + ".csv");
            ef.back() << "event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n";
        }
        // 5 whale users with 6,000 events each (>2048 -> triggers LSD radix sort) + 370k normal events
        for (int i = 0; i < 400000; ++i) {
            int fidx = (below(10) < 4) ? 0 : static_cast<int>(below(8));
            std::string u = (i < 30000) ? ("whale_" + std::to_string(i % 5))
                                        : ("u" + std::to_string(1 + below(15000)));
            if (below(5) == 0) u = "\"" + u + "\"";
            int64_t ts = base_ms + static_cast<int64_t>(below(14 * 86400)) * 1000LL +
                         (below(30) == 0 ? static_cast<int64_t>(below(999)) + 1 : 0);
            int64_t ing = ts + (below(100) == 0 ? 90000000LL : 60000LL);
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
            std::string prod = (below(200) == 0) ? "p99999" : ("p" + std::to_string(1 + below(5000)));
            ef[fidx] << "e" << eid_num << "," << u << "," << ts_s << "," << ing_s << ","
                     << et_s << "," << prod << "," << pr << "\n";
            if (below(2000) == 0) ef[fidx] << "malformed,line\n";
        }
    }
    {
        std::vector<std::ofstream> kf;
        for (int p = 0; p < 4; ++p) {
            kf.emplace_back(d + "/clicks/part-" + std::to_string(p) + ".csv");
            kf.back() << "click_id,user_id,campaign_id,ts\n";
        }
        for (int i = 0; i < 100000; ++i) {
            std::string u = (i < 5000) ? ("whale_" + std::to_string(i % 5))
                                       : ("u" + std::to_string(1 + below(15000)));
            int64_t ts = base_ms - 3 * 86400000LL + static_cast<int64_t>(below(17 * 86400)) * 1000LL;
            std::string ts_s = (below(3) == 0) ? std::to_string(ts) : fmt_iso(ts, below(6) == 0);
            kf[i & 3] << "k" << i << "," << u << ",c_" << (1 + below(500)) << "," << ts_s << "\n";
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
