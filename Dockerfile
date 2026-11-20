# Shopping Sessions: native C++20 zero-dependency pipeline with PGO.
# Builds for linux/amd64 natively on the Intel Xeon Cascade Lake+ scorers.
FROM gcc:13-bookworm AS builder

WORKDIR /build
COPY src/ /build/src/

# 1. Build PGO-instrumented binary (must share exact -o path so .gcda filenames match)
RUN g++ -O3 -std=c++20 -march=x86-64-v3 -mtune=cascadelake -flto=auto -DNDEBUG \
    -fprofile-generate=/tmp/pgo -pthread -static-libstdc++ -static-libgcc \
    /build/src/main.cpp -o /build/shopping_sessions

# 2. Generate realistic synthetic PGO training dataset & collect execution profile
RUN python3 - << 'EOF' && \
    NCPU=4 /build/shopping_sessions /tmp/pgo_in /tmp/pgo_out && \
    rm -rf /tmp/pgo_in /tmp/pgo_out /build/shopping_sessions
import json, os, random

rng = random.Random(1)
d = "/tmp/pgo_in"
for sub in ("events", "clicks", "links"):
    os.makedirs(f"{d}/{sub}", exist_ok=True)

with open(f"{d}/manifest.json", "w") as f:
    json.dump({"window_start": "2026-10-01T00:00:00Z", "window_end": "2026-10-15T00:00:00Z"}, f)
with open(f"{d}/products.csv", "w") as f:
    f.write("product_id,category\n" + "\n".join(f"p{i},cat{i % 20}" for i in range(1, 2001)) + "\n")
with open(f"{d}/campaigns.csv", "w") as f:
    f.write("campaign_id,channel\n" + "\n".join(f"c_{i},search" for i in range(1, 201)) + "\n")
with open(f"{d}/links/part-00000.csv", "w") as f:
    f.write("user_id_a,user_id_b\n")
    for i in range(1, 5000):
        f.write(f"u{i},ua{i}\n")

base = 1790812800000
with open(f"{d}/events/part-00000.csv", "w") as f:
    f.write("event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n")
    for i in range(50000):
        u = f"u{rng.randint(1, 5000)}"
        if rng.random() < 0.2:
            u = f'"{u}"'
        t = base + rng.randint(0, 14 * 86400 - 1) * 1000
        day = (t - base) // 86400000 + 1
        sec = (t // 1000) % 86400
        h, m, s = sec // 3600, (sec // 60) % 60, sec % 60
        r = rng.random()
        if r < 0.58:
            ts = f"2026-10-{day:02d}T{h:02d}:{m:02d}:{s:02d}Z"
        elif r < 0.68:
            ts = f"2026-10-{day:02d}T{h:02d}:{m:02d}:{s:02d}+00:00"
        else:
            ts = str(t)
        ing = f"2026-10-{day:02d}T{h:02d}:{m:02d}:{s:02d}Z"
        et = "purchase" if i % 30 == 0 else ("cart" if i % 10 == 0 else "view")
        pr = str(rng.randint(100, 50000)) if et == "purchase" else ""
        eid = f"e{i // 2}" if i % 25 == 0 else f"e{i}"
        f.write(f"{eid},{u},{ts},{ing},{et},p{rng.randint(1, 2000)},{pr}\n")

with open(f"{d}/clicks/part-00000.csv", "w") as f:
    f.write("click_id,user_id,campaign_id,ts\n")
    for i in range(15000):
        t = base + rng.randint(0, 14 * 86400 - 1) * 1000
        ts = str(t) if i % 3 == 0 else "2026-10-05T10:00:00Z"
        f.write(f"k{i},u{rng.randint(1, 5000)},c_{rng.randint(1, 200)},{ts}\n")
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
