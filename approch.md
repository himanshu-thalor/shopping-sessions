# Shopping Sessions — Engineering Approach, Architecture & Optimization Notes

> Written for peer/review agents to understand the full problem specification, every invariant/edge case, the C++20 implementation under `src/`, our measured benchmark progression, and the remaining optimization headroom.

---

## 1. Track Summary & Scorer Environment

- **Track:** Databricks — Shopping Sessions (`shopping-sessions-4`)
- **Goal:** Transform messy e-commerce clickstream CSVs (`events/*.csv`, `clicks/*.csv`, `links/*.csv`) + lookup tables (`products.csv`, `campaigns.csv`, `manifest.json`) into two exact Parquet tables: `/output/user_daily/` and `/output/campaign_daily/`.
- **Scoring:** Lowest container start-to-exit wall-clock time among **100% exact** solutions.
  - Practice: ~20M events (`5M` clicks), median of 3 if `< 20 s`. Current `#1` on leaderboard: `2.74 s` (starter PySpark baseline: `171.91 s`).
  - Final: ~60M events (~`15M` clicks), median of 3 (top 5 re-timed 5× on an isolated machine), 180 s cap.
- **Container / Hardware Constraints:**
  - `linux/amd64` Docker image, Google Cloud `n2` Intel Xeon Cascade Lake+ (`AVX-512` available), `8` vCPUs (`NCPU=8`, hyperthreads), `24 GB` RAM (no swap), read-only rootfs except `/tmp` (`64 MB` tmpfs), `uid 1000`, no network, `/data` read-only, `/output` & `/scratch` writable (`<= 50 GB`).
  - Repo limit `<= 50 MB`, Docker image `<= 8 GB`, stdout `<= 50 MB` (all diagnostics go to `stderr`).

---

## 2. Exact Rules & Critical Edge Cases (Per `rules.md` & 36 Fixtures)

1. **CSV Dialect (`src/csv_parse.hpp`):**
   - Only `\n` terminates a line; `\r` is an ordinary character inside fields (e.g., `1000\r` fails purchase price validation; `...Z\r` fails timestamp parsing).
   - Whole-field double quotes (`"u1"` -> `u1`, `""` -> null/empty); any embedded or unbalanced quote (`a"b`, `u5"`, `"a""b"`), wrong field count, or empty line is malformed and dropped.
   - Leading/trailing spaces are significant (` u1` != `u1 ` != `u1`), and strings are strictly case-sensitive (`u1` != `U1`).
   - All numeric checks accept **ASCII digits `0`–`9` only** (rejecting Unicode digits such as `٢٥٠` or `２５０`).
   - Line 0 of every CSV in `events/`, `clicks/`, and `links/` is the header; subsequent rows identical to the header (e.g. fixture `L3`) are data rows. Files may be empty, header-only, or lack a trailing `\n`.

2. **Timestamps (`src/csv_parse.hpp`):**
   - `F1`: `YYYY-MM-DDTHH:MM:SSZ` (20 bytes, uppercase `T`/`Z`).
   - `F2`: `YYYY-MM-DDTHH:MM:SS±HH:MM` (25 bytes, offset hours `00..23`, minutes `00..59`; `-00:00` is UTC).
   - `F3`: ASCII digits only (epoch ms, leading zeros allowed; numeric overflow beyond `int64_t` is outside any window).
   - Strict calendar/leap-year validation (`is_leap_year` handles `2028-02-29` valid vs `2027-02-29`, `2100-02-29`, `02-30`, hour `24`, second `60`).
   - Windows:
     - Events: `window_start <= ts < window_end`
     - Clicks: `window_start - 7 days <= ts < window_end`

3. **Linked IDs (`src/union_find.hpp`):**
   - Drop malformed lines or lines with either side empty.
   - Transitive connected components across all `links/*.csv` files (chains up to 400 links long, shared kiosk devices tying thousands of users into one component).
   - Canonical person name = **unsigned byte-wise minimum** (`std::memcmp`, NEVER signed `char`) among all IDs in the component, **even if that smallest ID never appears in any event or click** (`L2`).

4. **Events Processing Order (`src/main.cpp`):**
   - Parse -> drop malformed or empty `event_id`.
   - **Dedup on `event_id` keeping smallest `ingest_ts` BEFORE validation:** if the earliest-ingested copy has an invalid `ts`, invalid `type`, bad `price_cents`, empty `user_id`/`product_id`, or arrives late (`ingest_ts - ts > 24 h`), the whole event is dropped (`E5`)—later copies cannot rescue it.
   - Validation: `user_id` non-empty, `ts` in event window, `ingest_ts - ts <= 86,400,000 ms` (`ingest_ts < ts` kept), `event_type` in `{view, cart, purchase}`, `product_id` non-empty. `purchase` requires `price_cents` in `1..10,000,000`; `view`/`cart` ignore `price_cents`.
   - Sessionize per person across all linked IDs sorted by `ts`: new session iff `ts - prev_ts > 1,800,000 ms` (30 min). Equal timestamps share a session (`E16`).
   - Session day = UTC date of session start (`floor_div_day`); all events/revenue in a midnight-crossing session count on the start day (`E3`).
   - Categories from `products.csv`; missing products map to `"unknown"` (`E9`). `top_category` per `(person, day)` is highest summed purchase revenue, tie-broken byte-wise (`E8`), `NULL` when `revenue_cents == 0`.

5. **Clicks & Attribution (`src/main.cpp`):**
   - Parse -> drop malformed or empty `click_id`; keep valid clicks (`user_id` non-empty, `campaign_id` in `campaigns.csv`, `ts` in click window). Note: duplicate valid click rows have identical `(click_id, user_id, campaign_id, ts)` per `rules.md`, so neither latest-click credit nor distinct-campaign assists are altered.
   - Credit each session to the person's click in `[session_start - 7 days, session_start]` with latest `ts`, tie-broken by unsigned byte-wise smallest `click_id` (`A5`); otherwise `("organic", "organic")`.
   - Assists: every distinct non-`organic` campaign with at least one valid click in `[session_start - 7 days, session_start]` assists the session once (`A11`, `A12`), including the credited campaign. A campaign with only assists on a day gets a row with `0` in the three credited columns.

6. **Outputs (`src/parquet_writer.hpp`):**
   - `/output/user_daily/*.parquet`: `(user_id: STRING, day: DATE, sessions: INT32, events: INT32, revenue_cents: INT64, top_category: STRING nullable)`
   - `/output/campaign_daily/*.parquet`: `(campaign_id: STRING, channel: STRING, day: DATE, sessions: INT32, purchasing_sessions: INT32, revenue_cents: INT64, assisted_sessions: INT32)`
   - Valid schema-complete Parquet files emitted even on 0-row outputs (`E5`, `E13`).

---

## 3. Architecture & Implemented Optimizations

### High-Level Pipeline Flow

```
manifest.json, products.csv, campaigns.csv
        │
        ▼
[Phase 1] links/*.csv ──(pread + SIMD + 32-shard parallel intern + DSU)──► LinkTable
        │
        ▼
[Phase 2] events/*.csv ─(pread + SIMD + DateLut + SWAR F3)─► 512 event_id buckets (RawEvent, with tombstones)
        │
        ▼
[Phase 3] 512 buckets ──(L3 8B DedupIdx + 16-ahead LinkTable prefetch)──► 256 person shards (SurvivingEvent)
        │
        ▼
[Phase 4] clicks/*.csv ─(pread + SIMD + 16-ahead LinkTable prefetch)────► 256 person shards (SurvivingClick)
        │
        ▼
[Phase 5] 256 person shards (LPT descending load order across NCPU workers)
            ├─ Single-pass event & click CSR grouping (0 second hash probes)
            ├─ LSD Radix sort (nev > 2048) / std::sort on 8B ShardEvent
            ├─ Sessionizer + per-day top_category -> user_daily/part-<tid>.parquet
            └─ O(S + C) sliding-window [l, r) click credit + O(1) active_camps assists
        │
        ▼
[Phase 6] Reduce thread-local campaign_daily flat/hash accumulators ─► campaign_daily/part-0.parquet
```

### Detailed Design & Optimizations Implemented

1. **Zero-Dependency C++20 Binary + Persistent Thread Pool (`src/thread_pool.hpp`):**
   - Eliminates JVM/Python/Polars/Arrow startup overhead (~50 ms total runtime on 1M events) and avoids 5× thread spawn/join overhead by keeping `NCPU - 1` persistent workers parked on a generation counter.
   - Multi-stage `Dockerfile`: `gcc:13-bookworm` -> `debian:bookworm-slim` with `-static-libstdc++ -static-libgcc`, running as `uid 1000`.
   - Compiled with baseline `-march=x86-64-v3 -mtune=cascadelake` + runtime `__builtin_cpu_supports("avx512bw")` dispatch so AVX-512BW runs on Google Cloud Cascade Lake scorers and AVX2 runs on local machines without `SIGILL`.

2. **Chunked `pread()` + `POSIX_FADV_SEQUENTIAL` I/O (`src/io_util.hpp`):**
   - Replaced `mmap()` across 100+ CSVs (which caused `510,909` minor page faults and kernel `mmap_lock` contention across 8 threads) with 2 MB `\n`-bounded `pread()` chunks into one reusable buffer per worker thread (+64 zero padding bytes at EOF for safe SIMD loads).
   - Eliminated page-fault contention and cut system CPU time and minor page faults (`510k -> 355k`).

3. **Single-Pass Runtime-Dispatched AVX-512BW / AVX2 CSV Scanner (`src/csv_parse.hpp`):**
   - Replaces ~15 `std::memchr` calls per 72-byte event line with one 64-byte SIMD load classifying `\n`, `,`, and `"` simultaneously (`_mm512_cmpeq_epi8_mask` on Cascade Lake; dual `_mm256_movemask_epi8` on AVX2).
   - Extracts field boundaries via `ctzll` / `blsr` (`any &= any - 1`) and skips per-field quote scanning when `quote_cnt == 0`.
   - **32-entry L1 `DateLut`** validates `"YYYY-MM-DD"` in two loads for dates in the 28-day span around the window (covering F1, F2, and late `ingest_ts`), falling back to `parse_iso19` outside.
   - **Branchless SWAR 13-digit F3 parser** (`parse_13d_swar`) validates and converts standard 13-digit epoch ms without division or per-byte branching.

4. **Parallel 32-Shard `LinkTable` with 16-Byte `LinkSlot` + Deferred Prefetched Lookups (`src/union_find.hpp`, `src/main.cpp`):**
   - Link CSV parsing, string interning into 32 `OverflowArena` shards (`16 B` `LinkSlot { CompactStr key; uint32_t val; }`, `key.d[11] == 0` empty sentinel), and canonical ID back-propagation all run in parallel; only pure-integer Union-Find runs single-threaded.
   - **Deferred link resolution:** Phase 2 (`events`) skips `LinkTable` completely (meaning tombstones, late/invalid events, and duplicate event copies never probe `LinkTable`). Phase 3 resolves `LinkTable` only on surviving events using **16-ahead `__builtin_prefetch`** over `wyhash64(user_id)`. Phase 4 likewise uses a 16-item mini-batch with `__builtin_prefetch` on clicks.

5. **Compact `FastPack` Memory Layout & L3-Resident Dedup (`src/main.cpp`, `src/hash_util.hpp`):**
   - When the dataset window fits in `< 49.7 days` (`<= 0xFFFFFFFE ms`) and `categories <= 255` (true for all generated and fixture datasets, with an automatic `FastPack = false` 64-bit fallback template otherwise):
     - `RawEvent`: `48 B` (storing `ts_rel` relative to `window_start`, `(cat_id << 24) | price_cents`, and `CompactStr` 12-byte inline/overflow strings).
     - `SurvivingEvent`: **`24 B`** (`CompactStr user_id`, `uint32_t ts_rel`, `uint32_t price_cat`, `uint32_t person_hash32`).
     - `SurvivingClick`: **`36 B`**.
     - `ShardEvent`: **`8 B`** (`(uint64_t(ts_rel) << 32) | price_cat`), sorted with 4-pass 8-bit **LSD radix sort** when `nev > 2048` (handles 1%-2% whale users and shared-kiosk components in `O(N)` time).
   - **512 dedup buckets** using an **8-byte `DedupIdx { uint32_t idx_plus1; uint32_t hash32; }`** hash index table (~`1.5`–`3 MB` per thread, fitting comfortably inside Xeon L3 even at 60M events); duplicate winner updates only overwrite a 4-byte index.
   - `ChunkedVec` blocks capped at `<= 64 KB` (below glibc's `128 KB` `M_MMAP_THRESHOLD`) so allocator blocks reuse heap arenas without `mmap`/`munmap` syscalls.
   - **Single-pass CSR staging in Phase 5** caches `local_pid` from Pass 1, eliminating the second hash-table probe for both events and clicks.

6. **Streaming Zero-Dependency Parquet v1 Writer (`src/parquet_writer.hpp`):**
   - Direct Thrift Compact Protocol writer (`UNCOMPRESSED`, `PLAIN`, 250k-row row groups, 1 MB stdio buffers), emitting `INT32` for integer counts (`sessions`, `events`, `purchasing_sessions`, `assisted_sessions`) and `INT64` for `revenue_cents`, plus RLE runs (`>= 8` identical values) for nullable `top_category` definition levels.
   - Reduces 20M Parquet output from `93.6 MB` to `76.2 MB`.

7. **Build-Time Profile-Guided Optimization (`Dockerfile`):**
   - Untimed Docker build compiles with `-fprofile-generate=/tmp/pgo -flto=auto`, runs on a 50k-event/15k-click/5k-link synthetic dataset matching generator distributions, and recompiles with `-fprofile-use=/tmp/pgo -fprofile-correction -flto=auto`.

---

## 4. Measured Results & Benchmarks

All tests run with `NCPU=8 MEM_GB=24` on an Intel i9-14900HX (`AVX2` path; Cascade Lake scorers additionally use the compiled-in `AVX-512BW` path):

### Correctness Verification
- **Edge-case fixtures:** `36/36` passed both natively and inside the `linux/amd64` Docker container (`--network none --read-only --tmpfs /tmp:rw,size=64m --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 --cpus 8 --memory 24g --user 1000:1000`).
- **1M official dev set (`seed 1`, `1,000,000` events, `250,000` clicks):** Exact `MATCH` against `dev-answer/` (`user_daily`: `117,038` rows; `campaign_daily`: `27,400` rows; `0` missing, `0` extra, `0` differing, `0` duplicate keys) both natively and inside Docker.
- **Unit checks:** Exhaustive byte tests for `parse_13d_swar`, `DateLut` & leap years (`2028-02-29`, `2000-02-29`, `2027-02-29`, `2100-02-29`), and SIMD CSV quoting/`\r`/long-line/EOF edge cases.

### Performance Progression (Native Linux ext4)

| Benchmark | Initial C++20 (`mmap`, uncompacted) | Optimized C++20 (`pread`+SIMD+Pack) | Optimized + PGO (`Dockerfile` binary) | Speedup |
|---|---:|---:|---:|---:|
| **1M Dev (`dev/`)** | `88.3 ms` | `56.7 ms` | — | **1.56×** |
| **20M Scale (`seed 42`, run 2)** | `1,910.1 ms` | `965.1 ms` (`958.4 ms` best) | **`871.2 ms`** (`0.91 s` wall) | **2.19×** |
| **20M `links`** | `303.0 ms` | `89.6 ms` | `94.6 ms` (`88.3 ms` r1) | **3.2×** |
| **20M `events`** | `831.2 ms` | `384.4 ms` | `346.7 ms` (`338.7 ms` r1) | **2.40×** |
| **20M `dedup`** | `302.5 ms` | `218.9 ms` | `200.1 ms` | **1.51×** |
| **20M `clicks`** | `198.3 ms` | `83.3 ms` | `85.4 ms` (`81.7 ms` r1) | **2.32×** |
| **20M `shards`** | `271.3 ms` | `156.3 ms` | `142.7 ms` | **1.90×** |
| **20M Peak RSS** | `2,959,948 KB` (`2.82 GB`) | `1,384,792 KB` (`1.32 GB`) | `1,419,952 KB` (`1.35 GB`) | **2.08× less** |
| **20M CPU Utilization** | `700%` (`11.91 s` user) | `714%` (`5.84 s` user) | **`765%`** (`5.97 s` user) | +65% core scaling |
| **20M Parquet Output** | `93.6 MB` | `76.2 MB` | `76.2 MB` | **19% smaller** |

*(Also measured inside Docker Desktop over Windows bind mount on 1M `dev`: internal runtime `778.8 ms` -> `623.6 ms`.)*

---

## 5. Remaining Scope for Further Improvement (Ideas for Next Agent)

1. **Duplicate `click_id` Safeguard vs Rules (`src/main.cpp`):**
   - Currently Phase 4 does not run a global `click_id` dedup table because `rules.md` specifies duplicate clicks have identical values (`click_id, user_id, campaign_id, ts`). However, on a person with duplicated clicks, we sort those extra 1–3% copies in Phase 5; if any future adversarial dataset ever altered a duplicate click, deduplicating `(ts, click_id)` inside Phase 5 (for a single person, after sorting clicks) or hashing `click_id` would guard against it. Note: checking `click_id` uniqueness per person after sorting is essentially free (`O(nclk)` linear scan) ONLY if copies share `(user_id, ts)` — if copies differed in `user_id`, global dedup would be required, which `rules.md` explicitly states never happens ("Copies of a click have identical values").

2. **PGO Training Profile Coverage for `AVX-512BW` (`Dockerfile`):**
   - Look closely at `for_each_csv_line` in `src/csv_parse.hpp`: if the Docker build runs on a machine **without** AVX-512 (e.g., local build on Raptor Lake) or **with** AVX-512 (Google Cloud scorer), `-fprofile-generate` only trains whichever branch (`scan_csv_avx512` vs `scan_csv_avx2`) the build host supports. Since the scorer builds the Dockerfile **on the `n2` Cascade Lake machine itself**, it automatically profiles `scan_csv_avx512` on the scorer! Still, adding `-fprofile-values` or checking `g++` vs `clang++` (`clang-18` with ThinLTO + PGO + `libmimalloc`) is worth benchmarking.

3. **Eliminate `raw_ev_buckets` Intermediate `ChunkedVec` Copy in Phase 3 (`dedup` = 200 ms):**
   - Currently Phase 3 copies `raw_ev_buckets[*][b]` into `flat_evs` (`200 ms` total for dedup) before indexing `DedupIdx`. Instead, we could store a small `(thread_id, block_idx)` pointer array or flatten directly via `memcpy` per block (`memcpy` is already fast, so `200 ms` is mostly `wyhash64(user_id)` + `LinkTable` lookups + writing `ev_person_buckets`). Splitting `wyhash64` + `LinkTable` prefetch into `dedup` or tuning `kDedupBuckets = 1024` (halving `flat_evs` + `idx_table` per bucket to ~1.5 MB so it stays strictly inside L2/L3 cache slices per core) could shave another ~30–50 ms.

4. **Overlapping `pread()` I/O & Compute or `io_uring` / `readahead`:**
   - Currently `events` takes `338–347 ms` on 20M (~39% of total runtime), reading `1.8 GB` across `24`–`160` files. Calling `posix_fadvise(..., POSIX_FADV_WILLNEED)` across the first few chunks (or opening/advising files in parallel) and increasing `kChunkBytes` vs `1 MB`–`2 MB` (`512 KB` chunks improve tail load-balancing when a Zipf-sized largest file holds ~20% of rows) is worth testing.

5. **Fused Two-Vector Parsing in `events` + Fast `product_map` (`events` = 340 ms):**
   - In `events`, 97% of rows are `view` or `cart` (already skipping `product_map`), and `f[4]` is checked with `std::string_view::operator==`. Checking `f[4].size() == 4 && memcmp == "view"` or switching on `f[4].size()` (`4`, `8`) avoids redundant checks.
   - `product_map` (`50,000` products, only queried on the 3% `purchase` rows!) is currently ~2 MB (`FlatStrMap<uint32_t>`). Replacing `product_map` with a 16-byte `CompactStr` slot table (`~1.5 MB`) shrinks it inside L3 cache.
   - Also check `campaign_map` (`2,000` campaigns): fits in L1/L2 (~64 KB) if packed into a 16-byte slot table instead of `FlatStrMap` (`OverflowArena` indirection).

6. **Parallelize `campaign_daily` Write / Eliminate `UserDailyParquetWriter` `Buf` Copies:**
   - Writing `user_id` strings into `UserDailyParquetWriter` calls `Buf::le32` + `Buf::bytes` (`resize` + `memcpy`). Using a direct raw cursor `uint8_t* ptr` in a preallocated 8 MB row-group buffer removes `std::vector::resize` bounds checks on every row.
   - `user_daily` currently writes `UNCOMPRESSED` Parquet (fastest CPU, ~75 MB at 20M, ~225 MB at 60M), which is optimal on local NVMe/tmpfs scratch disk, though `LZ4_RAW` / `SNAPPY` shouldn't be needed since disk limit is 50 GB.

7. **Memory Allocator (`mimalloc` / `jemalloc`):**
   - Linking `-lmimalloc` (available via Debian `libmimalloc-dev` or header-only build) can speed up `ChunkedVec` / `std::vector` block allocations and cross-thread frees (`raw_ev_buckets` allocated on thread `t`, freed on thread `b` in Phase 3, and `ev_person_buckets` freed across threads in Phase 5!). Cross-thread `delete[]` in glibc `ptmalloc2` triggers arena locking; `mimalloc` handles cross-thread frees lock-free!

8. **Verify `60M`-Event Final Scale (`~5.4 GB` CSVs):**
   - At 60M events (`15M` clicks), peak RSS is projected at ~`4.1 GB` (well below `24 GB`), and projected runtime on Cascade Lake is ~`3–5 s` (well below the `180 s` cap). Worth running one local 60M-event test (`generate --seed 7 --events 60000000 --clicks 15000000`) to confirm peak RSS and shard load balance (especially for the 3–6 shared-kiosk components!).
