# Shopping Sessions — High-Performance C++20 Architecture & Optimization Log (`prompt-gods`)

## 1. Scorer Environment & Constraints
- **Target Hardware:** Google Cloud `n2` Intel Xeon Cascade Lake+ (`linux/amd64`, `AVX-512BW` + `AVX-512VL` + `BMI2` `_pext_u64`), `NCPU=8`, `MEM_GB=24`.
- **Container & Sandbox:** `debian:bookworm-slim` (built from `gcc:13-bookworm`), `--network none`, `--read-only`, `--tmpfs /tmp:rw,size=64m`, `--user 1000:1000`, `/data` read-only, `/output` writable.
- **Scale:** Practice ~20M events / 5M clicks; Final ~60M events / 15M clicks (180 s cap).
- **Correctness Requirement:** 100% bit-exact match across all 36 edge-case fixtures (`A1`–`J4`), 1M `dev/`, and 20M `d20m` in both `FastPack=true` and `FastPack=false` (`SS_FORCE_SLOW_PACK=1`) modes.

---

## 2. Root-Cause Diagnosis of Baseline Bottlenecks
1. **Un-inlined SIMD Delimiter & Row Callbacks (`csv_parse.hpp`):**
   - `csv_simd::delimiters_avx512` lacked matching target attributes on callers, causing ~65M out-of-line function calls and `vzeroupper` penalties.
   - Passing a single multi-call lambda (`on_row`) to both the hot SIMD path and the cold `split_csv_line` fallback caused GCC `-flto` to emit an out-of-line `call` on every row (27.5M calls at 20M scale) and a stack canary (`__stack_chk_fail`) for `tail_buf[256]`.
2. **PGO Mis-training (`Dockerfile`):**
   - The synthetic PGO generator previously emitted short numeric IDs (`e123`, `u456`), mis-training branch predictors away from the generator's real 9-byte base36 scrambled IDs (`e` + 8 base36 chars, `u` + 8 base36 chars, `k` + 8 base36 chars) and 20% whole-field quoting.
3. **Kernel `exit_mmap()` Teardown & Page Faults (`io_util.hpp`):**
   - Mapping ~1.86 GB of CSV files via `mmap` faulted ~476,000 4 KB PTEs and incurred `43–48 ms` of single-threaded kernel `_exit(0)` teardown on warm runs and a 4x slowdown on cold runs.
   - Switching to parallel `256 KB` `pread` chunks with 192-byte zero padding keeps buffers L2-resident (`66.6 GB/s` in microbenchmarks), eliminates EOF `SafeMargin`/`tail_buf[256]` branches, and drops `_exit(0)` teardown overhead from `~47 ms` to `~13.6 ms`.
4. **Memory Footprint & Cache Misses in Phases 1, A, B, and C:**
   - Shrinking `FastId64<Prefix>` to 8 bytes (inline `<Prefix><1..8 ASCII bytes>` with `__builtin_clzll` length and `__builtin_bswap64` byte-wise ordering, overflow bit 63 into length-prefixed `OverflowArena`) reduced struct sizes across the entire pipeline:
     - `RawEvent`: `32 B`
     - `SurvivingEvent`: `16 B`
     - `SurvivingClick`: `24 B`
     - `ShardEvent`: `8 B`
     - `ShardClick`: `16 B`
     - `PersonMapSlot`: `16 B`
   - Per-thread lock-free `ThreadBlockArena` (64 KB blocks backed by 2 MB THP-aligned slabs with zero-fault block recycling between phases) eliminated `std::vector` geometric reallocations.

---

## 3. Pipeline Architecture (Pass 1 + Pass 2)

### Startup & Metadata (`src/main.cpp`, `src/csv_parse.hpp`, `src/io_util.hpp`)
- Overlaps `load_products()` (`products.csv` -> `fast_prod_by_num` `uint8_t` direct index table for `p<num>` + `FlatStrMap` fallback), `load_campaigns()` (`campaigns.csv` -> `fast_camp_by_num` direct index table for `c_<num>` + `FlatStrMap` fallback), and parallel `open_one_pread_file()` across `events/*.csv` and `clicks/*.csv`.
- Separate `FastDateLut` tables (`ev_date_lut` covering 24 days and `clk_date_lut` covering 25 days with `hi8 = 1` empty sentinels to prevent 8-byte prefix collisions across month boundaries).
- `DT_REG` fast path in `list_csv_files()` avoids redundant `stat()` calls.

### Phase 1: 64-Shard Parallel `LinkTable` (`src/union_find.hpp`)
- **Sub-phase 1:** Parallel `256 KB` `pread` + AVX-512/AVX2 2-column CSV scan into 64 hash-partitioned `StrItem` (`16 B`) block slices.
- **Sub-phase 2:** Parallel 64-shard L2-resident string interning (`InternSlot` `16 B`).
- **Sub-phase 3:** Wait-free parallel Union-Find (halving path compression, linking larger root to smaller root over 4,096-edge chunks) + parallel atomic `min_node` resolution using `UserId64::less`.
- **Sub-phase 4:** Prunes identity roots (`c == i`, keeping `816,130` of `1,196,113` nodes at 20M) and populates 64 independent power-of-2 sub-tables (`16.0 MB` `LinkSlot`) and 64 independent 3-bit sub-Blooms (`2.0 MB`) in parallel by shard with **zero atomics/CAS**.
- Provides 2-stage software-prefetched lookups (`prefetch_bloom` -> `may_contain` -> `prefetch_slot` -> `resolve_with_hash` / `resolve_sv_with_hash`).

### Phase 2 (Phase A): Parallel Events Scan (`src/main.cpp`, `src/csv_parse.hpp`)
- Parallel `256 KB` `pread` chunks with 192-byte zero padding scanned via `scan_pread_buf_avx512<7>` / `scan_pread_buf_avx2<7>`.
- Separate single-call-site `handle_fast` (`__attribute__((always_inline))`) and out-of-line `scan_slow_line` eliminate per-row function calls in the hot loop.
- Branchless `_pext_u64` quote extraction (`extract_fields_64`), SWAR timestamp parsing (`parse_ts_f1_sec` + `parse_ts_any_with_hint` reusing the `ingest_ts` date prefix for `ts`), and L1-resident per-bucket block pointers (`RawEv* cur[512]`, `endp[512]`).
- Deduplication key (`event_id`, `ingest_rel_sec`) is recorded for every row with valid `event_id` and `ingest_ts` **before** event validation, preserving exact pre-validation deduplication semantics (`rules.md`).

### Phase B: Overlapped Deduplication + Clicks Scan (`src/main.cpp`)
- Dedicated click starter threads scan `clicks/*.csv` (`scan_pread_buf_avx512<4>` / `scan_pread_buf_avx2<4>`) with 32-item batched 2-stage Bloom + `LinkTable` prefetching while remaining threads drain `dedup_buckets`.
- Each dedup bucket uses a compact `DedupIdx` (`8 B`: `idx_plus1`, `hash32`) table at 65% max load (`1.0 MB` per bucket at 20M, L2-resident) with 16-ahead software prefetching, followed by 32-item batched 2-stage Bloom + `LinkTable` resolution and scatter into `person_shards` (`SurvivingEvent` `16 B`), immediately recycling `RawEvent` 64 KB blocks back to `ThreadBlockArena`.

### Phase C: LPT Person Shards, Branchless Sorting Networks & Attribution (`src/main.cpp`, `src/sort_u64.hpp`)
- Shards are scheduled in Longest-Processing-Time (LPT) descending order.
- Per-person counting sort groups events and clicks into contiguous per-person slices.
- **`sort_u64_hi31` (`src/sort_u64.hpp`):**
  - `n <= 16`: Compile-time unrolled branchless Bose-Nelson sorting networks (`cswap_u64` -> `cmp` + `cmov`).
  - `17 <= n <= 32`: `bn_sort<0, 16>` + `sort_u64_le16` + branchless 2-way merge.
  - `33 <= n <= 255`: Branchless Edelkamp-Weiss BlockQuicksort (`16`-element offset buffers) with sorting network base cases.
  - `n >= 256`: Single-scan histogram + 3-pass (`11/11/9`-bit) LSD radix sort on 31-bit relative timestamps with uniform-bucket pass skipping.
- **Sessionization & `user_daily`:** Handles event 0 outside the loop; tracks UTC day transitions via relative millisecond thresholds (`next_day_rel += 86400000LL`) with zero 64-bit divisions in `FastPack`; writes `user_daily/part-<tid>.parquet` directly via `FastByteCursor::write_fast_id64` and zero-copy category slices (`src/parquet_writer.hpp`).
- **Click Attribution & Assisted Sessions:**
  - Packs per-person clicks (`FastPack && nclk <= 65535 && num_camps <= 65535`) into 8-byte `uint64_t` keys `(ts_rel << 32) | (camp_id << 16) | idx`, sorting via `sort_u64_hi31` and only invoking `ClickId64::less` on equal-timestamp runs.
  - For `nclk <= 16` (87.7% of clicking persons), maintains active campaigns in a 16-entry L1 stack array (`SmallAct act[16]`); for `nclk > 16`, uses a packed 8-byte `CampState` (`count`, `enter_s`, `active_pos`) vector.
  - Accumulates assisted sessions in `O(1)` per click via day-level interval differencing (`K - enter_s`) into transposed thread-local tables `[day_idx * num_camps + camp_id]`.

### Phase D: Transposed Campaign Reduction (`src/main.cpp`, `src/parquet_writer.hpp`)
- Reduces transposed `flat_credits` and `flat_assists` across threads and writes `campaign_daily/part-0.parquet`.

---

## 4. Verification & Benchmarks

### Correctness Verification
- **Unit Tests:** `sort_u64` and `sort_u64_hi31` exhaustively verified for all lengths `n = 0..512` across 7 distributions (uniform random, reverse, sorted, constant, 0/1, mod-3, and `0`/`UINT64_MAX` extremes).
- **36/36 Official Fixtures (`A1`–`J4`):**
  - `FastPack=true`: `36/36 PASSED`
  - `FastPack=false` (`SS_FORCE_SLOW_PACK=1`): `36/36 PASSED`
- **1M `dev/` Dataset vs `dev-answer/`:** `MATCH` (`117,038` `user_daily` rows, `27,400` `campaign_daily` rows, 0 differences).
- **20M `d20m` Dataset (`20,000,000` events, `5,127,892` clicks, `400,000` links) vs Baseline:** `MATCH` (`2,246,306` `user_daily` rows, `28,014` `campaign_daily` rows, `3,407,788` total sessions, `6,680,740,085` revenue cents, `36,030,031` assisted sessions, 0 differences).

### 20M Benchmark Progression (`NCPU=8`, Warm Page Cache)
| Version | Internal `done in` | External `py_wall` | Phase Breakdown (`init / links / events / dedup+clk / shards / camp`) |
| :--- | :---: | :---: | :--- |
| Baseline (`git HEAD`) | `665.2 ms` | `~710 ms` | `links 113.0, events 246.7, dedup+clk 144.6, shards 158.4, camp 2.5` |
| Pass 1 (`FastId64` + 64-shard `LinkTable` + `ThreadBlockArena` + `mmap`) | `452.2 ms` | `500.0 ms` | `init 16.5, links 44.1, events 151.5, dedup+clk 121.0, shards 117.4, camp 1.7` |
| Pass 1 + PGO | `426.9 ms` (`441.3 ms` `taskset -c 0-7`) | `470.0 ms` (`486.0 ms` `taskset -c 0-7`) | `init 16.8, links 42.1, events 142.3, dedup+clk 113.8, shards 110.2, camp 1.7` |
| Pass 2 (`256 KB` `pread` + lock-free 64-subtable `LinkTable` + `sort_u64` networks/BlockQSort + L1 active set) | `474.0–475.0 ms` (`485.6 ms` PGO) | `491.2 ms` (`494.7 ms` PGO; `_exit` overhead cut from `47.8 ms` to `8.0–9.1 ms`; `shards` cut to `100.8–102.3 ms`; `links` cut to `41.2–41.6 ms`) | `init 17.6, links 43.1, events 160.6, dedup+clk 138.4, shards 113.5, camp 1.9` |
