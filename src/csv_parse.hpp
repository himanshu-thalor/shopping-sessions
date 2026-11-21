#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include "io_util.hpp"

#define SS_AVX512_ATTR __attribute__((target("avx512f,avx512bw,avx512vl,bmi2")))

[[gnu::always_inline]] inline bool is_ascii_digit(char c) noexcept {
    return static_cast<unsigned char>(c - '0') <= 9;
}

[[gnu::always_inline]] inline bool parse_field(const char* p, size_t len,
                                               std::string_view& out) noexcept {
    if (len == 0) {
        out = {};
        return true;
    }
    if (p[0] == '"') {
        if (len < 2 || p[len - 1] != '"') return false;
        if (len > 2 && std::memchr(p + 1, '"', len - 2) != nullptr) return false;
        out = std::string_view(p + 1, len - 2);
        return true;
    }
    if (std::memchr(p, '"', len) != nullptr) return false;
    out = std::string_view(p, len);
    return true;
}

template <size_t N>
[[gnu::always_inline]] inline bool split_csv_line(const char* line, size_t len,
                                                  std::string_view (&fields)[N]) noexcept {
    if (len == 0) return false;
    const char* cur = line;
    size_t rem = len;
    for (size_t i = 0; i + 1 < N; ++i) {
        const char* comma = static_cast<const char*>(std::memchr(cur, ',', rem));
        if (!comma) return false;
        const size_t flen = static_cast<size_t>(comma - cur);
        if (!parse_field(cur, flen, fields[i])) return false;
        cur = comma + 1;
        rem -= flen + 1;
    }
    if (std::memchr(cur, ',', rem) != nullptr) return false;
    return parse_field(cur, rem, fields[N - 1]);
}

inline bool is_leap_year(int64_t year) noexcept {
    return (year % 4 == 0) && (year % 100 != 0 || year % 400 == 0);
}

inline int64_t days_in_month(int64_t year, int64_t month) noexcept {
    static constexpr int8_t kDays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month == 2 && is_leap_year(year)) return 29;
    return kDays[month - 1];
}

inline int64_t days_from_civil(int64_t y, int64_t m, int64_t d) noexcept {
    y -= (m <= 2);
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const int64_t yoe = y - era * 400;
    const int64_t doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

inline void civil_from_days(int64_t days, int64_t& y, int64_t& m, int64_t& d) noexcept {
    const int64_t z = days + 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const uint64_t doe = static_cast<uint64_t>(z - era * 146097);
    const uint64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    y = static_cast<int64_t>(yoe) + era * 400;
    const uint64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const uint64_t mp = (5 * doy + 2) / 153;
    d = static_cast<int64_t>(doy - (153 * mp + 2) / 5 + 1);
    m = static_cast<int64_t>(mp < 10 ? mp + 3 : mp - 9);
    y += (m <= 2);
}

inline int32_t floor_div_day(int64_t ms) noexcept {
    static constexpr int64_t kDayMs = 86400000LL;
    int64_t d = ms / kDayMs;
    if (ms < 0 && ms % kDayMs != 0) --d;
    return static_cast<int32_t>(d);
}

[[gnu::always_inline]] inline bool parse_2d(const char* p, int64_t& out) noexcept {
    const uint8_t d0 = static_cast<uint8_t>(p[0] - '0');
    const uint8_t d1 = static_cast<uint8_t>(p[1] - '0');
    if (d0 > 9 || d1 > 9) return false;
    out = d0 * 10 + d1;
    return true;
}

[[gnu::always_inline]] inline bool parse_4d(const char* p, int64_t& out) noexcept {
    const uint8_t d0 = static_cast<uint8_t>(p[0] - '0');
    const uint8_t d1 = static_cast<uint8_t>(p[1] - '0');
    const uint8_t d2 = static_cast<uint8_t>(p[2] - '0');
    const uint8_t d3 = static_cast<uint8_t>(p[3] - '0');
    if (d0 > 9 || d1 > 9 || d2 > 9 || d3 > 9) return false;
    out = d0 * 1000 + d1 * 100 + d2 * 10 + d3;
    return true;
}

[[gnu::always_inline]] inline int32_t parse_hms_swar(const char* s) noexcept {
    uint64_t w;
    std::memcpy(&w, s, 8);
    if (__builtin_expect((w & 0x0000FF0000FF0000ULL) != 0x00003A00003A0000ULL, 0)) return -1;
    const uint64_t d = w - 0x30303A30303A3030ULL;
    const uint64_t bad = (d | (d + 0x7676007676007676ULL)) & 0x8080008080008080ULL;
    if (__builtin_expect(bad != 0, 0)) return -1;
    const uint32_t hh = static_cast<uint32_t>((d & 0xFFu) * 10u + ((d >> 8) & 0xFFu));
    const uint32_t mm = static_cast<uint32_t>(((d >> 24) & 0xFFu) * 10u + ((d >> 32) & 0xFFu));
    const uint32_t ss = static_cast<uint32_t>(((d >> 48) & 0xFFu) * 10u + ((d >> 56) & 0xFFu));
    if (__builtin_expect((hh > 23u) | (mm > 59u) | (ss > 59u), 0)) return -1;
    return static_cast<int32_t>(hh * 3600u + mm * 60u + ss);
}

struct FastDateLut {
    struct Entry {
        uint64_t hi8 = 1;  // 1 == empty sentinel (never matches ASCII)
        int64_t midnight_ms = 0;
        int32_t midnight_rel_sec = 0;
        uint16_t lo2 = 0;
    };
    std::array<Entry, 32> entries{};
    bool exact_rel_sec = false;

    void build(int64_t start_ms, int64_t count_days, int64_t win_start_ms) noexcept {
        entries.fill(Entry{});
        exact_rel_sec = (win_start_ms % 1000LL == 0);
        const int64_t d0 = floor_div_day(start_ms);
        const int64_t count = std::min<int64_t>(28, count_days);
        for (int64_t i = 0; i < count; ++i) {
            const int64_t day_num = d0 + i;
            int64_t y = 0, m = 0, d = 0;
            civil_from_days(day_num, y, m, d);
            if (y < 0 || y > 9999 || d < 1 || d > 31) continue;
            char buf[10];
            buf[0] = static_cast<char>('0' + (y / 1000) % 10);
            buf[1] = static_cast<char>('0' + (y / 100) % 10);
            buf[2] = static_cast<char>('0' + (y / 10) % 10);
            buf[3] = static_cast<char>('0' + y % 10);
            buf[4] = '-';
            buf[5] = static_cast<char>('0' + m / 10);
            buf[6] = static_cast<char>('0' + m % 10);
            buf[7] = '-';
            buf[8] = static_cast<char>('0' + d / 10);
            buf[9] = static_cast<char>('0' + d % 10);
            const size_t idx =
                (static_cast<uint8_t>(buf[8]) * 10u + static_cast<uint8_t>(buf[9])) & 31u;
            Entry& e = entries[idx];
            if (e.hi8 != 1) continue;
            std::memcpy(&e.hi8, buf, 8);
            std::memcpy(&e.lo2, buf + 8, 2);
            e.midnight_ms = day_num * 86400000LL;
            const int64_t rsec = (e.midnight_ms - win_start_ms) / 1000LL;
            if (rsec < std::numeric_limits<int32_t>::min() + 86400 ||
                rsec > std::numeric_limits<int32_t>::max() - 86400) {
                exact_rel_sec = false;
            }
            e.midnight_rel_sec = static_cast<int32_t>(rsec);
        }
    }
};

inline bool parse_iso19(const char* s, int64_t& out_utc_ms) noexcept {
    if (s[4] != '-' || s[7] != '-' || s[10] != 'T' || s[13] != ':' || s[16] != ':') {
        return false;
    }
    int64_t year, month, day, hour, minute, second;
    if (!parse_4d(s, year) || !parse_2d(s + 5, month) || !parse_2d(s + 8, day) ||
        !parse_2d(s + 11, hour) || !parse_2d(s + 14, minute) || !parse_2d(s + 17, second)) {
        return false;
    }
    if (month < 1 || month > 12) return false;
    if (day < 1 || day > days_in_month(year, month)) return false;
    if (hour > 23 || minute > 59 || second > 59) return false;
    const int64_t days = days_from_civil(year, month, day);
    out_utc_ms = days * 86400000LL + (hour * 3600LL + minute * 60LL + second) * 1000LL;
    return true;
}

[[gnu::always_inline]] inline bool parse_iso19_swar(const char* s, const FastDateLut& lut,
                                                    int64_t& out_utc_ms) noexcept {
    if (__builtin_expect(s[10] != 'T', 0)) return false;
    const size_t idx = (static_cast<uint8_t>(s[8]) * 10u + static_cast<uint8_t>(s[9])) & 31u;
    const FastDateLut::Entry& e = lut.entries[idx];
    uint64_t hi8;
    uint16_t lo2;
    std::memcpy(&hi8, s, 8);
    std::memcpy(&lo2, s + 8, 2);
    if (__builtin_expect(e.hi8 == hi8 && e.lo2 == lo2, 1)) {
        const int32_t hms = parse_hms_swar(s + 11);
        if (__builtin_expect(hms < 0, 0)) return false;
        out_utc_ms = e.midnight_ms + static_cast<int64_t>(hms) * 1000LL;
        return true;
    }
    return parse_iso19(s, out_utc_ms);
}

inline bool parse_ts_f1(std::string_view sv, int64_t& out_ms) noexcept {
    if (sv.size() != 20 || sv[19] != 'Z') return false;
    return parse_iso19(sv.data(), out_ms);
}

[[gnu::always_inline]] inline int32_t clamp_rel_sec(int64_t ms, int64_t win_start_ms) noexcept {
    int64_t sec = (ms - win_start_ms) / 1000LL;
    if (sec < std::numeric_limits<int32_t>::min()) return std::numeric_limits<int32_t>::min();
    if (sec > std::numeric_limits<int32_t>::max()) return std::numeric_limits<int32_t>::max();
    return static_cast<int32_t>(sec);
}

[[gnu::always_inline]] inline bool parse_ts_f1_sec(std::string_view sv, const FastDateLut& lut,
                                                   int64_t win_start_ms, int64_t& out_ms,
                                                   int32_t& out_rel_sec, uint64_t& out_hi8,
                                                   uint16_t& out_lo2, int64_t& out_midnight_ms,
                                                   bool& out_lut_hit) noexcept {
    if (__builtin_expect(sv.size() != 20, 0)) return false;
    const char* s = sv.data();
    if (__builtin_expect(s[10] != 'T' || s[19] != 'Z', 0)) return false;
    const size_t idx = (static_cast<uint8_t>(s[8]) * 10u + static_cast<uint8_t>(s[9])) & 31u;
    const FastDateLut::Entry& e = lut.entries[idx];
    uint64_t hi8;
    uint16_t lo2;
    std::memcpy(&hi8, s, 8);
    std::memcpy(&lo2, s + 8, 2);
    if (__builtin_expect(e.hi8 == hi8 && e.lo2 == lo2, 1)) {
        const int32_t hms = parse_hms_swar(s + 11);
        if (__builtin_expect(hms < 0, 0)) return false;
        out_ms = e.midnight_ms + static_cast<int64_t>(hms) * 1000LL;
        out_rel_sec = lut.exact_rel_sec ? (e.midnight_rel_sec + hms)
                                        : clamp_rel_sec(out_ms, win_start_ms);
        out_hi8 = hi8;
        out_lo2 = lo2;
        out_midnight_ms = e.midnight_ms;
        out_lut_hit = true;
        return true;
    }
    if (!parse_iso19(s, out_ms)) return false;
    out_rel_sec = clamp_rel_sec(out_ms, win_start_ms);
    out_lut_hit = false;
    return true;
}

[[gnu::always_inline]] inline bool parse_13d_swar(const char* s, int64_t& out_ms) noexcept {
    uint64_t w0;
    uint32_t w1;
    std::memcpy(&w0, s, 8);
    std::memcpy(&w1, s + 8, 4);
    const uint64_t bad0 =
        (w0 | (w0 + 0x4646464646464646ULL) | (w0 - 0x3030303030303030ULL)) &
        0x8080808080808080ULL;
    const uint64_t w1u = static_cast<uint64_t>(w1);
    const uint64_t bad1 =
        (w1u | (w1u + 0x46464646ULL) | (w1u - 0x30303030ULL)) & 0x80808080ULL;
    const uint8_t last_d = static_cast<uint8_t>(s[12] - '0');
    if (__builtin_expect((bad0 | bad1) != 0 || last_d > 9, 0)) return false;

    uint64_t d0 = w0 - 0x3030303030303030ULL;
    d0 = ((d0 * 10 + (d0 >> 8)) & 0x00FF00FF00FF00FFULL);
    d0 = ((d0 * 100 + (d0 >> 16)) & 0x0000FFFF0000FFFFULL);
    d0 = ((d0 * 10000 + (d0 >> 32)) & 0x00000000FFFFFFFFULL);

    uint64_t d1 = w1u - 0x30303030ULL;
    d1 = ((d1 * 10 + (d1 >> 8)) & 0x00FF00FFULL);
    d1 = ((d1 * 100 + (d1 >> 16)) & 0x0000FFFFULL);

    out_ms = static_cast<int64_t>(d0 * 100000ULL + d1 * 10ULL + last_d);
    return true;
}

[[gnu::noinline]] inline bool parse_ts_f3_slow(const char* s, size_t n, int64_t& out_ms) noexcept {
    if (n == 0) return false;
    uint64_t val = 0;
    constexpr uint64_t max_i64 = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    bool overflow = false;
    for (size_t i = 0; i < n; ++i) {
        if (!is_ascii_digit(s[i])) return false;
        const uint64_t d = static_cast<uint64_t>(s[i] - '0');
        if (!overflow) {
            if (val > (max_i64 - d) / 10) {
                overflow = true;
            } else {
                val = val * 10 + d;
            }
        }
    }
    if (overflow) return false;
    out_ms = static_cast<int64_t>(val);
    return true;
}

[[gnu::always_inline]] inline bool parse_ts_any_with_hint(std::string_view sv,
                                                          const FastDateLut& lut,
                                                          bool hint_valid, uint64_t hint_hi8,
                                                          uint16_t hint_lo2,
                                                          int64_t hint_midnight_ms,
                                                          int64_t& out_ms) noexcept {
    const size_t n = sv.size();
    const char* s = sv.data();
    if (n == 20 && s[19] == 'Z' && s[10] == 'T') {
        uint64_t hi8;
        uint16_t lo2;
        std::memcpy(&hi8, s, 8);
        std::memcpy(&lo2, s + 8, 2);
        if (__builtin_expect(hint_valid && hi8 == hint_hi8 && lo2 == hint_lo2, 1)) {
            const int32_t hms = parse_hms_swar(s + 11);
            if (__builtin_expect(hms < 0, 0)) return false;
            out_ms = hint_midnight_ms + static_cast<int64_t>(hms) * 1000LL;
            return true;
        }
        return parse_iso19_swar(s, lut, out_ms);
    }
    if (n == 13) {
        return parse_13d_swar(s, out_ms);
    }
    if (n == 25 && (s[19] == '+' || s[19] == '-') && s[22] == ':' && s[10] == 'T') {
        int64_t local_ms = 0;
        if (!parse_iso19_swar(s, lut, local_ms)) return false;
        int64_t off_h = 0, off_m = 0;
        if (!parse_2d(s + 20, off_h) || !parse_2d(s + 23, off_m)) return false;
        if (off_h > 23 || off_m > 59) return false;
        const int64_t off_ms = (off_h * 60LL + off_m) * 60000LL;
        out_ms = (s[19] == '+') ? (local_ms - off_ms) : (local_ms + off_ms);
        return true;
    }
    return parse_ts_f3_slow(s, n, out_ms);
}

[[gnu::always_inline]] inline bool parse_ts_any_swar(std::string_view sv, const FastDateLut& lut,
                                                     int64_t& out_ms) noexcept {
    return parse_ts_any_with_hint(sv, lut, false, 0, 0, 0, out_ms);
}

inline bool parse_ts_any(std::string_view sv, int64_t& out_ms) noexcept {
    static constexpr FastDateLut kEmptyLut{};
    return parse_ts_any_with_hint(sv, kEmptyLut, false, 0, 0, 0, out_ms);
}

[[gnu::always_inline]] inline bool parse_price_cents(std::string_view sv,
                                                     uint32_t& out_price) noexcept {
    const size_t n = sv.size();
    if (__builtin_expect(n == 0, 0)) return false;
    const char* s = sv.data();
    uint64_t val = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t d = static_cast<uint8_t>(s[i] - '0');
        if (__builtin_expect(d > 9, 0)) return false;
        val = val * 10 + d;
        if (__builtin_expect(val > 10000000ULL, 0)) return false;
    }
    if (__builtin_expect(val == 0, 0)) return false;
    out_price = static_cast<uint32_t>(val);
    return true;
}

namespace csv_simd {

struct Delimiters {
    uint64_t commas;
    uint64_t newlines;
    uint64_t quotes;
};

[[gnu::always_inline]] inline uint64_t match64(__m256i low, __m256i high, char needle) noexcept {
    const __m256i target = _mm256_set1_epi8(needle);
    const uint32_t a = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(low, target)));
    const uint32_t b = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(high, target)));
    return a | (static_cast<uint64_t>(b) << 32);
}

[[gnu::always_inline]] inline Delimiters delimiters_avx2(const char* p) noexcept {
    const __m256i low = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    const __m256i high = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32));
    return {match64(low, high, ','), match64(low, high, '\n'), match64(low, high, '"')};
}

SS_AVX512_ATTR [[gnu::always_inline]] inline Delimiters delimiters_avx512(const char* p) noexcept {
    const __m512i v = _mm512_loadu_si512(p);
    return {_mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(',')),
            _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8('\n')),
            _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8('"'))};
}

// Branchless quote/field extraction from 64-byte delimiter masks.
// Returns bytes consumed up through Kth separator, or 0 to fall back.
template <size_t K, bool EndOfLine>
[[gnu::always_inline]] inline size_t extract_fields_64(const Delimiters& masks, const char* p,
                                                       std::string_view* fields) noexcept {
    const uint64_t separators = masks.commas | masks.newlines;
    uint64_t rest = separators;
    for (size_t i = 1; i < K; ++i) rest &= rest - 1;
    if (__builtin_expect(!rest, 0)) return 0;
    const uint64_t last = rest & -rest;
    const uint64_t through_last = last | (last - 1);
    if (__builtin_expect((masks.newlines & through_last) != (EndOfLine ? last : 0), 0)) return 0;
    uint64_t ends = separators & through_last;
    const uint64_t quotes = masks.quotes & through_last;
    const uint64_t starts = ((ends << 1) | 1) & through_last;
    const uint64_t opening = quotes & starts;
    const uint64_t closing = (quotes << 1) & ends;
    const unsigned quoted = static_cast<unsigned>(_pext_u64(opening, starts));
    if (__builtin_expect((opening ^ (closing >> 1)) != quotes ||
                             quoted != _pext_u64(closing, ends),
                         0)) {
        return 0;
    }
    size_t start = 0;
    for (size_t i = 0; i < K; ++i) {
        const size_t end = static_cast<size_t>(__builtin_ctzll(ends));
        const size_t trim = (quoted >> i) & 1u;
        fields[i] = std::string_view(p + start + trim, end - start - 2 * trim);
        ends &= ends - 1;
        start = end + 1;
    }
    return start;
}

template <size_t K, bool EndOfLine>
[[gnu::always_inline]] inline size_t read_fields_avx2(const char* p,
                                                      std::string_view* fields) noexcept {
    return extract_fields_64<K, EndOfLine>(delimiters_avx2(p), p, fields);
}

template <size_t K, bool EndOfLine>
SS_AVX512_ATTR [[gnu::always_inline]] inline size_t read_fields_avx512(
    const char* p, std::string_view* fields) noexcept {
    return extract_fields_64<K, EndOfLine>(delimiters_avx512(p), p, fields);
}

// Fast 32-byte single-load AVX2 path for 2-column links CSVs.
[[gnu::always_inline]] inline size_t read_fields2_32_avx2(const char* p,
                                                          std::string_view* fields) noexcept {
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    const uint32_t commas =
        static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(','))));
    const uint32_t nls =
        static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8('\n'))));
    const uint32_t seps = commas | nls;
    uint32_t rest = seps & (seps - 1u);
    if (__builtin_expect(!rest, 0)) return 0;
    const uint32_t last = rest & -rest;
    const uint32_t through_last = last | (last - 1u);
    if (__builtin_expect((nls & through_last) != last, 0)) return 0;
    const uint32_t quotes = static_cast<uint32_t>(
                                _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8('"')))) &
                            through_last;
    uint32_t ends = seps & through_last;
    const uint32_t starts = ((ends << 1) | 1u) & through_last;
    const uint32_t opening = quotes & starts;
    const uint32_t closing = (quotes << 1) & ends;
    const uint32_t quoted = _pext_u32(opening, starts);
    if (__builtin_expect((opening ^ (closing >> 1)) != quotes ||
                             quoted != _pext_u32(closing, ends),
                         0)) {
        return 0;
    }
    const size_t e0 = static_cast<size_t>(__builtin_ctz(ends));
    const size_t t0 = quoted & 1u;
    fields[0] = std::string_view(p + t0, e0 - 2 * t0);
    ends &= ends - 1u;
    const size_t e1 = static_cast<size_t>(__builtin_ctz(ends));
    const size_t t1 = (quoted >> 1) & 1u;
    fields[1] = std::string_view(p + e0 + 1 + t1, e1 - e0 - 1 - 2 * t1);
    return e1 + 1;
}

template <size_t N>
[[gnu::always_inline]] inline size_t read_row_avx2(const char* p,
                                                   std::string_view* f) noexcept {
    if constexpr (N == 2) {
        return read_fields2_32_avx2(p, f);
    } else if constexpr (N == 7) {
        const size_t first = read_fields_avx2<3, false>(p, f);
        const size_t last = first ? read_fields_avx2<4, true>(p + first, f + 3) : 0;
        return last ? (first + last) : 0;
    } else {
        return read_fields_avx2<N, true>(p, f);
    }
}

template <size_t N>
SS_AVX512_ATTR [[gnu::always_inline]] inline size_t read_row_avx512(const char* p,
                                                                    std::string_view* f) noexcept {
    if constexpr (N == 7) {
        const size_t first = read_fields_avx512<3, false>(p, f);
        const size_t last = first ? read_fields_avx512<4, true>(p + first, f + 3) : 0;
        return last ? (first + last) : 0;
    } else {
        return read_fields_avx512<N, true>(p, f);
    }
}

template <size_t N, typename SlowCb>
[[gnu::noinline]] size_t scan_slow_line(const char* p, size_t rem,
                                        SlowCb& on_slow_row) noexcept {
    const char* nl = static_cast<const char*>(std::memchr(p, '\n', rem));
    const size_t w = nl ? static_cast<size_t>(nl - p) : rem;
    std::string_view f[N];
    if (split_csv_line<N>(p, w, f)) {
        on_slow_row(f);
    }
    return w + 1;
}

#define SS_SCAN_PREAD_BODY(READ_EXPR)                            \
    const char* const buf = view.data;                           \
    size_t pos = view.pos;                                       \
    const size_t limit = view.limit;                             \
    const size_t len = view.len;                                 \
    std::string_view f[N];                                       \
    while (pos < limit) {                                        \
        const char* p = buf + pos;                               \
        const size_t used = (READ_EXPR);                         \
        if (__builtin_expect(used != 0, 1)) {                    \
            pos += used;                                         \
            on_fast_row(f);                                      \
        } else {                                                 \
            pos += scan_slow_line<N>(p, len - pos, on_slow_row); \
        }                                                        \
    }

template <size_t N, typename FastCb, typename SlowCb>
[[gnu::noinline]] void scan_pread_buf_avx2(const PreadView& view, FastCb&& on_fast_row,
                                           SlowCb&& on_slow_row) {
    SS_SCAN_PREAD_BODY(read_row_avx2<N>(p, f))
}

template <size_t N, typename FastCb, typename SlowCb>
SS_AVX512_ATTR [[gnu::noinline]] void scan_pread_buf_avx512(const PreadView& view,
                                                            FastCb&& on_fast_row,
                                                            SlowCb&& on_slow_row) {
    SS_SCAN_PREAD_BODY(read_row_avx512<N>(p, f))
}

#undef SS_SCAN_PREAD_BODY

}  // namespace csv_simd

inline bool cpu_has_avx512bw() noexcept {
    static const bool has = __builtin_cpu_supports("avx512bw") &&
                            __builtin_cpu_supports("avx512vl");
    return has;
}

inline void parse_manifest(const std::string& path, int64_t& win_start_ms,
                           int64_t& win_end_ms) {
    auto raw = read_whole_file(path);
    std::string_view text(raw.data(), raw.size());

    auto extract_ts = [&](const char* key) -> int64_t {
        std::string needle = std::string("\"") + key + "\"";
        size_t pos = text.find(needle);
        if (pos == std::string_view::npos) {
            throw std::runtime_error(std::string("missing key in manifest: ") + key);
        }
        pos = text.find(':', pos + needle.size());
        if (pos == std::string_view::npos) throw std::runtime_error("malformed manifest");
        size_t q1 = text.find('"', pos + 1);
        size_t q2 = (q1 == std::string_view::npos) ? std::string_view::npos : text.find('"', q1 + 1);
        if (q1 == std::string_view::npos || q2 == std::string_view::npos) {
            throw std::runtime_error("malformed manifest value");
        }
        int64_t ms = 0;
        if (!parse_ts_f1(text.substr(q1 + 1, q2 - q1 - 1), ms)) {
            throw std::runtime_error(std::string("bad timestamp in manifest: ") + key);
        }
        return ms;
    };

    win_start_ms = extract_ts("window_start");
    win_end_ms = extract_ts("window_end");
}
