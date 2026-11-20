#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include "io_util.hpp"

inline bool is_ascii_digit(char c) noexcept {
    return static_cast<unsigned char>(c - '0') <= 9;
}

inline bool parse_field(const char* p, size_t len, std::string_view& out) noexcept {
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
inline bool split_csv_line(const char* line, size_t len,
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

inline bool parse_2d(const char* p, int64_t& out) noexcept {
    const uint8_t d0 = static_cast<uint8_t>(p[0] - '0');
    const uint8_t d1 = static_cast<uint8_t>(p[1] - '0');
    if (d0 > 9 || d1 > 9) return false;
    out = d0 * 10 + d1;
    return true;
}

inline bool parse_4d(const char* p, int64_t& out) noexcept {
    const uint8_t d0 = static_cast<uint8_t>(p[0] - '0');
    const uint8_t d1 = static_cast<uint8_t>(p[1] - '0');
    const uint8_t d2 = static_cast<uint8_t>(p[2] - '0');
    const uint8_t d3 = static_cast<uint8_t>(p[3] - '0');
    if (d0 > 9 || d1 > 9 || d2 > 9 || d3 > 9) return false;
    out = d0 * 1000 + d1 * 100 + d2 * 10 + d3;
    return true;
}

struct DateLut {
    struct Entry {
        uint64_t hi8 = 0;  // "YYYY-MM-"
        int64_t midnight_ms = 0;
        uint16_t lo2 = 0;  // "DD"
    };
    std::array<Entry, 32> entries{};

    void build(int64_t start_ms, int64_t end_ms) noexcept {
        entries.fill(Entry{});
        const int64_t d0 = floor_div_day(start_ms);
        const int64_t d1 = floor_div_day(end_ms) + 7;  // cover late ingests too
        const int64_t count = std::min<int64_t>(28, d1 - d0 + 1);
        for (int64_t i = 0; i < count; ++i) {
            const int64_t day_num = d0 + i;
            int64_t y = 0, m = 0, d = 0;
            civil_from_days(day_num, y, m, d);
            if (y < 0 || y > 9999 || d < 1 || d > 31) continue;
            Entry& e = entries[static_cast<size_t>(d)];
            if (e.hi8 != 0) continue;
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
            std::memcpy(&e.hi8, buf, 8);
            std::memcpy(&e.lo2, buf + 8, 2);
            e.midnight_ms = day_num * 86400000LL;
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

inline bool parse_iso19_fast(const char* s, const DateLut& lut, int64_t& out_utc_ms) noexcept {
    if (s[10] != 'T' || s[13] != ':' || s[16] != ':') return false;
    const uint8_t d0 = static_cast<uint8_t>(s[8] - '0');
    const uint8_t d1 = static_cast<uint8_t>(s[9] - '0');
    if (d0 <= 9 && d1 <= 9) {
        const size_t dd = static_cast<size_t>(d0 * 10 + d1);
        if (dd <= 31) {
            const DateLut::Entry& e = lut.entries[dd];
            uint64_t hi8;
            uint16_t lo2;
            std::memcpy(&hi8, s, 8);
            std::memcpy(&lo2, s + 8, 2);
            if (e.hi8 == hi8 && e.lo2 == lo2) {
                int64_t hour, minute, second;
                if (!parse_2d(s + 11, hour) || !parse_2d(s + 14, minute) ||
                    !parse_2d(s + 17, second)) {
                    return false;
                }
                if (hour > 23 || minute > 59 || second > 59) return false;
                out_utc_ms =
                    e.midnight_ms + (hour * 3600LL + minute * 60LL + second) * 1000LL;
                return true;
            }
        }
    }
    return parse_iso19(s, out_utc_ms);
}

inline bool parse_ts_f1(std::string_view sv, int64_t& out_ms) noexcept {
    if (sv.size() != 20 || sv[19] != 'Z') return false;
    return parse_iso19(sv.data(), out_ms);
}

inline bool parse_ts_f1_fast(std::string_view sv, const DateLut& lut, int64_t& out_ms) noexcept {
    if (sv.size() != 20 || sv[19] != 'Z') return false;
    return parse_iso19_fast(sv.data(), lut, out_ms);
}

inline bool parse_13d_swar(const char* s, int64_t& out_ms) noexcept {
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
    if ((bad0 | bad1) != 0 || last_d > 9) return false;

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

inline bool parse_ts_any_fast(std::string_view sv, const DateLut& lut, int64_t& out_ms) noexcept {
    const size_t n = sv.size();
    const char* s = sv.data();
    if (n == 20 && s[19] == 'Z') {
        return parse_iso19_fast(s, lut, out_ms);
    }
    if (n == 13) {
        return parse_13d_swar(s, out_ms);
    }
    if (n == 25 && (s[19] == '+' || s[19] == '-') && s[22] == ':') {
        int64_t local_ms = 0;
        if (!parse_iso19_fast(s, lut, local_ms)) return false;
        int64_t off_h = 0, off_m = 0;
        if (!parse_2d(s + 20, off_h) || !parse_2d(s + 23, off_m)) return false;
        if (off_h > 23 || off_m > 59) return false;
        const int64_t off_ms = (off_h * 60LL + off_m) * 60000LL;
        out_ms = (s[19] == '+') ? (local_ms - off_ms) : (local_ms + off_ms);
        return true;
    }
    if (n >= 1) {
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
    return false;
}

inline bool parse_ts_any(std::string_view sv, int64_t& out_ms) noexcept {
    static constexpr DateLut kEmptyLut{};
    return parse_ts_any_fast(sv, kEmptyLut, out_ms);
}

inline bool parse_price_cents(std::string_view sv, uint32_t& out_price) noexcept {
    const size_t n = sv.size();
    if (n == 0) return false;
    const char* s = sv.data();
    uint64_t val = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t d = static_cast<uint8_t>(s[i] - '0');
        if (d > 9) return false;
        val = val * 10 + d;
        if (val > 10000000ULL) return false;
    }
    if (val == 0) return false;
    out_price = static_cast<uint32_t>(val);
    return true;
}

namespace csv_simd {

template <size_t N, typename Callback>
inline void dispatch_line(const char* buf, size_t line_start, size_t line_end,
                          const size_t (&comma_pos)[N], size_t comma_cnt,
                          const size_t (&quote_pos)[2 * N], size_t quote_cnt,
                          bool bad_line, Callback&& cb) {
    if (bad_line || line_end <= line_start || comma_cnt != N - 1 || (quote_cnt & 1u) != 0) {
        return;
    }
    std::string_view fields[N];
    if (quote_cnt == 0) {
        for (size_t fi = 0; fi < N; ++fi) {
            const size_t fs = (fi == 0) ? line_start : (comma_pos[fi - 1] + 1);
            const size_t fe = (fi + 1 == N) ? line_end : comma_pos[fi];
            fields[fi] = std::string_view(buf + fs, fe - fs);
        }
        cb(fields);
        return;
    }
    size_t qi = 0;
    for (size_t fi = 0; fi < N; ++fi) {
        const size_t fs = (fi == 0) ? line_start : (comma_pos[fi - 1] + 1);
        const size_t fe = (fi + 1 == N) ? line_end : comma_pos[fi];
        if (qi < quote_cnt && quote_pos[qi] == fs) {
            if (fe - fs >= 2 && quote_pos[qi + 1] == fe - 1) {
                fields[fi] = std::string_view(buf + fs + 1, (fe - fs) - 2);
                qi += 2;
            } else {
                return;
            }
        } else {
            if (qi < quote_cnt && quote_pos[qi] < fe) return;
            fields[fi] = std::string_view(buf + fs, fe - fs);
        }
    }
    if (qi == quote_cnt) {
        cb(fields);
    }
}

template <size_t N, typename Callback>
__attribute__((target("avx512f,avx512bw,bmi,bmi2"))) void scan_csv_avx512(
    const char* buf, size_t data_start, size_t limit_in_buf, size_t buf_len, Callback&& cb) {
    if (data_start >= limit_in_buf) return;

    const __m512i v_nl = _mm512_set1_epi8('\n');
    const __m512i v_cm = _mm512_set1_epi8(',');
    const __m512i v_qt = _mm512_set1_epi8('"');

    size_t pos = data_start;
    size_t line_start = data_start;
    size_t comma_pos[N]{};
    size_t quote_pos[2 * N]{};
    size_t comma_cnt = 0;
    size_t quote_cnt = 0;
    bool bad_line = false;

    while (pos < buf_len) {
        const size_t rem = buf_len - pos;
        const uint64_t valid_mask = (rem >= 64) ? ~0ULL : ((1ULL << rem) - 1ULL);
        const __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(buf + pos));
        const uint64_t nl_m = static_cast<uint64_t>(_mm512_cmpeq_epi8_mask(v, v_nl)) & valid_mask;
        const uint64_t cm_m = static_cast<uint64_t>(_mm512_cmpeq_epi8_mask(v, v_cm)) & valid_mask;
        const uint64_t qt_m = static_cast<uint64_t>(_mm512_cmpeq_epi8_mask(v, v_qt)) & valid_mask;

        uint64_t any = nl_m | cm_m | qt_m;
        while (any != 0) {
            const unsigned bit = static_cast<unsigned>(__builtin_ctzll(any));
            const uint64_t bmask = 1ULL << bit;
            any &= any - 1;
            const size_t abs_pos = pos + bit;
            if (cm_m & bmask) {
                if (!bad_line) {
                    if (comma_cnt + 1 < N) {
                        comma_pos[comma_cnt++] = abs_pos;
                    } else {
                        bad_line = true;
                    }
                }
            } else if (qt_m & bmask) {
                if (!bad_line) {
                    if (quote_cnt < 2 * N) {
                        quote_pos[quote_cnt++] = abs_pos;
                    } else {
                        bad_line = true;
                    }
                }
            } else {
                dispatch_line<N>(buf, line_start, abs_pos, comma_pos, comma_cnt, quote_pos,
                                 quote_cnt, bad_line, cb);
                line_start = abs_pos + 1;
                comma_cnt = 0;
                quote_cnt = 0;
                bad_line = false;
                if (line_start >= limit_in_buf) return;
            }
        }
        pos += (rem >= 64) ? 64 : rem;
    }

    if (line_start < limit_in_buf && buf_len > line_start) {
        dispatch_line<N>(buf, line_start, buf_len, comma_pos, comma_cnt, quote_pos, quote_cnt,
                         bad_line, cb);
    }
}

template <size_t N, typename Callback>
__attribute__((target("avx2,bmi,bmi2"))) void scan_csv_avx2(
    const char* buf, size_t data_start, size_t limit_in_buf, size_t buf_len, Callback&& cb) {
    if (data_start >= limit_in_buf) return;

    const __m256i v_nl = _mm256_set1_epi8('\n');
    const __m256i v_cm = _mm256_set1_epi8(',');
    const __m256i v_qt = _mm256_set1_epi8('"');

    size_t pos = data_start;
    size_t line_start = data_start;
    size_t comma_pos[N]{};
    size_t quote_pos[2 * N]{};
    size_t comma_cnt = 0;
    size_t quote_cnt = 0;
    bool bad_line = false;

    while (pos < buf_len) {
        const size_t rem = buf_len - pos;
        const uint64_t valid_mask = (rem >= 64) ? ~0ULL : ((1ULL << rem) - 1ULL);
        const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buf + pos));
        const __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buf + pos + 32));

        const uint64_t nl_m =
            (static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, v_nl))) |
             (static_cast<uint64_t>(
                  static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, v_nl))))
              << 32)) &
            valid_mask;
        const uint64_t cm_m =
            (static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, v_cm))) |
             (static_cast<uint64_t>(
                  static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, v_cm))))
              << 32)) &
            valid_mask;
        const uint64_t qt_m =
            (static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, v_qt))) |
             (static_cast<uint64_t>(
                  static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, v_qt))))
              << 32)) &
            valid_mask;

        uint64_t any = nl_m | cm_m | qt_m;
        while (any != 0) {
            const unsigned bit = static_cast<unsigned>(__builtin_ctzll(any));
            const uint64_t bmask = 1ULL << bit;
            any &= any - 1;
            const size_t abs_pos = pos + bit;
            if (cm_m & bmask) {
                if (!bad_line) {
                    if (comma_cnt + 1 < N) {
                        comma_pos[comma_cnt++] = abs_pos;
                    } else {
                        bad_line = true;
                    }
                }
            } else if (qt_m & bmask) {
                if (!bad_line) {
                    if (quote_cnt < 2 * N) {
                        quote_pos[quote_cnt++] = abs_pos;
                    } else {
                        bad_line = true;
                    }
                }
            } else {
                dispatch_line<N>(buf, line_start, abs_pos, comma_pos, comma_cnt, quote_pos,
                                 quote_cnt, bad_line, cb);
                line_start = abs_pos + 1;
                comma_cnt = 0;
                quote_cnt = 0;
                bad_line = false;
                if (line_start >= limit_in_buf) return;
            }
        }
        pos += (rem >= 64) ? 64 : rem;
    }

    if (line_start < limit_in_buf && buf_len > line_start) {
        dispatch_line<N>(buf, line_start, buf_len, comma_pos, comma_cnt, quote_pos, quote_cnt,
                         bad_line, cb);
    }
}

}  // namespace csv_simd

template <size_t N, typename Callback>
inline void for_each_csv_line(const char* buf, size_t data_start, size_t limit_in_buf,
                              size_t buf_len, Callback&& cb) {
    static const bool has_avx512bw = __builtin_cpu_supports("avx512bw");
    if (has_avx512bw) {
        csv_simd::scan_csv_avx512<N>(buf, data_start, limit_in_buf, buf_len, cb);
    } else {
        csv_simd::scan_csv_avx2<N>(buf, data_start, limit_in_buf, buf_len, cb);
    }
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
