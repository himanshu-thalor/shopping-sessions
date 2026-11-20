#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

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

inline int32_t floor_div_day(int64_t ms) noexcept {
    static constexpr int64_t kDayMs = 86400000LL;
    int64_t d = ms / kDayMs;
    if (ms < 0 && ms % kDayMs != 0) --d;
    return static_cast<int32_t>(d);
}

inline bool parse_2d(const char* p, int64_t& out) noexcept {
    if (!is_ascii_digit(p[0]) || !is_ascii_digit(p[1])) return false;
    out = (p[0] - '0') * 10 + (p[1] - '0');
    return true;
}

inline bool parse_4d(const char* p, int64_t& out) noexcept {
    if (!is_ascii_digit(p[0]) || !is_ascii_digit(p[1]) ||
        !is_ascii_digit(p[2]) || !is_ascii_digit(p[3])) {
        return false;
    }
    out = (p[0] - '0') * 1000 + (p[1] - '0') * 100 + (p[2] - '0') * 10 + (p[3] - '0');
    return true;
}

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

inline bool parse_ts_f1(std::string_view sv, int64_t& out_ms) noexcept {
    if (sv.size() != 20 || sv[19] != 'Z') return false;
    return parse_iso19(sv.data(), out_ms);
}

inline bool parse_ts_any(std::string_view sv, int64_t& out_ms) noexcept {
    const size_t n = sv.size();
    const char* s = sv.data();
    if (n == 20 && s[19] == 'Z') {
        return parse_iso19(s, out_ms);
    }
    if (n == 25 && (s[19] == '+' || s[19] == '-') && s[22] == ':') {
        int64_t local_ms = 0;
        if (!parse_iso19(s, local_ms)) return false;
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
        if (overflow) return false;  // Valid digits only, strictly outside any valid window
        out_ms = static_cast<int64_t>(val);
        return true;
    }
    return false;
}

inline bool parse_price_cents(std::string_view sv, uint32_t& out_price) noexcept {
    const size_t n = sv.size();
    if (n == 0) return false;
    const char* s = sv.data();
    uint64_t val = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!is_ascii_digit(s[i])) return false;
        val = val * 10 + static_cast<uint64_t>(s[i] - '0');
        if (val > 10000000ULL) return false;
    }
    if (val == 0) return false;
    out_price = static_cast<uint32_t>(val);
    return true;
}

inline void parse_manifest(const std::string& path, int64_t& win_start_ms,
                           int64_t& win_end_ms) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open manifest: " + path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    auto extract_ts = [&](const char* key) -> int64_t {
        std::string needle = std::string("\"") + key + "\"";
        size_t pos = text.find(needle);
        if (pos == std::string::npos) {
            throw std::runtime_error(std::string("missing key in manifest: ") + key);
        }
        pos = text.find(':', pos + needle.size());
        if (pos == std::string::npos) throw std::runtime_error("malformed manifest");
        size_t q1 = text.find('"', pos + 1);
        size_t q2 = (q1 == std::string::npos) ? std::string::npos : text.find('"', q1 + 1);
        if (q1 == std::string::npos || q2 == std::string::npos) {
            throw std::runtime_error("malformed manifest value");
        }
        int64_t ms = 0;
        if (!parse_ts_f1(std::string_view(text.data() + q1 + 1, q2 - q1 - 1), ms)) {
            throw std::runtime_error(std::string("bad timestamp in manifest: ") + key);
        }
        return ms;
    };

    win_start_ms = extract_ts("window_start");
    win_end_ms = extract_ts("window_end");
}
