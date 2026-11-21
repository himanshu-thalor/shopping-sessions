#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

#include "hash_util.hpp"

namespace parquet_detail {

class Buf {
   public:
    std::vector<uint8_t> data;

    void clear() noexcept { data.clear(); }
    void reserve(size_t n) { data.reserve(n); }
    size_t size() const noexcept { return data.size(); }

    void u8(uint8_t v) { data.push_back(v); }

    void bytes(const void* p, size_t n) {
        if (n == 0) return;
        const size_t old = data.size();
        data.resize(old + n);
        std::memcpy(data.data() + old, p, n);
    }

    void le32(uint32_t v) {
        const size_t old = data.size();
        data.resize(old + 4);
        std::memcpy(data.data() + old, &v, 4);
    }

    void le64(uint64_t v) {
        const size_t old = data.size();
        data.resize(old + 8);
        std::memcpy(data.data() + old, &v, 8);
    }

    void varint(uint64_t v) {
        while (v >= 0x80) {
            data.push_back(static_cast<uint8_t>((v & 0x7F) | 0x80));
            v >>= 7;
        }
        data.push_back(static_cast<uint8_t>(v));
    }

    void zigzag32(int32_t v) {
        const uint32_t uv = static_cast<uint32_t>(v);
        varint(static_cast<uint32_t>((uv << 1) ^ static_cast<uint32_t>(v >> 31)));
    }

    void zigzag64(int64_t v) {
        const uint64_t uv = static_cast<uint64_t>(v);
        varint((uv << 1) ^ static_cast<uint64_t>(v >> 63));
    }

    void field_hdr(int16_t& last_id, int16_t field_id, uint8_t type) {
        const int16_t delta = field_id - last_id;
        if (delta > 0 && delta <= 15) {
            u8(static_cast<uint8_t>((delta << 4) | type));
        } else {
            u8(type);
            zigzag32(field_id);
        }
        last_id = field_id;
    }

    void field_i32(int16_t& last_id, int16_t field_id, int32_t v) {
        field_hdr(last_id, field_id, 5);
        zigzag32(v);
    }

    void field_i64(int16_t& last_id, int16_t field_id, int64_t v) {
        field_hdr(last_id, field_id, 6);
        zigzag64(v);
    }

    void field_str(int16_t& last_id, int16_t field_id, std::string_view sv) {
        field_hdr(last_id, field_id, 8);
        varint(sv.size());
        bytes(sv.data(), sv.size());
    }

    void field_list_hdr(int16_t& last_id, int16_t field_id, uint8_t elem_type, size_t count) {
        field_hdr(last_id, field_id, 9);
        if (count < 15) {
            u8(static_cast<uint8_t>((count << 4) | elem_type));
        } else {
            u8(static_cast<uint8_t>(0xF0 | elem_type));
            varint(count);
        }
    }

    void struct_end() { u8(0); }
};

class FastByteCursor {
   public:
    explicit FastByteCursor(size_t initial_cap = 1 << 20)
        : buf_(std::make_unique_for_overwrite<uint8_t[]>(initial_cap)),
          cap_(initial_cap),
          cur_(buf_.get()),
          end_(buf_.get() + initial_cap) {}

    void clear() noexcept { cur_ = buf_.get(); }
    const uint8_t* data() const noexcept { return buf_.get(); }
    size_t size() const noexcept { return static_cast<size_t>(cur_ - buf_.get()); }

    [[gnu::always_inline]] inline void write_len_str(std::string_view sv) {
        const uint32_t slen = static_cast<uint32_t>(sv.size());
        const size_t need = 4 + slen;
        if (__builtin_expect(static_cast<size_t>(end_ - cur_) < need, 0)) {
            grow(need);
        }
        std::memcpy(cur_, &slen, 4);
        if (slen > 0) std::memcpy(cur_ + 4, sv.data(), slen);
        cur_ += need;
    }

    [[gnu::always_inline]] inline void write_fast_id64(FastId64<'u'> id,
                                                       const OverflowArena* const* arenas) {
        if (__builtin_expect(id.is_inline(), 1)) {
            if (__builtin_expect(static_cast<size_t>(end_ - cur_) < 16, 0)) {
                grow(16);
            }
            const uint32_t slen = id.inline_len();
            std::memcpy(cur_, &slen, 4);
            cur_[4] = 'u';
            std::memcpy(cur_ + 5, &id.raw, 8);
            cur_ += 4 + slen;
        } else {
            char scratch[9];
            write_len_str(id.view(arenas, scratch));
        }
    }

   private:
    __attribute__((noinline)) void grow(size_t need) {
        const size_t used = static_cast<size_t>(cur_ - buf_.get());
        size_t ncap = cap_ * 2;
        while (ncap < used + need) ncap *= 2;
        auto nb = std::make_unique_for_overwrite<uint8_t[]>(ncap);
        if (used > 0) std::memcpy(nb.get(), buf_.get(), used);
        buf_ = std::move(nb);
        cap_ = ncap;
        cur_ = buf_.get() + used;
        end_ = buf_.get() + ncap;
    }

    std::unique_ptr<uint8_t[]> buf_;
    size_t cap_;
    uint8_t* cur_;
    uint8_t* end_;
};

enum class ColKind : uint8_t {
    UTF8_REQ,
    UTF8_OPT,
    DATE_REQ,
    INT32_REQ,
    INT64_REQ,
};

struct ColDef {
    const char* name;
    ColKind kind;
};

struct ChunkMeta {
    int32_t type = 0;
    std::string_view name;
    bool optional = false;
    int64_t num_values = 0;
    int64_t total_size = 0;
    int64_t data_page_offset = 0;
};

class RawFile {
   public:
    static constexpr size_t kSmallBufBytes = 65536;

    explicit RawFile(const std::string& path)
        : path_(path), io_buf_(std::make_unique_for_overwrite<char[]>(kSmallBufBytes)) {
        fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd_ < 0) throw std::runtime_error("cannot open output parquet: " + path);
        write_raw("PAR1", 4);
    }

    ~RawFile() {
        if (fd_ >= 0) ::close(fd_);
    }

    RawFile(const RawFile&) = delete;
    RawFile& operator=(const RawFile&) = delete;

    int64_t offset() const noexcept { return offset_; }

    void write_raw(const void* p, size_t n) {
        if (n == 0) return;
        offset_ += static_cast<int64_t>(n);
        if (n >= 32768) {
            flush_buf();
            write_fd(p, n);
            return;
        }
        if (buf_pos_ + n > kSmallBufBytes) {
            flush_buf();
        }
        std::memcpy(io_buf_.get() + buf_pos_, p, n);
        buf_pos_ += n;
    }

    void close_file() {
        if (fd_ >= 0) {
            flush_buf();
            const int rc = ::close(fd_);
            fd_ = -1;
            if (rc != 0) throw std::runtime_error("parquet close failed: " + path_);
        }
    }

   private:
    void write_fd(const void* p, size_t n) {
        const char* cp = static_cast<const char*>(p);
        size_t done = 0;
        while (done < n) {
            const ssize_t w = ::write(fd_, cp + done, n - done);
            if (w < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error("parquet write failed: " + path_);
            }
            done += static_cast<size_t>(w);
        }
    }

    void flush_buf() {
        if (buf_pos_ > 0) {
            write_fd(io_buf_.get(), buf_pos_);
            buf_pos_ = 0;
        }
    }

    std::string path_;
    std::unique_ptr<char[]> io_buf_;
    size_t buf_pos_ = 0;
    int fd_ = -1;
    int64_t offset_ = 0;
};

inline void encode_def_levels_bit1(Buf& out, const uint8_t* defs, size_t n) {
    const size_t len_pos = out.size();
    out.le32(0);
    const size_t start_pos = out.size();

    size_t i = 0;
    while (i < n) {
        const uint8_t v = defs[i] & 1u;
        size_t run = 1;
        while (i + run < n && (defs[i + run] & 1u) == v) ++run;
        if (run >= 8) {
            out.varint(run << 1);
            out.u8(v);
            i += run;
        } else {
            size_t bp_end = i + 8;
            while (bp_end < n && (bp_end - i) < 63 * 8) {
                const uint8_t nv = defs[bp_end] & 1u;
                size_t nrun = 1;
                while (bp_end + nrun < n && nrun < 8 && (defs[bp_end + nrun] & 1u) == nv) ++nrun;
                if (nrun >= 8) break;
                bp_end += 8;
            }
            if (bp_end > n) bp_end = n;
            const size_t groups = (bp_end - i + 7) / 8;
            out.u8(static_cast<uint8_t>((groups << 1) | 1));
            for (size_t g = 0; g < groups; ++g) {
                uint8_t byte = 0;
                const size_t base = i + g * 8;
                const size_t lim = std::min<size_t>(8, n - base);
                for (size_t b = 0; b < lim; ++b) {
                    byte |= static_cast<uint8_t>((defs[base + b] & 1u) << b);
                }
                out.u8(byte);
            }
            i += groups * 8;
        }
    }

    const uint32_t rle_bytes = static_cast<uint32_t>(out.size() - start_pos);
    std::memcpy(out.data.data() + len_pos, &rle_bytes, 4);
}

inline void write_page_header(Buf& hdr, int32_t body_len, int32_t num_values) {
    hdr.clear();
    int16_t last = 0;
    hdr.field_i32(last, 1, 0);
    hdr.field_i32(last, 2, body_len);
    hdr.field_i32(last, 3, body_len);
    hdr.field_hdr(last, 5, 12);
    {
        int16_t dlast = 0;
        hdr.field_i32(dlast, 1, num_values);
        hdr.field_i32(dlast, 2, 0);
        hdr.field_i32(dlast, 3, 3);
        hdr.field_i32(dlast, 4, 3);
        hdr.struct_end();
    }
    hdr.struct_end();
}

struct RowGroupRecord {
    int64_t num_rows = 0;
    int64_t total_byte_size = 0;
    std::vector<ChunkMeta> cols;
};

inline void write_footer(RawFile& file, const ColDef* schema, size_t ncols,
                         int64_t total_rows, const std::vector<RowGroupRecord>& rgs) {
    Buf f;
    int16_t last = 0;
    f.field_i32(last, 1, 1);

    f.field_list_hdr(last, 2, 12, 1 + ncols);
    {
        int16_t sl = 0;
        f.field_str(sl, 4, "schema");
        f.field_i32(sl, 5, static_cast<int32_t>(ncols));
        f.struct_end();
    }
    for (size_t c = 0; c < ncols; ++c) {
        int16_t sl = 0;
        const ColKind k = schema[c].kind;
        const int32_t ptype =
            (k == ColKind::DATE_REQ || k == ColKind::INT32_REQ) ? 1 : (k == ColKind::INT64_REQ ? 2 : 6);
        const int32_t rep = (k == ColKind::UTF8_OPT) ? 1 : 0;
        f.field_i32(sl, 1, ptype);
        f.field_i32(sl, 3, rep);
        f.field_str(sl, 4, schema[c].name);
        if (k == ColKind::UTF8_REQ || k == ColKind::UTF8_OPT) {
            f.field_i32(sl, 6, 0);
            f.field_hdr(sl, 10, 12);
            {
                int16_t ll = 0;
                f.field_hdr(ll, 1, 12);
                f.struct_end();
                f.struct_end();
            }
        } else if (k == ColKind::DATE_REQ) {
            f.field_i32(sl, 6, 6);
            f.field_hdr(sl, 10, 12);
            {
                int16_t ll = 0;
                f.field_hdr(ll, 6, 12);
                f.struct_end();
                f.struct_end();
            }
        }
        f.struct_end();
    }

    f.field_i64(last, 3, total_rows);

    f.field_list_hdr(last, 4, 12, rgs.size());
    for (const auto& rg : rgs) {
        int16_t rl = 0;
        f.field_list_hdr(rl, 1, 12, rg.cols.size());
        for (const auto& cm : rg.cols) {
            int16_t cl = 0;
            f.field_i64(cl, 2, cm.data_page_offset);
            f.field_hdr(cl, 3, 12);
            {
                int16_t ml = 0;
                f.field_i32(ml, 1, cm.type);
                f.field_hdr(ml, 2, 9);
                if (cm.optional) {
                    f.u8((2 << 4) | 5);
                    f.zigzag32(0);
                    f.zigzag32(3);
                } else {
                    f.u8((1 << 4) | 5);
                    f.zigzag32(0);
                }
                f.field_list_hdr(ml, 3, 8, 1);
                f.varint(cm.name.size());
                f.bytes(cm.name.data(), cm.name.size());
                f.field_i32(ml, 4, 0);
                f.field_i64(ml, 5, cm.num_values);
                f.field_i64(ml, 6, cm.total_size);
                f.field_i64(ml, 7, cm.total_size);
                f.field_i64(ml, 9, cm.data_page_offset);
                f.struct_end();
            }
            f.struct_end();
        }
        f.field_i64(rl, 2, rg.total_byte_size);
        f.field_i64(rl, 3, rg.num_rows);
        f.struct_end();
    }
    f.struct_end();

    const uint32_t footer_len = static_cast<uint32_t>(f.size());
    file.write_raw(f.data.data(), f.data.size());
    uint8_t tail[8];
    std::memcpy(tail, &footer_len, 4);
    std::memcpy(tail + 4, "PAR1", 4);
    file.write_raw(tail, 8);
    file.close_file();
}

}  // namespace parquet_detail

class UserDailyParquetWriter {
   public:
    static constexpr int64_t kRowGroupRows = 131072;

    explicit UserDailyParquetWriter(const std::string& path)
        : file_(path),
          col_user_(2 << 20),
          vals_top_cat_(512 << 10),
          col_day_(std::make_unique_for_overwrite<int32_t[]>(kRowGroupRows)),
          col_sessions_(std::make_unique_for_overwrite<int32_t[]>(kRowGroupRows)),
          col_events_(std::make_unique_for_overwrite<int32_t[]>(kRowGroupRows)),
          col_revenue_(std::make_unique_for_overwrite<int64_t[]>(kRowGroupRows)),
          def_top_cat_(std::make_unique_for_overwrite<uint8_t[]>(kRowGroupRows)) {}

    [[gnu::always_inline]] inline void add_row(FastId64<'u'> user_id,
                                               const OverflowArena* const* arenas, int32_t day,
                                               int32_t sessions, int32_t events,
                                               int64_t revenue_cents, bool has_top_cat,
                                               std::string_view top_cat) {
        col_user_.write_fast_id64(user_id, arenas);

        const size_t i = static_cast<size_t>(rg_rows_);
        col_day_[i] = day;
        col_sessions_[i] = sessions;
        col_events_[i] = events;
        col_revenue_[i] = revenue_cents;
        def_top_cat_[i] = has_top_cat ? 1 : 0;

        if (has_top_cat) {
            vals_top_cat_.write_len_str(top_cat);
        }

        ++rg_rows_;
        ++total_rows_;
        if (__builtin_expect(rg_rows_ >= kRowGroupRows, 0)) {
            flush_row_group();
        }
    }

    void finish() {
        if (closed_) return;
        flush_row_group();
        static constexpr parquet_detail::ColDef kSchema[6] = {
            {"user_id", parquet_detail::ColKind::UTF8_REQ},
            {"day", parquet_detail::ColKind::DATE_REQ},
            {"sessions", parquet_detail::ColKind::INT32_REQ},
            {"events", parquet_detail::ColKind::INT32_REQ},
            {"revenue_cents", parquet_detail::ColKind::INT64_REQ},
            {"top_category", parquet_detail::ColKind::UTF8_OPT},
        };
        parquet_detail::write_footer(file_, kSchema, 6, total_rows_, rgs_);
        closed_ = true;
    }

   private:
    void flush_row_group() {
        if (rg_rows_ == 0) return;
        parquet_detail::RowGroupRecord rg;
        rg.num_rows = rg_rows_;
        const size_t nrows = static_cast<size_t>(rg_rows_);

        auto emit_raw = [&](int32_t type, std::string_view name, bool optional,
                            const void* body_ptr, size_t body_len) {
            parquet_detail::write_page_header(hdr_tmp_, static_cast<int32_t>(body_len),
                                              static_cast<int32_t>(rg_rows_));
            const int64_t off = file_.offset();
            file_.write_raw(hdr_tmp_.data.data(), hdr_tmp_.size());
            file_.write_raw(body_ptr, body_len);
            const int64_t tot = static_cast<int64_t>(hdr_tmp_.size() + body_len);
            rg.total_byte_size += tot;
            rg.cols.push_back(parquet_detail::ChunkMeta{type, name, optional, rg_rows_, tot, off});
        };

        emit_raw(6, "user_id", false, col_user_.data(), col_user_.size());
        emit_raw(1, "day", false, col_day_.get(), nrows * 4);
        emit_raw(1, "sessions", false, col_sessions_.get(), nrows * 4);
        emit_raw(1, "events", false, col_events_.get(), nrows * 4);
        emit_raw(2, "revenue_cents", false, col_revenue_.get(), nrows * 8);

        def_buf_.clear();
        parquet_detail::encode_def_levels_bit1(def_buf_, def_top_cat_.get(), nrows);
        const size_t top_body_len = def_buf_.size() + vals_top_cat_.size();
        parquet_detail::write_page_header(hdr_tmp_, static_cast<int32_t>(top_body_len),
                                          static_cast<int32_t>(rg_rows_));
        const int64_t top_off = file_.offset();
        file_.write_raw(hdr_tmp_.data.data(), hdr_tmp_.size());
        file_.write_raw(def_buf_.data.data(), def_buf_.size());
        file_.write_raw(vals_top_cat_.data(), vals_top_cat_.size());
        const int64_t top_tot = static_cast<int64_t>(hdr_tmp_.size() + top_body_len);
        rg.total_byte_size += top_tot;
        rg.cols.push_back(
            parquet_detail::ChunkMeta{6, "top_category", true, rg_rows_, top_tot, top_off});

        rgs_.push_back(std::move(rg));
        col_user_.clear();
        vals_top_cat_.clear();
        rg_rows_ = 0;
    }

    parquet_detail::RawFile file_;
    parquet_detail::FastByteCursor col_user_, vals_top_cat_;
    std::unique_ptr<int32_t[]> col_day_, col_sessions_, col_events_;
    std::unique_ptr<int64_t[]> col_revenue_;
    std::unique_ptr<uint8_t[]> def_top_cat_;
    parquet_detail::Buf def_buf_, hdr_tmp_;
    std::vector<parquet_detail::RowGroupRecord> rgs_;
    int64_t rg_rows_ = 0;
    int64_t total_rows_ = 0;
    bool closed_ = false;
};

class CampaignDailyParquetWriter {
   public:
    static constexpr int64_t kRowGroupRows = 250000;

    explicit CampaignDailyParquetWriter(const std::string& path) : file_(path) {}

    void add_row(std::string_view campaign_id, std::string_view channel, int32_t day,
                 int32_t sessions, int32_t purchasing_sessions, int64_t revenue_cents,
                 int32_t assisted_sessions) {
        const uint32_t clen = static_cast<uint32_t>(campaign_id.size());
        col_camp_.le32(clen);
        col_camp_.bytes(campaign_id.data(), clen);

        const uint32_t chlen = static_cast<uint32_t>(channel.size());
        col_chan_.le32(chlen);
        col_chan_.bytes(channel.data(), chlen);

        col_day_.le32(static_cast<uint32_t>(day));
        col_sess_.le32(static_cast<uint32_t>(sessions));
        col_purch_.le32(static_cast<uint32_t>(purchasing_sessions));
        col_rev_.le64(static_cast<uint64_t>(revenue_cents));
        col_assist_.le32(static_cast<uint32_t>(assisted_sessions));

        ++rg_rows_;
        ++total_rows_;
        if (rg_rows_ >= kRowGroupRows) {
            flush_row_group();
        }
    }

    void finish() {
        if (closed_) return;
        flush_row_group();
        static constexpr parquet_detail::ColDef kSchema[7] = {
            {"campaign_id", parquet_detail::ColKind::UTF8_REQ},
            {"channel", parquet_detail::ColKind::UTF8_REQ},
            {"day", parquet_detail::ColKind::DATE_REQ},
            {"sessions", parquet_detail::ColKind::INT32_REQ},
            {"purchasing_sessions", parquet_detail::ColKind::INT32_REQ},
            {"revenue_cents", parquet_detail::ColKind::INT64_REQ},
            {"assisted_sessions", parquet_detail::ColKind::INT32_REQ},
        };
        parquet_detail::write_footer(file_, kSchema, 7, total_rows_, rgs_);
        closed_ = true;
    }

   private:
    void flush_row_group() {
        if (rg_rows_ == 0) return;
        parquet_detail::RowGroupRecord rg;
        rg.num_rows = rg_rows_;

        auto emit_col = [&](int32_t type, std::string_view name, const parquet_detail::Buf& body) {
            parquet_detail::write_page_header(hdr_tmp_, static_cast<int32_t>(body.size()),
                                              static_cast<int32_t>(rg_rows_));
            const int64_t off = file_.offset();
            file_.write_raw(hdr_tmp_.data.data(), hdr_tmp_.size());
            file_.write_raw(body.data.data(), body.size());
            const int64_t tot = static_cast<int64_t>(hdr_tmp_.size() + body.size());
            rg.total_byte_size += tot;
            rg.cols.push_back(parquet_detail::ChunkMeta{type, name, false, rg_rows_, tot, off});
        };

        emit_col(6, "campaign_id", col_camp_);
        emit_col(6, "channel", col_chan_);
        emit_col(1, "day", col_day_);
        emit_col(1, "sessions", col_sess_);
        emit_col(1, "purchasing_sessions", col_purch_);
        emit_col(2, "revenue_cents", col_rev_);
        emit_col(1, "assisted_sessions", col_assist_);

        rgs_.push_back(std::move(rg));
        col_camp_.clear();
        col_chan_.clear();
        col_day_.clear();
        col_sess_.clear();
        col_purch_.clear();
        col_rev_.clear();
        col_assist_.clear();
        rg_rows_ = 0;
    }

    parquet_detail::RawFile file_;
    parquet_detail::Buf col_camp_, col_chan_, col_day_, col_sess_, col_purch_, col_rev_, col_assist_;
    parquet_detail::Buf hdr_tmp_;
    std::vector<parquet_detail::RowGroupRecord> rgs_;
    int64_t rg_rows_ = 0;
    int64_t total_rows_ = 0;
    bool closed_ = false;
};
