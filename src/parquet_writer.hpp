#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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
    explicit RawFile(const std::string& path) : path_(path), io_buf_(1 << 20) {
        fp_ = std::fopen(path.c_str(), "wb");
        if (!fp_) throw std::runtime_error("cannot open output parquet: " + path);
        std::setvbuf(fp_, io_buf_.data(), _IOFBF, io_buf_.size());
        write_raw("PAR1", 4);
    }

    ~RawFile() {
        if (fp_) std::fclose(fp_);
    }

    RawFile(const RawFile&) = delete;
    RawFile& operator=(const RawFile&) = delete;

    int64_t offset() const noexcept { return offset_; }

    void write_raw(const void* p, size_t n) {
        if (n == 0) return;
        if (std::fwrite(p, 1, n, fp_) != n) {
            throw std::runtime_error("parquet write failed: " + path_);
        }
        offset_ += static_cast<int64_t>(n);
    }

    void close_file() {
        if (fp_) {
            if (std::fflush(fp_) != 0 || std::fclose(fp_) != 0) {
                fp_ = nullptr;
                throw std::runtime_error("parquet close failed: " + path_);
            }
            fp_ = nullptr;
        }
    }

   private:
    std::string path_;
    std::vector<char> io_buf_;
    FILE* fp_ = nullptr;
    int64_t offset_ = 0;
};

// Hybrid RLE / Bit-Packed Hybrid encoder for bit_width = 1 definition levels:
// uses RLE runs for >= 8 identical values, bit-packed chunks (<= 63 groups of 8) otherwise.
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
    hdr.field_i32(last, 1, 0);         // PageType::DATA_PAGE
    hdr.field_i32(last, 2, body_len);  // uncompressed_page_size
    hdr.field_i32(last, 3, body_len);  // compressed_page_size
    hdr.field_hdr(last, 5, 12);        // data_page_header struct
    {
        int16_t dlast = 0;
        hdr.field_i32(dlast, 1, num_values);
        hdr.field_i32(dlast, 2, 0);  // Encoding::PLAIN
        hdr.field_i32(dlast, 3, 3);  // definition_level_encoding = RLE
        hdr.field_i32(dlast, 4, 3);  // repetition_level_encoding = RLE
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
            f.field_i32(sl, 6, 0);  // ConvertedType::UTF8
            f.field_hdr(sl, 10, 12);
            {
                int16_t ll = 0;
                f.field_hdr(ll, 1, 12);  // STRING = 1
                f.struct_end();
                f.struct_end();
            }
        } else if (k == ColKind::DATE_REQ) {
            f.field_i32(sl, 6, 6);  // ConvertedType::DATE
            f.field_hdr(sl, 10, 12);
            {
                int16_t ll = 0;
                f.field_hdr(ll, 6, 12);  // DATE = 6
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
    static constexpr int64_t kRowGroupRows = 250000;

    explicit UserDailyParquetWriter(const std::string& path) : file_(path) {
        col_user_.reserve(1 << 20);
        col_day_.reserve(256 << 10);
        col_sessions_.reserve(256 << 10);
        col_events_.reserve(256 << 10);
        col_revenue_.reserve(512 << 10);
        def_top_cat_.reserve(65536);
        vals_top_cat_.reserve(256 << 10);
    }

    void add_row(std::string_view user_id, int32_t day, int32_t sessions, int32_t events,
                 int64_t revenue_cents, bool has_top_cat, std::string_view top_cat) {
        const uint32_t ulen = static_cast<uint32_t>(user_id.size());
        col_user_.le32(ulen);
        col_user_.bytes(user_id.data(), ulen);

        col_day_.le32(static_cast<uint32_t>(day));
        col_sessions_.le32(static_cast<uint32_t>(sessions));
        col_events_.le32(static_cast<uint32_t>(events));
        col_revenue_.le64(static_cast<uint64_t>(revenue_cents));

        def_top_cat_.push_back(has_top_cat ? 1 : 0);
        if (has_top_cat) {
            const uint32_t clen = static_cast<uint32_t>(top_cat.size());
            vals_top_cat_.le32(clen);
            vals_top_cat_.bytes(top_cat.data(), clen);
        }

        ++rg_rows_;
        ++total_rows_;
        if (rg_rows_ >= kRowGroupRows) {
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

        auto emit_col = [&](int32_t type, std::string_view name, bool optional,
                            const parquet_detail::Buf& body) {
            parquet_detail::write_page_header(hdr_tmp_, static_cast<int32_t>(body.size()),
                                              static_cast<int32_t>(rg_rows_));
            const int64_t off = file_.offset();
            file_.write_raw(hdr_tmp_.data.data(), hdr_tmp_.size());
            file_.write_raw(body.data.data(), body.size());
            const int64_t tot = static_cast<int64_t>(hdr_tmp_.size() + body.size());
            rg.total_byte_size += tot;
            rg.cols.push_back(parquet_detail::ChunkMeta{type, name, optional, rg_rows_, tot, off});
        };

        emit_col(6, "user_id", false, col_user_);
        emit_col(1, "day", false, col_day_);
        emit_col(1, "sessions", false, col_sessions_);
        emit_col(1, "events", false, col_events_);
        emit_col(2, "revenue_cents", false, col_revenue_);

        body_top_cat_.clear();
        parquet_detail::encode_def_levels_bit1(body_top_cat_, def_top_cat_.data(),
                                               def_top_cat_.size());
        body_top_cat_.bytes(vals_top_cat_.data.data(), vals_top_cat_.size());
        emit_col(6, "top_category", true, body_top_cat_);

        rgs_.push_back(std::move(rg));
        col_user_.clear();
        col_day_.clear();
        col_sessions_.clear();
        col_events_.clear();
        col_revenue_.clear();
        def_top_cat_.clear();
        vals_top_cat_.clear();
        rg_rows_ = 0;
    }

    parquet_detail::RawFile file_;
    parquet_detail::Buf col_user_, col_day_, col_sessions_, col_events_, col_revenue_;
    std::vector<uint8_t> def_top_cat_;
    parquet_detail::Buf vals_top_cat_, body_top_cat_, hdr_tmp_;
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
