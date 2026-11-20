#pragma once

#include <cstdint>
#include <cstring>
#include <new>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

inline uint64_t wyhash_mix(uint64_t a, uint64_t b) noexcept {
    __uint128_t r = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
    return static_cast<uint64_t>(r) ^ static_cast<uint64_t>(r >> 64);
}

inline uint64_t wyhash_read8(const char* p) noexcept {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

inline uint64_t wyhash_read4(const char* p) noexcept {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

inline uint64_t wyhash64(const char* data, size_t len) noexcept {
    static constexpr uint64_t s0 = 0xa0761d6478bd642full;
    static constexpr uint64_t s1 = 0xe7037ed1a0b428dbull;
    static constexpr uint64_t s2 = 0x8ebc6af09c88c6e3ull;
    static constexpr uint64_t s3 = 0x589965cc75374cc3ull;

    uint64_t seed = wyhash_mix(s0 ^ len, s1);
    uint64_t a = 0, b = 0;
    const char* p = data;
    size_t rem = len;

    if (rem <= 16) {
        if (rem >= 4) {
            a = (wyhash_read4(p) << 32) | wyhash_read4(p + ((rem >> 3) << 2));
            b = (wyhash_read4(p + rem - 4) << 32) |
                wyhash_read4(p + rem - 4 - ((rem >> 3) << 2));
        } else if (rem > 0) {
            a = (static_cast<uint64_t>(static_cast<uint8_t>(p[0])) << 16) |
                (static_cast<uint64_t>(static_cast<uint8_t>(p[rem >> 1])) << 8) |
                static_cast<uint64_t>(static_cast<uint8_t>(p[rem - 1]));
        }
    } else {
        size_t i = rem;
        if (i > 48) {
            uint64_t see1 = seed, see2 = seed;
            do {
                seed = wyhash_mix(wyhash_read8(p) ^ s1, wyhash_read8(p + 8) ^ seed);
                see1 = wyhash_mix(wyhash_read8(p + 16) ^ s2, wyhash_read8(p + 24) ^ see1);
                see2 = wyhash_mix(wyhash_read8(p + 32) ^ s3, wyhash_read8(p + 40) ^ see2);
                p += 48;
                i -= 48;
            } while (i > 48);
            seed ^= see1 ^ see2;
        }
        while (i > 16) {
            seed = wyhash_mix(wyhash_read8(p) ^ s1, wyhash_read8(p + 8) ^ seed);
            i -= 16;
            p += 16;
        }
        a = wyhash_read8(data + len - 16);
        b = wyhash_read8(data + len - 8);
    }
    a ^= s1;
    b ^= seed;
    __uint128_t r = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
    return wyhash_mix(static_cast<uint64_t>(r) ^ s0 ^ len,
                      static_cast<uint64_t>(r >> 64) ^ s1);
}

inline uint64_t wyhash64(std::string_view sv) noexcept {
    return wyhash64(sv.data(), sv.size());
}

inline bool byte_less(std::string_view a, std::string_view b) noexcept {
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    if (n > 0) {
        const int cmp = std::memcmp(a.data(), b.data(), n);
        if (cmp != 0) return cmp < 0;
    }
    return a.size() < b.size();
}

struct OverflowArena {
    static constexpr size_t BLOCK_SHIFT = 20;  // 1 MB blocks
    static constexpr size_t BLOCK_SIZE = size_t(1) << BLOCK_SHIFT;
    static constexpr size_t BLOCK_MASK = BLOCK_SIZE - 1;

    std::vector<char*> blocks;
    size_t pos = BLOCK_SIZE;

    OverflowArena() = default;
    OverflowArena(const OverflowArena&) = delete;
    OverflowArena& operator=(const OverflowArena&) = delete;
    OverflowArena(OverflowArena&& other) noexcept
        : blocks(std::move(other.blocks)), pos(other.pos) {
        other.pos = BLOCK_SIZE;
    }
    OverflowArena& operator=(OverflowArena&& other) noexcept {
        if (this != &other) {
            free_all();
            blocks = std::move(other.blocks);
            pos = other.pos;
            other.pos = BLOCK_SIZE;
        }
        return *this;
    }
    ~OverflowArena() { free_all(); }

    void free_all() noexcept {
        for (char* b : blocks) {
            delete[] b;
        }
        std::vector<char*>().swap(blocks);
        pos = BLOCK_SIZE;
    }

    uint32_t push(std::string_view sv) {
        const size_t len = sv.size();
        if (len > BLOCK_SIZE) {
            char* big = new char[len];
            std::memcpy(big, sv.data(), len);
            const uint32_t bidx = static_cast<uint32_t>(blocks.size());
            blocks.push_back(big);
            const size_t extra = (len - 1) >> BLOCK_SHIFT;
            for (size_t i = 0; i < extra; ++i) {
                blocks.push_back(nullptr);
            }
            pos = BLOCK_SIZE;
            return bidx << BLOCK_SHIFT;
        }
        if (pos + len > BLOCK_SIZE) {
            blocks.push_back(new char[BLOCK_SIZE]);
            pos = 0;
        }
        const uint32_t bidx = static_cast<uint32_t>(blocks.size() - 1);
        const uint32_t off = (bidx << BLOCK_SHIFT) | static_cast<uint32_t>(pos);
        if (len > 0) {
            std::memcpy(blocks.back() + pos, sv.data(), len);
        }
        pos += len;
        return off;
    }

    std::string_view get(uint32_t off, uint32_t len) const noexcept {
        return std::string_view(blocks[off >> BLOCK_SHIFT] + (off & BLOCK_MASK), len);
    }
};

struct alignas(4) CompactStr {
    uint8_t d[12];

    static CompactStr make_inline(std::string_view sv) noexcept {
        CompactStr s{};
        if (!sv.empty()) {
            std::memcpy(s.d, sv.data(), sv.size());
        }
        s.d[11] = static_cast<uint8_t>(sv.size());
        return s;
    }

    static CompactStr make_overflow(uint32_t off, uint32_t len, uint16_t arena_id) noexcept {
        CompactStr s{};
        std::memcpy(s.d, &off, 4);
        std::memcpy(s.d + 4, &len, 4);
        std::memcpy(s.d + 8, &arena_id, 2);
        s.d[11] = 0xFF;
        return s;
    }

    static CompactStr make_str(std::string_view sv, uint16_t arena_id, OverflowArena& arena) {
        if (sv.size() <= 11) {
            return make_inline(sv);
        }
        const uint32_t off = arena.push(sv);
        return make_overflow(off, static_cast<uint32_t>(sv.size()), arena_id);
    }

    static CompactStr make_linked(uint32_t node_id) noexcept {
        CompactStr s{};
        std::memcpy(s.d, &node_id, 4);
        s.d[11] = 0xFE;
        return s;
    }

    uint8_t tag() const noexcept { return d[11]; }
    bool empty_sentinel() const noexcept { return d[11] == 0; }
    bool is_linked() const noexcept { return d[11] == 0xFE; }
    bool is_overflow() const noexcept { return d[11] == 0xFF; }
    bool is_inline() const noexcept { return d[11] <= 11; }

    uint32_t linked_id() const noexcept {
        uint32_t v;
        std::memcpy(&v, d, 4);
        return v;
    }

    void overflow_info(uint32_t& off, uint32_t& len, uint16_t& arena_id) const noexcept {
        std::memcpy(&off, d, 4);
        std::memcpy(&len, d + 4, 4);
        std::memcpy(&arena_id, d + 8, 2);
    }

    std::string_view raw_view(const OverflowArena* arenas) const noexcept {
        if (d[11] <= 11) {
            return std::string_view(reinterpret_cast<const char*>(d), d[11]);
        }
        uint32_t off, len;
        uint16_t aid;
        overflow_info(off, len, aid);
        return arenas[aid].get(off, len);
    }

    static bool raw_equals(const CompactStr& a, const CompactStr& b,
                           const OverflowArena* arenas) noexcept {
        uint32_t a0, a1, a2, b0, b1, b2;
        std::memcpy(&a0, a.d, 4);
        std::memcpy(&a1, a.d + 4, 4);
        std::memcpy(&a2, a.d + 8, 4);
        std::memcpy(&b0, b.d, 4);
        std::memcpy(&b1, b.d + 4, 4);
        std::memcpy(&b2, b.d + 8, 4);
        if (a0 == b0 && a1 == b1 && a2 == b2) return true;
        if (a.d[11] != b.d[11] || a.d[11] != 0xFF) return false;
        if (a1 != b1) return false;
        uint32_t a_off, b_off;
        uint16_t a_aid, b_aid;
        a.overflow_info(a_off, a1, a_aid);
        b.overflow_info(b_off, b1, b_aid);
        const char* pa = arenas[a_aid].blocks[a_off >> OverflowArena::BLOCK_SHIFT] +
                         (a_off & OverflowArena::BLOCK_MASK);
        const char* pb = arenas[b_aid].blocks[b_off >> OverflowArena::BLOCK_SHIFT] +
                         (b_off & OverflowArena::BLOCK_MASK);
        return std::memcmp(pa, pb, a1) == 0;
    }

    static bool equals_cross(const CompactStr& a, const OverflowArena* arenas_a,
                             const CompactStr& b, const OverflowArena* arenas_b) noexcept {
        if (a.d[11] != b.d[11]) return false;
        if (a.d[11] <= 11 || a.d[11] == 0xFE) {
            uint32_t a0, a1, a2, b0, b1, b2;
            std::memcpy(&a0, a.d, 4);
            std::memcpy(&a1, a.d + 4, 4);
            std::memcpy(&a2, a.d + 8, 4);
            std::memcpy(&b0, b.d, 4);
            std::memcpy(&b1, b.d + 4, 4);
            std::memcpy(&b2, b.d + 8, 4);
            return a0 == b0 && a1 == b1 && a2 == b2;
        }
        std::string_view sa = a.raw_view(arenas_a);
        std::string_view sb = b.raw_view(arenas_b);
        return sa.size() == sb.size() &&
               (sa.empty() || std::memcmp(sa.data(), sb.data(), sa.size()) == 0);
    }

    static bool equals_sv(const CompactStr& a, const OverflowArena* arenas_a,
                          std::string_view sv) noexcept {
        if (sv.size() <= 11) {
            if (a.d[11] != sv.size()) return false;
            return sv.empty() || std::memcmp(a.d, sv.data(), sv.size()) == 0;
        }
        if (a.d[11] != 0xFF) return false;
        std::string_view sa = a.raw_view(arenas_a);
        return sa.size() == sv.size() && std::memcmp(sa.data(), sv.data(), sv.size()) == 0;
    }
};
static_assert(sizeof(CompactStr) == 12);
static_assert(std::is_trivially_copyable_v<CompactStr>);

template <typename V>
class FlatStrMap {
   public:
    struct Slot {
        uint64_t hash = 0;  // 0 == empty
        uint32_t off = 0;
        uint32_t len = 0;
        V val{};
    };

    FlatStrMap() = default;

    void reserve(size_t expected) {
        size_t cap = 16;
        while (cap * 11 < expected * 20) cap <<= 1;
        if (cap > table_.size()) {
            rehash(cap);
        }
    }

    size_t size() const noexcept { return count_; }
    bool empty() const noexcept { return count_ == 0; }

    const V* find(std::string_view sv) const noexcept {
        if (table_.empty()) return nullptr;
        uint64_t h = wyhash64(sv);
        if (h == 0) h = 1;
        return find_with_hash(sv, h);
    }

    const V* find_with_hash(std::string_view sv, uint64_t h) const noexcept {
        if (table_.empty()) return nullptr;
        if (h == 0) h = 1;
        const size_t mask = table_.size() - 1;
        size_t idx = static_cast<size_t>(h) & mask;
        for (;;) {
            const Slot& s = table_[idx];
            if (s.hash == 0) return nullptr;
            if (s.hash == h && s.len == sv.size()) {
                std::string_view stored = arena_.get(s.off, s.len);
                if (sv.size() == 0 || std::memcmp(stored.data(), sv.data(), sv.size()) == 0) {
                    return &s.val;
                }
            }
            idx = (idx + 1) & mask;
        }
    }

    std::pair<V*, bool> emplace(std::string_view sv, const V& val) {
        if ((count_ + 1) * 20 >= table_.size() * 11) {
            rehash(table_.empty() ? 1024 : table_.size() * 2);
        }
        uint64_t h = wyhash64(sv);
        if (h == 0) h = 1;
        const size_t mask = table_.size() - 1;
        size_t idx = static_cast<size_t>(h) & mask;
        for (;;) {
            Slot& s = table_[idx];
            if (s.hash == 0) {
                s.hash = h;
                s.off = arena_.push(sv);
                s.len = static_cast<uint32_t>(sv.size());
                s.val = val;
                ++count_;
                return {&s.val, true};
            }
            if (s.hash == h && s.len == sv.size()) {
                std::string_view stored = arena_.get(s.off, s.len);
                if (sv.size() == 0 || std::memcmp(stored.data(), sv.data(), sv.size()) == 0) {
                    return {&s.val, false};
                }
            }
            idx = (idx + 1) & mask;
        }
    }

    void clear_and_free() {
        std::vector<Slot>().swap(table_);
        arena_.free_all();
        count_ = 0;
    }

   private:
    void rehash(size_t new_cap) {
        std::vector<Slot> next(new_cap);
        const size_t mask = new_cap - 1;
        for (const Slot& s : table_) {
            if (s.hash == 0) continue;
            size_t idx = static_cast<size_t>(s.hash) & mask;
            while (next[idx].hash != 0) idx = (idx + 1) & mask;
            next[idx] = s;
        }
        table_.swap(next);
    }

    OverflowArena arena_;
    std::vector<Slot> table_;
    size_t count_ = 0;
};

template <typename T, size_t BlockCap = ((65536u / sizeof(T)) >= 256u ? (65536u / sizeof(T)) : 256u)>
class ChunkedVec {
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(std::is_trivially_destructible_v<T>);

   public:
    ChunkedVec() = default;
    ChunkedVec(const ChunkedVec&) = delete;
    ChunkedVec& operator=(const ChunkedVec&) = delete;
    ChunkedVec(ChunkedVec&& other) noexcept
        : blocks_(std::move(other.blocks_)), count_(other.count_), tail_pos_(other.tail_pos_) {
        other.count_ = 0;
        other.tail_pos_ = BlockCap;
    }
    ChunkedVec& operator=(ChunkedVec&& other) noexcept {
        if (this != &other) {
            clear_and_free();
            blocks_ = std::move(other.blocks_);
            count_ = other.count_;
            tail_pos_ = other.tail_pos_;
            other.count_ = 0;
            other.tail_pos_ = BlockCap;
        }
        return *this;
    }
    ~ChunkedVec() { clear_and_free(); }

    void push_back(const T& item) {
        if (tail_pos_ == BlockCap) {
            blocks_.push_back(static_cast<T*>(::operator new[](sizeof(T) * BlockCap)));
            tail_pos_ = 0;
        }
        blocks_.back()[tail_pos_++] = item;
        ++count_;
    }

    size_t size() const noexcept { return count_; }
    bool empty() const noexcept { return count_ == 0; }

    template <typename Fn>
    void for_each_block(Fn&& fn) const {
        if (blocks_.empty()) return;
        const size_t nblocks = blocks_.size();
        for (size_t i = 0; i + 1 < nblocks; ++i) {
            fn(blocks_[i], BlockCap);
        }
        fn(blocks_.back(), tail_pos_);
    }

    template <typename Fn>
    void for_each(Fn&& fn) const {
        for_each_block([&](const T* ptr, size_t n) {
            for (size_t i = 0; i < n; ++i) {
                fn(ptr[i]);
            }
        });
    }

    void clear_and_free() noexcept {
        for (T* b : blocks_) {
            ::operator delete[](b);
        }
        std::vector<T*>().swap(blocks_);
        count_ = 0;
        tail_pos_ = BlockCap;
    }

   private:
    std::vector<T*> blocks_;
    size_t count_ = 0;
    size_t tail_pos_ = BlockCap;
};
