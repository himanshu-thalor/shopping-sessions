#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>
#include <string_view>
#include <sys/mman.h>
#include <type_traits>
#include <utility>
#include <vector>

[[gnu::always_inline]] inline uint64_t wyhash_mix(uint64_t a, uint64_t b) noexcept {
    __uint128_t r = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
    return static_cast<uint64_t>(r) ^ static_cast<uint64_t>(r >> 64);
}

[[gnu::always_inline]] inline uint64_t wyhash_read8(const char* p) noexcept {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

[[gnu::always_inline]] inline uint64_t wyhash_read4(const char* p) noexcept {
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

template <typename T>
class HugeBuffer {
   public:
    HugeBuffer() = default;
    HugeBuffer(const HugeBuffer&) = delete;
    HugeBuffer& operator=(const HugeBuffer&) = delete;
    HugeBuffer(HugeBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_), mapped_bytes_(other.mapped_bytes_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
        other.mapped_bytes_ = 0;
    }
    HugeBuffer& operator=(HugeBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            mapped_bytes_ = other.mapped_bytes_;
            other.ptr_ = nullptr;
            other.count_ = 0;
            other.mapped_bytes_ = 0;
        }
        return *this;
    }
    ~HugeBuffer() { release(); }

    void assign_zero(size_t n) {
        release();
        if (n == 0) return;
        const size_t bytes = n * sizeof(T);
        constexpr size_t kHuge = size_t(2) << 20;
        if (bytes >= kHuge) {
            const size_t aligned_bytes = (bytes + kHuge - 1) & ~(kHuge - 1);
            const size_t map_sz = aligned_bytes + kHuge;
            void* raw = ::mmap(nullptr, map_sz, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (raw != MAP_FAILED) {
                uintptr_t addr = reinterpret_cast<uintptr_t>(raw);
                uintptr_t aligned = (addr + kHuge - 1) & ~(uintptr_t(kHuge) - 1);
                if (aligned > addr) ::munmap(raw, aligned - addr);
                size_t tail = (addr + map_sz) - (aligned + aligned_bytes);
                if (tail > 0) ::munmap(reinterpret_cast<void*>(aligned + aligned_bytes), tail);
                ::madvise(reinterpret_cast<void*>(aligned), aligned_bytes, MADV_HUGEPAGE);
                ptr_ = reinterpret_cast<T*>(aligned);
                count_ = n;
                mapped_bytes_ = aligned_bytes;
                return;
            }
        }
        ptr_ = new T[n]{};
        count_ = n;
        mapped_bytes_ = 0;
    }

    void release() noexcept {
        if (ptr_) {
            if (mapped_bytes_ > 0) {
                ::munmap(ptr_, mapped_bytes_);
            } else {
                delete[] ptr_;
            }
        }
        ptr_ = nullptr;
        count_ = 0;
        mapped_bytes_ = 0;
    }

    T& operator[](size_t i) noexcept { return ptr_[i]; }
    const T& operator[](size_t i) const noexcept { return ptr_[i]; }
    size_t size() const noexcept { return count_; }
    bool empty() const noexcept { return count_ == 0; }
    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }

   private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
    size_t mapped_bytes_ = 0;
};

// Per-worker lock-free 16 KiB block allocator backed by 2 MiB-aligned THP slabs.
class ThreadBlockArena {
   public:
    static constexpr size_t kBlockBytes = 8192;

    ThreadBlockArena() = default;
    ThreadBlockArena(const ThreadBlockArena&) = delete;
    ThreadBlockArena& operator=(const ThreadBlockArena&) = delete;

    ThreadBlockArena(ThreadBlockArena&& other) noexcept
        : slabs_(std::move(other.slabs_)),
          free_blocks_(std::move(other.free_blocks_)),
          slab_cur_(other.slab_cur_),
          slab_end_(other.slab_end_),
          next_slab_bytes_(other.next_slab_bytes_) {
        other.slab_cur_ = nullptr;
        other.slab_end_ = nullptr;
    }

    ThreadBlockArena& operator=(ThreadBlockArena&& other) noexcept {
        if (this != &other) {
            release_all();
            slabs_ = std::move(other.slabs_);
            free_blocks_ = std::move(other.free_blocks_);
            slab_cur_ = other.slab_cur_;
            slab_end_ = other.slab_end_;
            next_slab_bytes_ = other.next_slab_bytes_;
            other.slab_cur_ = nullptr;
            other.slab_end_ = nullptr;
        }
        return *this;
    }

    ~ThreadBlockArena() { release_all(); }

    void* alloc() {
        if (__builtin_expect(!free_blocks_.empty(), 1)) {
            void* p = free_blocks_.back();
            free_blocks_.pop_back();
            return p;
        }
        if (__builtin_expect(slab_cur_ == slab_end_, 0)) {
            alloc_slab();
        }
        void* p = slab_cur_;
        slab_cur_ += kBlockBytes;
        return p;
    }

    void recycle(void* p) {
        if (p) free_blocks_.push_back(p);
    }

    // Reuse all already-faulted 2-MiB THP slabs as a zero-fault contiguous bump arena!
    void reset_to_bump() noexcept {
        free_blocks_.clear();
        bump_slab_idx_ = 0;
        if (!slabs_.empty()) {
            slab_cur_ = static_cast<char*>(slabs_[0].ptr);
            slab_end_ = slab_cur_ + slabs_[0].bytes;
        } else {
            slab_cur_ = nullptr;
            slab_end_ = nullptr;
        }
    }

    template <typename T>
    T* bump_alloc(size_t count) {
        if (count == 0) return nullptr;
        const size_t bytes = ((count * sizeof(T)) + 63u) & ~size_t(63u);
        while (__builtin_expect(static_cast<size_t>(slab_end_ - slab_cur_) < bytes, 0)) {
            if (bump_slab_idx_ + 1 < slabs_.size()) {
                ++bump_slab_idx_;
                slab_cur_ = static_cast<char*>(slabs_[bump_slab_idx_].ptr);
                slab_end_ = slab_cur_ + slabs_[bump_slab_idx_].bytes;
            } else {
                alloc_slab(bytes);
                bump_slab_idx_ = slabs_.size() - 1;
            }
        }
        T* p = reinterpret_cast<T*>(slab_cur_);
        slab_cur_ += bytes;
        return p;
    }

    void release_all() noexcept {
        for (const auto& s : slabs_) {
            ::munmap(s.ptr, s.bytes);
        }
        std::vector<Slab>().swap(slabs_);
        std::vector<void*>().swap(free_blocks_);
        slab_cur_ = nullptr;
        slab_end_ = nullptr;
        bump_slab_idx_ = 0;
    }

   private:
    struct Slab {
        void* ptr;
        size_t bytes;
    };

    __attribute__((noinline)) void alloc_slab(size_t min_bytes = 0) {
        constexpr size_t kHuge = size_t(2) << 20;
        size_t slab_bytes = next_slab_bytes_;
        if (next_slab_bytes_ < (size_t(16) << 20)) next_slab_bytes_ <<= 1;
        if (min_bytes > slab_bytes) {
            slab_bytes = (min_bytes + kHuge - 1) & ~(kHuge - 1);
        }

        const size_t map_sz = slab_bytes + kHuge;
        void* raw = ::mmap(nullptr, map_sz, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) throw std::bad_alloc();

        const uintptr_t addr = reinterpret_cast<uintptr_t>(raw);
        const uintptr_t aligned = (addr + kHuge - 1) & ~(uintptr_t(kHuge) - 1);
        if (aligned > addr) ::munmap(raw, aligned - addr);
        const size_t tail = (addr + map_sz) - (aligned + slab_bytes);
        if (tail > 0) ::munmap(reinterpret_cast<void*>(aligned + slab_bytes), tail);
        ::madvise(reinterpret_cast<void*>(aligned), slab_bytes, MADV_HUGEPAGE);

        slabs_.push_back(Slab{reinterpret_cast<void*>(aligned), slab_bytes});
        slab_cur_ = reinterpret_cast<char*>(aligned);
        slab_end_ = slab_cur_ + slab_bytes;
    }

    std::vector<Slab> slabs_;
    std::vector<void*> free_blocks_;
    char* slab_cur_ = nullptr;
    char* slab_end_ = nullptr;
    size_t next_slab_bytes_ = size_t(4) << 20;
    size_t bump_slab_idx_ = 0;
};

template <typename T>
struct BlockSlice {
    T* ptr;
    uint32_t len;
};

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

    [[gnu::noinline]] uint64_t push_with_len(std::string_view sv) {
        const uint32_t len = static_cast<uint32_t>(sv.size());
        const size_t need = 4 + static_cast<size_t>(len);
        if (need > BLOCK_SIZE) {
            char* big = new char[need];
            std::memcpy(big, &len, 4);
            if (len > 0) std::memcpy(big + 4, sv.data(), len);
            const uint64_t bidx = static_cast<uint64_t>(blocks.size());
            blocks.push_back(big);
            pos = BLOCK_SIZE;
            return bidx << BLOCK_SHIFT;
        }
        if (pos + need > BLOCK_SIZE) {
            blocks.push_back(new char[BLOCK_SIZE]);
            pos = 0;
        }
        const uint64_t bidx = static_cast<uint64_t>(blocks.size() - 1);
        const uint64_t off = (bidx << BLOCK_SHIFT) | static_cast<uint64_t>(pos);
        char* dst = blocks.back() + pos;
        std::memcpy(dst, &len, 4);
        if (len > 0) std::memcpy(dst + 4, sv.data(), len);
        pos += need;
        return off;
    }

    std::string_view get_len_str(uint64_t off) const noexcept {
        const char* p = blocks[off >> BLOCK_SHIFT] + (off & BLOCK_MASK);
        uint32_t len = 0;
        std::memcpy(&len, p, 4);
        return std::string_view(p + 4, len);
    }
};

// 8-byte compact ID optimized for "<Prefix><1..8 ASCII bytes>" (bit 63 == 0) with overflow fallback.
// 0 == empty sentinel. All hot paths are [[gnu::always_inline]]; all overflow paths are [[gnu::noinline]].
template <char Prefix>
struct FastId64 {
    uint64_t raw = 0;

    [[gnu::always_inline]] static inline uint64_t check_inline_word(uint64_t w,
                                                                    size_t len) noexcept {
        const uint64_t m = (~0ULL) >> ((9u - static_cast<unsigned>(len)) << 3);
        w &= m;
        const uint64_t active = w | ~m;
        const uint64_t bad =
            (w | ((active - 0x0101010101010101ULL) & ~active)) & 0x8080808080808080ULL;
        return __builtin_expect(bad == 0, 1) ? w : 0ULL;
    }

    // Caller guarantees at least 9 readable bytes starting at p.
    [[gnu::always_inline]] static inline FastId64 try_inline_padded(const char* p,
                                                                    size_t len) noexcept {
        if (__builtin_expect(len == 9 && p[0] == Prefix, 1)) {
            uint64_t w;
            std::memcpy(&w, p + 1, 8);
            const uint64_t bad =
                (w | ((w - 0x0101010101010101ULL) & ~w)) & 0x8080808080808080ULL;
            return __builtin_expect(bad == 0, 1) ? FastId64{w} : FastId64{0};
        }
        if (__builtin_expect(len - 2 <= 7 && p[0] == Prefix, 1)) {
            uint64_t w;
            std::memcpy(&w, p + 1, 8);
            return FastId64{check_inline_word(w, len)};
        }
        return FastId64{0};
    }

    [[gnu::always_inline]] static inline FastId64 try_inline_safe(std::string_view sv) noexcept {
        const size_t len = sv.size();
        if (__builtin_expect(len - 2 <= 7 && sv[0] == Prefix, 1)) {
            uint64_t w = 0;
            std::memcpy(&w, sv.data() + 1, len - 1);
            return FastId64{check_inline_word(w, len)};
        }
        return FastId64{0};
    }

    [[gnu::noinline]] static FastId64 make_overflow(std::string_view sv, uint16_t arena_id,
                                                    OverflowArena& arena) {
        const uint64_t off = arena.push_with_len(sv);
        return FastId64{(1ULL << 63) | (static_cast<uint64_t>(arena_id & 0x7FFFu) << 48) |
                        (off & 0x0000FFFFFFFFFFFFULL)};
    }

    // Caller guarantees at least 9 readable bytes at sv.data().
    [[gnu::always_inline]] static inline FastId64 make_str_padded(std::string_view sv,
                                                                  uint16_t arena_id,
                                                                  OverflowArena& arena) {
        FastId64 id = try_inline_padded(sv.data(), sv.size());
        if (__builtin_expect(id.raw != 0, 1)) return id;
        return make_overflow(sv, arena_id, arena);
    }

    [[gnu::always_inline]] static inline FastId64 make_str_safe(std::string_view sv,
                                                                uint16_t arena_id,
                                                                OverflowArena& arena) {
        FastId64 id = try_inline_safe(sv);
        if (__builtin_expect(id.raw != 0, 1)) return id;
        return make_overflow(sv, arena_id, arena);
    }

    [[gnu::always_inline]] inline bool empty() const noexcept { return raw == 0; }
    [[gnu::always_inline]] inline bool is_inline() const noexcept {
        return (raw >> 63) == 0 && raw != 0;
    }
    [[gnu::always_inline]] inline bool is_overflow() const noexcept { return (raw >> 63) != 0; }

    [[gnu::always_inline]] inline uint16_t overflow_arena_id() const noexcept {
        return static_cast<uint16_t>((raw >> 48) & 0x7FFFu);
    }

    [[gnu::always_inline]] inline uint64_t overflow_off() const noexcept {
        return raw & 0x0000FFFFFFFFFFFFULL;
    }

    [[gnu::always_inline]] inline uint32_t inline_len() const noexcept {
        return 9u - (static_cast<unsigned>(__builtin_clzll(raw)) >> 3);
    }

    // Caller guarantees dst has at least 16 writable bytes.
    [[gnu::always_inline]] inline uint32_t write_inline_to(char* dst) const noexcept {
        dst[0] = Prefix;
        std::memcpy(dst + 1, &raw, 8);
        return inline_len();
    }

    [[gnu::always_inline]] inline std::string_view view(const OverflowArena* const* arenas,
                                                        char (&scratch)[9]) const noexcept {
        if (__builtin_expect(is_inline(), 1)) {
            scratch[0] = Prefix;
            std::memcpy(scratch + 1, &raw, 8);
            return std::string_view(scratch, inline_len());
        }
        if (raw == 0) return {};
        return arenas[overflow_arena_id()]->get_len_str(overflow_off());
    }

    [[gnu::noinline]] uint64_t hash_overflow(const OverflowArena* const* arenas) const noexcept {
        return wyhash64(arenas[overflow_arena_id()]->get_len_str(overflow_off()));
    }

    [[gnu::always_inline]] inline uint64_t hash(const OverflowArena* const* arenas) const noexcept {
        if (__builtin_expect((raw >> 63) == 0, 1)) {
            return wyhash_mix(raw, 0xe7037ed1a0b428dbull);
        }
        return hash_overflow(arenas);
    }

    [[gnu::always_inline]] static inline uint64_t hash_sv(std::string_view sv) noexcept {
        FastId64 id = try_inline_safe(sv);
        if (__builtin_expect(id.raw != 0, 1)) {
            return wyhash_mix(id.raw, 0xe7037ed1a0b428dbull);
        }
        return wyhash64(sv);
    }

    [[gnu::noinline]] static bool equals_overflow(FastId64 a, FastId64 b,
                                                  const OverflowArena* const* arenas) noexcept {
        const std::string_view sa = arenas[a.overflow_arena_id()]->get_len_str(a.overflow_off());
        const std::string_view sb = arenas[b.overflow_arena_id()]->get_len_str(b.overflow_off());
        return sa.size() == sb.size() &&
               (sa.empty() || std::memcmp(sa.data(), sb.data(), sa.size()) == 0);
    }

    [[gnu::always_inline]] static inline bool equals(FastId64 a, FastId64 b,
                                                     const OverflowArena* const* arenas) noexcept {
        if (a.raw == b.raw) return true;
        if (__builtin_expect(((a.raw & b.raw) >> 63) == 0, 1)) return false;
        return equals_overflow(a, b, arenas);
    }

    [[gnu::noinline]] static bool less_overflow(FastId64 a, FastId64 b,
                                                const OverflowArena* const* arenas) noexcept {
        char sa_buf[9], sb_buf[9];
        return byte_less(a.view(arenas, sa_buf), b.view(arenas, sb_buf));
    }

    [[gnu::always_inline]] static inline bool less(FastId64 a, FastId64 b,
                                                   const OverflowArena* const* arenas) noexcept {
        if (__builtin_expect(((a.raw | b.raw) >> 63) == 0, 1)) {
            return __builtin_bswap64(a.raw) < __builtin_bswap64(b.raw);
        }
        return less_overflow(a, b, arenas);
    }

    bool operator==(const FastId64& o) const noexcept { return raw == o.raw; }
    bool operator!=(const FastId64& o) const noexcept { return raw != o.raw; }
};
static_assert(sizeof(FastId64<'u'>) == 8);
static_assert(std::is_trivially_copyable_v<FastId64<'u'>>);

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
        const uint64_t h = hash_sv(sv);
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
        const uint64_t h = hash_sv(sv);
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
    static uint64_t hash_sv(std::string_view sv) noexcept {
        const uint64_t h = wyhash64(sv);
        return h ? h : 1;
    }

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
