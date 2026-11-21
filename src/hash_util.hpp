#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <string_view>
#include <sys/mman.h>
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

inline uint64_t hash_inline_12(uint64_t lo, uint32_t hi) noexcept {
    uint64_t h = wyhash_mix(lo ^ 0xa0761d6478bd642full,
                            static_cast<uint64_t>(hi) ^ 0xe7037ed1a0b428dbull);
    h = wyhash_mix(h, 0x8ebc6af09c88c6e3ull);
    return h ? h : 1;
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

class HugeBlockPool {
   public:
    explicit HugeBlockPool(size_t block_bytes) : block_bytes_(block_bytes) {}
    HugeBlockPool(const HugeBlockPool&) = delete;
    HugeBlockPool& operator=(const HugeBlockPool&) = delete;

    ~HugeBlockPool() {
        for (const auto& r : regions_) {
            ::munmap(r.ptr, r.bytes);
        }
    }

    size_t block_bytes() const noexcept { return block_bytes_; }

    size_t alloc_batch(void** out, size_t max_n) {
        std::lock_guard<std::mutex> lk(mu_);
        size_t got = 0;
        while (got < max_n && !free_list_.empty()) {
            out[got++] = free_list_.back();
            free_list_.pop_back();
        }
        while (got < max_n) {
            grow_locked();
            while (got < max_n && !free_list_.empty()) {
                out[got++] = free_list_.back();
                free_list_.pop_back();
            }
        }
        return got;
    }

    void free_batch(void* const* in, size_t n) {
        if (n == 0) return;
        std::lock_guard<std::mutex> lk(mu_);
        free_list_.insert(free_list_.end(), in, in + n);
    }

   private:
    struct Region {
        void* ptr;
        size_t bytes;
    };

    void grow_locked() {
        constexpr size_t kHuge = size_t(2) << 20;
        const size_t reg_bytes = next_reg_bytes_;
        if (next_reg_bytes_ < (size_t(16) << 20)) next_reg_bytes_ <<= 1;

        const size_t map_sz = reg_bytes + kHuge;
        void* raw = ::mmap(nullptr, map_sz, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) throw std::bad_alloc();

        uintptr_t addr = reinterpret_cast<uintptr_t>(raw);
        uintptr_t aligned = (addr + kHuge - 1) & ~(uintptr_t(kHuge) - 1);
        if (aligned > addr) ::munmap(raw, aligned - addr);
        const size_t tail = (addr + map_sz) - (aligned + reg_bytes);
        if (tail > 0) ::munmap(reinterpret_cast<void*>(aligned + reg_bytes), tail);
        ::madvise(reinterpret_cast<void*>(aligned), reg_bytes, MADV_HUGEPAGE);

        regions_.push_back(Region{reinterpret_cast<void*>(aligned), reg_bytes});
        char* base = reinterpret_cast<char*>(aligned);
        const size_t nblocks = reg_bytes / block_bytes_;
        free_list_.reserve(free_list_.size() + nblocks);
        // Push in reverse so pops yield ascending addresses
        for (size_t i = nblocks; i-- > 0;) {
            free_list_.push_back(base + i * block_bytes_);
        }
    }

    size_t block_bytes_;
    size_t next_reg_bytes_ = size_t(2) << 20;
    std::mutex mu_;
    std::vector<Region> regions_;
    std::vector<void*> free_list_;
};

class ThreadBlockCache {
   public:
    explicit ThreadBlockCache(HugeBlockPool* pool = nullptr) : pool_(pool) {}

    ~ThreadBlockCache() { flush_all(); }

    void init(HugeBlockPool* pool) {
        flush_all();
        pool_ = pool;
    }

    void* alloc() {
        if (cache_.empty()) {
            void* tmp[8];
            const size_t got = pool_->alloc_batch(tmp, 8);
            cache_.insert(cache_.end(), tmp, tmp + got);
        }
        void* p = cache_.back();
        cache_.pop_back();
        return p;
    }

    void free(void* p) {
        cache_.push_back(p);
        if (cache_.size() >= 64) {
            pool_->free_batch(cache_.data() + 32, cache_.size() - 32);
            cache_.resize(32);
        }
    }

    void flush_all() {
        if (pool_ && !cache_.empty()) {
            pool_->free_batch(cache_.data(), cache_.size());
            cache_.clear();
        }
    }

   private:
    HugeBlockPool* pool_ = nullptr;
    std::vector<void*> cache_;
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
        if (sv.size() >= 8) {
            // Overlap the two fixed-width loads for 8..11 byte identifiers.
            std::memcpy(s.d, sv.data(), 8);
            std::memcpy(s.d + sv.size() - 4, sv.data() + sv.size() - 4, 4);
        } else if (!sv.empty()) {
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

inline uint64_t fast_str_hash(std::string_view sv) noexcept {
    if (sv.size() <= 11) {
        CompactStr cs = CompactStr::make_inline(sv);
        uint64_t lo;
        uint32_t hi;
        std::memcpy(&lo, cs.d, 8);
        std::memcpy(&hi, cs.d + 8, 4);
        return hash_inline_12(lo, hi);
    }
    const uint64_t h = wyhash64(sv);
    return h ? h : 1;
}

inline uint64_t fast_compact_hash(const CompactStr& cs, const OverflowArena* arenas) noexcept {
    if (cs.is_inline()) {
        uint64_t lo;
        uint32_t hi;
        std::memcpy(&lo, cs.d, 8);
        std::memcpy(&hi, cs.d + 8, 4);
        return hash_inline_12(lo, hi);
    }
    const uint64_t h = wyhash64(cs.raw_view(arenas));
    return h ? h : 1;
}

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
        const uint64_t h = fast_str_hash(sv);
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
        const uint64_t h = fast_str_hash(sv);
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

template <typename T, size_t BlockCap = 1024>
class ChunkedVec {
    static_assert((BlockCap & (BlockCap - 1)) == 0);
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(std::is_trivially_destructible_v<T>);

   public:
    explicit ChunkedVec(ThreadBlockCache* cache = nullptr) : cache_(cache) {}
    ChunkedVec(const ChunkedVec&) = delete;
    ChunkedVec& operator=(const ChunkedVec&) = delete;
    ChunkedVec(ChunkedVec&& other) noexcept
        : cache_(other.cache_),
          blocks_(std::move(other.blocks_)),
          cur_(other.cur_),
          end_(other.end_),
          full_count_(other.full_count_) {
        other.cur_ = nullptr;
        other.end_ = nullptr;
        other.full_count_ = 0;
    }
    ChunkedVec& operator=(ChunkedVec&& other) noexcept {
        if (this != &other) {
            clear_and_free();
            cache_ = other.cache_;
            blocks_ = std::move(other.blocks_);
            cur_ = other.cur_;
            end_ = other.end_;
            full_count_ = other.full_count_;
            other.cur_ = nullptr;
            other.end_ = nullptr;
            other.full_count_ = 0;
        }
        return *this;
    }
    ~ChunkedVec() { clear_and_free(); }

    void set_cache(ThreadBlockCache* cache) noexcept { cache_ = cache; }

    void push_back(const T& item) {
        if (__builtin_expect(cur_ == end_, 0)) {
            grow();
        }
        *cur_++ = item;
    }

    size_t size() const noexcept {
        if (blocks_.empty()) return 0;
        return full_count_ + static_cast<size_t>(cur_ - blocks_.back());
    }

    bool empty() const noexcept { return blocks_.empty(); }

    size_t num_blocks() const noexcept { return blocks_.size(); }

    T* block_ptr(size_t bi) const noexcept { return blocks_[bi]; }

    size_t block_len(size_t bi) const noexcept {
        return (bi + 1 < blocks_.size()) ? BlockCap
                                         : static_cast<size_t>(cur_ - blocks_.back());
    }

    template <typename Fn>
    void for_each_block(Fn&& fn) const {
        if (blocks_.empty()) return;
        const size_t nblocks = blocks_.size();
        for (size_t i = 0; i + 1 < nblocks; ++i) {
            fn(blocks_[i], BlockCap);
        }
        fn(blocks_.back(), static_cast<size_t>(cur_ - blocks_.back()));
    }

    void free_to(ThreadBlockCache* target_cache) noexcept {
        for (T* b : blocks_) {
            if (target_cache) {
                target_cache->free(b);
            } else {
                ::operator delete[](b);
            }
        }
        std::vector<T*>().swap(blocks_);
        cur_ = nullptr;
        end_ = nullptr;
        full_count_ = 0;
    }

    void clear_and_free() noexcept { free_to(cache_); }

   private:
    __attribute__((noinline)) void grow() {
        if (!blocks_.empty()) {
            full_count_ += BlockCap;
        }
        T* b = cache_ ? static_cast<T*>(cache_->alloc())
                      : static_cast<T*>(::operator new[](sizeof(T) * BlockCap));
        blocks_.push_back(b);
        cur_ = b;
        end_ = b + BlockCap;
    }

    ThreadBlockCache* cache_ = nullptr;
    std::vector<T*> blocks_;
    T* cur_ = nullptr;
    T* end_ = nullptr;
    size_t full_count_ = 0;
};
