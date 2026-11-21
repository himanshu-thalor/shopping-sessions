#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace detail_sort {

[[gnu::always_inline]] inline void cswap_u64(uint64_t& a, uint64_t& b) noexcept {
    const uint64_t x = a;
    const uint64_t y = b;
    a = (x < y) ? x : y;
    b = (x < y) ? y : x;
}

template <size_t P1, size_t N1, size_t P2, size_t N2>
[[gnu::always_inline]] inline void bn_merge(uint64_t* a) noexcept {
    if constexpr (N1 == 1 && N2 == 1) {
        cswap_u64(a[P1], a[P2]);
    } else if constexpr (N1 == 1 && N2 == 2) {
        cswap_u64(a[P1], a[P2 + 1]);
        cswap_u64(a[P1], a[P2]);
    } else if constexpr (N1 == 2 && N2 == 1) {
        cswap_u64(a[P1], a[P2]);
        cswap_u64(a[P1 + 1], a[P2]);
    } else {
        constexpr size_t M1 = N1 / 2;
        constexpr size_t M2 = (N1 & 1u) ? (N2 / 2) : ((N2 + 1) / 2);
        bn_merge<P1, M1, P2, M2>(a);
        bn_merge<P1 + M1, N1 - M1, P2 + M2, N2 - M2>(a);
        bn_merge<P1 + M1, N1 - M1, P2, M2>(a);
    }
}

template <size_t Pos, size_t Len>
[[gnu::always_inline]] inline void bn_sort(uint64_t* a) noexcept {
    if constexpr (Len <= 1) {
        return;
    } else {
        constexpr size_t M = Len / 2;
        bn_sort<Pos, M>(a);
        bn_sort<Pos + M, Len - M>(a);
        bn_merge<Pos, M, Pos + M, Len - M>(a);
    }
}

[[gnu::always_inline]] inline void sort_u64_le16(uint64_t* arr, size_t n) noexcept {
    switch (n) {
        case 0:
        case 1:
            return;
        case 2:
            bn_sort<0, 2>(arr);
            return;
        case 3:
            bn_sort<0, 3>(arr);
            return;
        case 4:
            bn_sort<0, 4>(arr);
            return;
        case 5:
            bn_sort<0, 5>(arr);
            return;
        case 6:
            bn_sort<0, 6>(arr);
            return;
        case 7:
            bn_sort<0, 7>(arr);
            return;
        case 8:
            bn_sort<0, 8>(arr);
            return;
        case 9:
            bn_sort<0, 9>(arr);
            return;
        case 10:
            bn_sort<0, 10>(arr);
            return;
        case 11:
            bn_sort<0, 11>(arr);
            return;
        case 12:
            bn_sort<0, 12>(arr);
            return;
        case 13:
            bn_sort<0, 13>(arr);
            return;
        case 14:
            bn_sort<0, 14>(arr);
            return;
        case 15:
            bn_sort<0, 15>(arr);
            return;
        case 16:
            bn_sort<0, 16>(arr);
            return;
        default:
            return;
    }
}

[[gnu::noinline]] inline void sort_u64_17_32(uint64_t* arr, size_t n) noexcept {
    bn_sort<0, 16>(arr);
    const size_t nb = n - 16;
    sort_u64_le16(arr + 16, nb);

    uint64_t tmp[32];
    std::memcpy(tmp, arr, n * sizeof(uint64_t));
    const uint64_t* const a = tmp;
    const uint64_t* const b = tmp + 16;
    size_t ia = 0, ib = 0, k = 0;
    while (ia < 16 && ib < nb) {
        const uint64_t va = a[ia];
        const uint64_t vb = b[ib];
        const bool take_a = (va <= vb);
        arr[k++] = take_a ? va : vb;
        ia += static_cast<size_t>(take_a);
        ib += static_cast<size_t>(!take_a);
    }
    if (ia < 16) {
        std::memcpy(arr + k, a + ia, (16 - ia) * sizeof(uint64_t));
    } else if (ib < nb) {
        std::memcpy(arr + k, b + ib, (nb - ib) * sizeof(uint64_t));
    }
}

[[gnu::always_inline]] inline void sort_u64_le32(uint64_t* arr, size_t n,
                                                 std::vector<uint64_t>&) noexcept {
    if (n <= 16) {
        sort_u64_le16(arr, n);
    } else {
        sort_u64_17_32(arr, n);
    }
}

[[gnu::noinline]] inline void sort_u64_33_64(uint64_t* arr, size_t n) noexcept {
    bn_sort<0, 16>(arr);
    bn_sort<16, 16>(arr);
    bn_merge<0, 16, 16, 16>(arr);
    const size_t nb = n - 32;
    if (nb <= 16) {
        sort_u64_le16(arr + 32, nb);
    } else {
        sort_u64_17_32(arr + 32, nb);
    }

    uint64_t tmp[64];
    std::memcpy(tmp, arr, n * sizeof(uint64_t));
    const uint64_t* const a = tmp;
    const uint64_t* const b = tmp + 32;
    size_t ia = 0, ib = 0, k = 0;
    while (ia < 32 && ib < nb) {
        const uint64_t va = a[ia];
        const uint64_t vb = b[ib];
        const bool take_a = (va <= vb);
        arr[k++] = take_a ? va : vb;
        ia += static_cast<size_t>(take_a);
        ib += static_cast<size_t>(!take_a);
    }
    if (ia < 32) {
        std::memcpy(arr + k, a + ia, (32 - ia) * sizeof(uint64_t));
    } else if (ib < nb) {
        std::memcpy(arr + k, b + ib, (nb - ib) * sizeof(uint64_t));
    }
}

// Branchless Edelkamp-Weiss BlockQuicksort for n in [33, 255].
inline void block_quicksort_u64(uint64_t* lo, uint64_t* hi, std::vector<uint64_t>& scratch) {
    static constexpr size_t kBlk = 16;
    while (static_cast<size_t>(hi - lo) > 32) {
        const size_t n = static_cast<size_t>(hi - lo);
        uint64_t* mid = lo + (n >> 1);
        uint64_t* last = hi - 1;
        cswap_u64(*lo, *mid);
        cswap_u64(*lo, *last);
        cswap_u64(*mid, *last);
        std::swap(*lo, *mid);
        const uint64_t piv = *lo;

        uint8_t off_l[kBlk], off_r[kBlk];
        uint64_t* blk_l = lo + 1;
        uint64_t* scan_l = lo + 1;
        uint64_t* blk_r = hi;
        uint64_t* scan_r = hi;
        size_t pos_l = 0, num_l = 0;
        size_t pos_r = 0, num_r = 0;

        for (;;) {
            if (pos_l == num_l) {
                blk_l = scan_l;
                pos_l = 0;
                num_l = 0;
                const size_t avail = static_cast<size_t>(scan_r - scan_l);
                if (avail < (pos_r == num_r ? 2 * kBlk : kBlk)) break;
                for (size_t i = 0; i < kBlk; ++i) {
                    off_l[num_l] = static_cast<uint8_t>(i);
                    num_l += static_cast<size_t>(scan_l[i] >= piv);
                }
                scan_l += kBlk;
                if (num_l == 0) continue;
            }
            if (pos_r == num_r) {
                blk_r = scan_r;
                pos_r = 0;
                num_r = 0;
                const size_t avail = static_cast<size_t>(scan_r - scan_l);
                if (avail < kBlk) break;
                scan_r -= kBlk;
                for (size_t i = 0; i < kBlk; ++i) {
                    off_r[num_r] = static_cast<uint8_t>(i);
                    num_r += static_cast<size_t>(scan_r[kBlk - 1 - i] <= piv);
                }
                if (num_r == 0) continue;
            }
            const size_t sw = std::min(num_l - pos_l, num_r - pos_r);
            for (size_t i = 0; i < sw; ++i) {
                std::swap(blk_l[off_l[pos_l + i]], blk_r[-1 - static_cast<ptrdiff_t>(off_r[pos_r + i])]);
            }
            pos_l += sw;
            pos_r += sw;
        }

        if (pos_r < num_r) {
            for (size_t k = num_r; k > pos_r; --k) {
                std::swap(*(blk_r - 1 - static_cast<ptrdiff_t>(off_r[k - 1])), *scan_r);
                ++scan_r;
            }
        }
        if (pos_l < num_l) {
            for (size_t k = num_l; k > pos_l; --k) {
                --scan_l;
                std::swap(blk_l[off_l[k - 1]], *scan_l);
            }
        }

        uint64_t* i = scan_l;
        uint64_t* j = scan_r;
        for (;;) {
            while (i < j && *i < piv) ++i;
            while (j > i && *(j - 1) > piv) --j;
            if (i >= j) break;
            --j;
            std::swap(*i, *j);
            ++i;
        }

        uint64_t* piv_pos = i - 1;
        std::swap(*lo, *piv_pos);

        if (piv_pos - lo < hi - (piv_pos + 1)) {
            block_quicksort_u64(lo, piv_pos, scratch);
            lo = piv_pos + 1;
        } else {
            block_quicksort_u64(piv_pos + 1, hi, scratch);
            hi = piv_pos;
        }
    }
    sort_u64_le32(lo, static_cast<size_t>(hi - lo), scratch);
}

}  // namespace detail_sort

// Sort full 64-bit unsigned keys.
inline void sort_u64(uint64_t* arr, size_t n, std::vector<uint64_t>& scratch) {
    if (n <= 1) return;
    if (n <= 32) {
        detail_sort::sort_u64_le32(arr, n, scratch);
        return;
    }
    if (n <= 64) {
        detail_sort::sort_u64_33_64(arr, n);
        return;
    }
    if (n < 256) {
        detail_sort::block_quicksort_u64(arr, arr + n, scratch);
        return;
    }

    uint32_t cnt[6][2048]{};
    for (size_t i = 0; i < n; ++i) {
        const uint64_t v = arr[i];
        ++cnt[0][v & 0x7FFu];
        ++cnt[1][(v >> 11) & 0x7FFu];
        ++cnt[2][(v >> 22) & 0x7FFu];
        ++cnt[3][(v >> 33) & 0x7FFu];
        ++cnt[4][(v >> 44) & 0x7FFu];
        ++cnt[5][v >> 55];
    }

    if (scratch.size() < n) scratch.resize(n);
    uint64_t* src = arr;
    uint64_t* dst = scratch.data();

    static constexpr unsigned kShifts[6] = {0, 11, 22, 33, 44, 55};
    static constexpr size_t kRads[6] = {2048, 2048, 2048, 2048, 2048, 512};
    static constexpr uint64_t kMasks[6] = {0x7FFu, 0x7FFu, 0x7FFu, 0x7FFu, 0x7FFu, 0x1FFu};

    for (int p = 0; p < 6; ++p) {
        const size_t rad = kRads[p];
        bool single = false;
        for (size_t b = 0; b < rad; ++b) {
            if (cnt[p][b] == n) {
                single = true;
                break;
            }
        }
        if (single) continue;
        uint32_t off[2048];
        uint32_t sum = 0;
        for (size_t b = 0; b < rad; ++b) {
            off[b] = sum;
            sum += cnt[p][b];
        }
        const unsigned sh = kShifts[p];
        const uint64_t mk = kMasks[p];
        for (size_t i = 0; i < n; ++i) {
            const size_t k = static_cast<size_t>((src[i] >> sh) & mk);
            dst[off[k]++] = src[i];
        }
        std::swap(src, dst);
    }

    if (src != arr) {
        std::memcpy(arr, src, n * sizeof(uint64_t));
    }
}

// Sort 64-bit values where only the 31-bit key in bits [32..62] determines order.
inline void sort_u64_hi31(uint64_t* arr, size_t n, std::vector<uint64_t>& scratch) {
    if (n <= 1) return;
    if (n <= 32) {
        detail_sort::sort_u64_le32(arr, n, scratch);
        return;
    }
    if (n <= 64) {
        detail_sort::sort_u64_33_64(arr, n);
        return;
    }
    if (n < 256) {
        detail_sort::block_quicksort_u64(arr, arr + n, scratch);
        return;
    }

    uint32_t cnt0[2048]{};
    uint32_t cnt1[2048]{};
    uint32_t cnt2[512]{};
    for (size_t i = 0; i < n; ++i) {
        const uint32_t k = static_cast<uint32_t>(arr[i] >> 32);
        ++cnt0[k & 0x7FFu];
        ++cnt1[(k >> 11) & 0x7FFu];
        ++cnt2[k >> 22];
    }

    if (scratch.size() < n) scratch.resize(n);
    uint64_t* src = arr;
    uint64_t* dst = scratch.data();

    auto do_pass = [&](const uint32_t* cnt, size_t rad, unsigned shift, uint32_t mask) {
        for (size_t b = 0; b < rad; ++b) {
            if (cnt[b] == n) return;
        }
        uint32_t off[2048];
        uint32_t sum = 0;
        for (size_t b = 0; b < rad; ++b) {
            off[b] = sum;
            sum += cnt[b];
        }
        for (size_t i = 0; i < n; ++i) {
            const uint32_t k = static_cast<uint32_t>(src[i] >> shift) & mask;
            dst[off[k]++] = src[i];
        }
        std::swap(src, dst);
    };

    do_pass(cnt0, 2048, 32, 0x7FFu);
    do_pass(cnt1, 2048, 43, 0x7FFu);
    do_pass(cnt2, 512, 54, 0x1FFu);

    if (src != arr) {
        std::memcpy(arr, src, n * sizeof(uint64_t));
    }
}
