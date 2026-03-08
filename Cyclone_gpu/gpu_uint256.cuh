#ifndef GPU_UINT256_CUH
#define GPU_UINT256_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// 256-bit unsigned integer for GPU (4 × 64-bit words, little-endian)
typedef struct {
    uint64_t d[4];
} uint256_t;

// ============================================================
// Basic 256-bit arithmetic
// ============================================================

__device__ __forceinline__ void u256_set_zero(uint256_t *a) {
    a->d[0] = a->d[1] = a->d[2] = a->d[3] = 0ULL;
}

__device__ __forceinline__ void u256_copy(uint256_t *dst, const uint256_t *src) {
    dst->d[0] = src->d[0];
    dst->d[1] = src->d[1];
    dst->d[2] = src->d[2];
    dst->d[3] = src->d[3];
}

__device__ __forceinline__ void u256_set64(uint256_t *a, uint64_t v) {
    a->d[0] = v; a->d[1] = a->d[2] = a->d[3] = 0ULL;
}

__device__ __forceinline__ int u256_is_zero(const uint256_t *a) {
    return (a->d[0] == 0 && a->d[1] == 0 && a->d[2] == 0 && a->d[3] == 0);
}

// Compare: return 1 if a>b, 0 if a==b, -1 if a<b
__device__ __forceinline__ int u256_cmp(const uint256_t *a, const uint256_t *b) {
    for (int i = 3; i >= 0; i--) {
        if (a->d[i] > b->d[i]) return  1;
        if (a->d[i] < b->d[i]) return -1;
    }
    return 0;
}

// 256-bit addition, returns carry
__device__ __forceinline__ uint64_t u256_add(uint256_t *r,
                                              const uint256_t *a,
                                              const uint256_t *b)
{
    uint64_t c = 0;
    uint64_t t;
    t = a->d[0] + b->d[0]; c = (t < a->d[0]) ? 1 : 0; r->d[0] = t;
    t = a->d[1] + b->d[1]; uint64_t c2 = (t < a->d[1]) ? 1 : 0;
    t += c; c = c2 + ((t < c) ? 1 : 0); r->d[1] = t;
    t = a->d[2] + b->d[2]; c2 = (t < a->d[2]) ? 1 : 0;
    t += c; c = c2 + ((t < c) ? 1 : 0); r->d[2] = t;
    t = a->d[3] + b->d[3]; c2 = (t < a->d[3]) ? 1 : 0;
    t += c; c = c2 + ((t < c) ? 1 : 0); r->d[3] = t;
    return c;
}

// 256-bit subtraction (no borrow expected from outside)
__device__ __forceinline__ void u256_sub(uint256_t *r,
                                          const uint256_t *a,
                                          const uint256_t *b)
{
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t ai = a->d[i], bi = b->d[i];
        uint64_t t1 = ai - bi;
        uint64_t b1 = (ai < bi) ? 1ULL : 0ULL;
        uint64_t t2 = t1 - borrow;
        uint64_t b2 = (t1 < borrow) ? 1ULL : 0ULL;
        r->d[i] = t2;
        borrow = b1 + b2; // at most 1 (proved in comments)
    }
}

// Right-shift by 1 bit
__device__ __forceinline__ void u256_rshift1(uint256_t *r, const uint256_t *a) {
    r->d[0] = (a->d[0] >> 1) | (a->d[1] << 63);
    r->d[1] = (a->d[1] >> 1) | (a->d[2] << 63);
    r->d[2] = (a->d[2] >> 1) | (a->d[3] << 63);
    r->d[3] = (a->d[3] >> 1);
}

// Get bit at position pos (0 = LSB)
__device__ __forceinline__ int u256_bit(const uint256_t *a, int pos) {
    return (int)((a->d[pos >> 6] >> (pos & 63)) & 1ULL);
}

// ============================================================
// secp256k1 prime  p = 2^256 - 0x1000003D1
// p in words (little-endian): { 0xFFFFFFFEFFFFFC2F, 0xFFFFFFFFFFFFFFFF,
//                                0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF }
// ============================================================

__device__ __constant__ uint256_t GPU_SECP256K1_P = {{
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
}};

// beta = 2^32 + 977 = 0x1000003D1  (p = 2^256 - beta)
static __constant__ uint64_t GPU_K1_BETA = 0x1000003D1ULL;

// ============================================================
// Fast secp256k1 modular addition / subtraction
// ============================================================

__device__ __forceinline__ void fe_add(uint256_t *r,
                                        const uint256_t *a,
                                        const uint256_t *b)
{
    uint64_t carry = u256_add(r, a, b);
    if (carry || u256_cmp(r, &GPU_SECP256K1_P) >= 0) {
        u256_sub(r, r, &GPU_SECP256K1_P);
    }
}

__device__ __forceinline__ void fe_sub(uint256_t *r,
                                        const uint256_t *a,
                                        const uint256_t *b)
{
    if (u256_cmp(a, b) >= 0) {
        u256_sub(r, a, b);
    } else {
        uint256_t tmp;
        u256_sub(&tmp, a, b);        // underflows to 2^256 + (a-b)
        u256_add(r, &tmp, &GPU_SECP256K1_P);
    }
}

__device__ __forceinline__ void fe_neg(uint256_t *r, const uint256_t *a) {
    if (u256_is_zero(a)) { *r = *a; return; }
    u256_sub(r, &GPU_SECP256K1_P, a);
}

// ============================================================
// Fast secp256k1 modular multiplication: r = a * b mod p
// Uses p = 2^256 - beta for fast reduction (512 → 256 bits)
// ============================================================

// fe_mul using unsigned __int128 partial products
__device__ void fe_mul(uint256_t *r,
                        const uint256_t *a,
                        const uint256_t *b)
{
    // 256×256 → 512-bit schoolbook multiply using unsigned __int128
    typedef unsigned __int128 u128;
    uint64_t t[8] = {0,0,0,0,0,0,0,0};

    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            u128 acc = (u128)a->d[i] * b->d[j] + t[i+j] + carry;
            t[i+j] = (uint64_t)acc;
            carry   = (uint64_t)(acc >> 64);
        }
        t[i+4] += carry; // t[i+4] starts at 0 for i=0..3, so no overflow
    }

    // Reduce: high 256 bits (t[4..7]) × beta added to low 256 bits (t[0..3])
    // beta = 0x1000003D1, p = 2^256 - beta
    uint64_t carry2 = 0;
    for (int i = 0; i < 4; i++) {
        u128 acc = (u128)t[i+4] * GPU_K1_BETA + t[i] + carry2;
        t[i]   = (uint64_t)acc;
        carry2 = (uint64_t)(acc >> 64);
    }

    // Second reduction: remaining overflow × beta
    u128 acc2 = (u128)carry2 * GPU_K1_BETA;
    uint64_t carry3 = 0;
    u128 a0 = (u128)t[0] + (uint64_t)acc2;
    t[0] = (uint64_t)a0; carry3 = (uint64_t)(a0 >> 64);
    u128 a1 = (u128)t[1] + (uint64_t)(acc2 >> 64) + carry3;
    t[1] = (uint64_t)a1; carry3 = (uint64_t)(a1 >> 64);
    u128 a2 = (u128)t[2] + carry3;
    t[2] = (uint64_t)a2; carry3 = (uint64_t)(a2 >> 64);
    t[3] += carry3;

    r->d[0]=t[0]; r->d[1]=t[1]; r->d[2]=t[2]; r->d[3]=t[3];

    // Final conditional subtraction (very rare)
    if (u256_cmp(r, &GPU_SECP256K1_P) >= 0)
        u256_sub(r, r, &GPU_SECP256K1_P);
}

// Modular squaring: r = a^2 mod p
__device__ __forceinline__ void fe_sqr(uint256_t *r, const uint256_t *a) {
    fe_mul(r, a, a);
}

// ============================================================
// Modular inverse using Fermat's little theorem: a^(p-2) mod p
// Uses a chain of squarings and multiplications tailored for
// p = 2^256 - 2^32 - 977 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// ============================================================
__device__ void fe_inv(uint256_t *r, const uint256_t *a)
{
    // Compute a^(p-2) mod p using a fixed addition chain for p-2
    // p-2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    // Standard square-and-multiply using the bit pattern of p-2
    uint256_t base, tmp;
    u256_copy(&base, a);
    // Start with result = 1
    u256_set64(r, 1ULL);

    // Exponent e = p - 2 in little-endian bits:
    // word[0] = 0xFFFFFFFEFFFFFC2D, word[1..3] = 0xFFFFFFFFFFFFFFFF
    uint64_t exp[4] = {
        0xFFFFFFFEFFFFFC2DULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL
    };

    for (int i = 0; i < 256; i++) {
        uint64_t bit = (exp[i >> 6] >> (i & 63)) & 1ULL;
        if (bit) {
            fe_mul(&tmp, r, &base);
            u256_copy(r, &tmp);
        }
        if (i < 255) {
            fe_sqr(&tmp, &base);
            u256_copy(&base, &tmp);
        }
    }
}

// ============================================================
// Host versions of the above (for CPU-side GTable computation)
// These use __uint128_t instead of __umul64hi
// ============================================================

static inline void h_u256_copy(uint256_t *dst, const uint256_t *src) {
    dst->d[0] = src->d[0]; dst->d[1] = src->d[1];
    dst->d[2] = src->d[2]; dst->d[3] = src->d[3];
}
static inline void h_u256_set64(uint256_t *a, uint64_t v) {
    a->d[0] = v; a->d[1] = a->d[2] = a->d[3] = 0ULL;
}
static inline int h_u256_is_zero(const uint256_t *a) {
    return !(a->d[0]|a->d[1]|a->d[2]|a->d[3]);
}
static inline int h_u256_cmp(const uint256_t *a, const uint256_t *b) {
    for (int i = 3; i >= 0; i--) {
        if (a->d[i] > b->d[i]) return  1;
        if (a->d[i] < b->d[i]) return -1;
    }
    return 0;
}
static inline uint64_t h_u256_add(uint256_t *r,
                                    const uint256_t *a,
                                    const uint256_t *b)
{
    typedef unsigned __int128 u128;
    u128 acc;
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        acc = (u128)a->d[i] + b->d[i] + carry;
        r->d[i] = (uint64_t)acc;
        carry   = (uint64_t)(acc >> 64);
    }
    return carry;
}
static inline void h_u256_sub(uint256_t *r,
                                const uint256_t *a,
                                const uint256_t *b)
{
    typedef unsigned __int128 u128;
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        u128 t = (u128)a->d[i] - b->d[i] - borrow;
        r->d[i] = (uint64_t)t;
        borrow  = (t >> 127) & 1ULL;
    }
}

static const uint256_t H_P = {{
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
}};
static const uint64_t H_BETA = 0x1000003D1ULL;

static inline void h_fe_add(uint256_t *r,
                              const uint256_t *a, const uint256_t *b)
{
    uint64_t c = h_u256_add(r, a, b);
    if (c || h_u256_cmp(r, &H_P) >= 0) h_u256_sub(r, r, &H_P);
}
static inline void h_fe_sub(uint256_t *r,
                              const uint256_t *a, const uint256_t *b)
{
    if (h_u256_cmp(a, b) >= 0) { h_u256_sub(r, a, b); }
    else {
        uint256_t tmp; h_u256_sub(&tmp, a, b);
        h_u256_add(r, &tmp, &H_P);
    }
}
static inline void h_fe_mul(uint256_t *r,
                              const uint256_t *a, const uint256_t *b)
{
    typedef unsigned __int128 u128;
    uint64_t t[8] = {};
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            u128 acc = (u128)a->d[i] * b->d[j] + t[i+j] + carry;
            t[i+j]   = (uint64_t)acc;
            carry     = (uint64_t)(acc >> 64);
        }
        t[i+4] += carry;
    }
    // Reduce high 256 bits by beta
    uint64_t hi[5] = {};
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        u128 acc = (u128)t[i+4] * H_BETA + carry;
        hi[i] = (uint64_t)acc;
        carry  = (uint64_t)(acc >> 64);
    }
    hi[4] = carry;
    carry = 0;
    for (int i = 0; i < 4; i++) {
        u128 acc = (u128)t[i] + hi[i] + carry;
        t[i] = (uint64_t)acc;
        carry = (uint64_t)(acc >> 64);
    }
    uint64_t overflow = hi[4] + carry;
    // second reduction
    u128 acc2 = (u128)overflow * H_BETA;
    carry = 0;
    u128 a0 = (u128)t[0] + (uint64_t)acc2; t[0] = (uint64_t)a0; carry = (uint64_t)(a0>>64);
    u128 a1 = (u128)t[1] + (uint64_t)(acc2>>64) + carry; t[1]=(uint64_t)a1; carry=(uint64_t)(a1>>64);
    u128 a2 = (u128)t[2] + carry; t[2]=(uint64_t)a2; carry=(uint64_t)(a2>>64);
    u128 a3 = (u128)t[3] + carry; t[3]=(uint64_t)a3;
    r->d[0]=t[0]; r->d[1]=t[1]; r->d[2]=t[2]; r->d[3]=t[3];
    if (h_u256_cmp(r, &H_P) >= 0) h_u256_sub(r, r, &H_P);
}
static inline void h_fe_sqr(uint256_t *r, const uint256_t *a) {
    h_fe_mul(r, a, a);
}
static inline void h_fe_inv(uint256_t *r, const uint256_t *a)
{
    uint256_t base, tmp;
    h_u256_copy(&base, a);
    h_u256_set64(r, 1ULL);
    uint64_t exp[4] = {
        0xFFFFFFFEFFFFFC2DULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL
    };
    for (int i = 0; i < 256; i++) {
        uint64_t bit = (exp[i>>6] >> (i&63)) & 1ULL;
        if (bit) { h_fe_mul(&tmp, r, &base); h_u256_copy(r, &tmp); }
        if (i < 255) { h_fe_sqr(&tmp, &base); h_u256_copy(&base, &tmp); }
    }
}

#endif // GPU_UINT256_CUH
