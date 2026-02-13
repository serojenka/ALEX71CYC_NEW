#ifndef CUDA_UINT256_CUH
#define CUDA_UINT256_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// 256-bit unsigned integer for GPU (uses 4x 64-bit words in little-endian)
typedef struct {
    uint64_t d[4];
} uint256_t;

// Device functions for 256-bit arithmetic
__device__ __forceinline__ void uint256_set_zero(uint256_t* a) {
    a->d[0] = 0;
    a->d[1] = 0;
    a->d[2] = 0;
    a->d[3] = 0;
}

__device__ __forceinline__ void uint256_set(uint256_t* dst, const uint256_t* src) {
    dst->d[0] = src->d[0];
    dst->d[1] = src->d[1];
    dst->d[2] = src->d[2];
    dst->d[3] = src->d[3];
}

__device__ __forceinline__ void uint256_set_u64(uint256_t* a, uint64_t val) {
    a->d[0] = val;
    a->d[1] = 0;
    a->d[2] = 0;
    a->d[3] = 0;
}

// Add with carry
__device__ __forceinline__ uint64_t add_with_carry(uint64_t* result, uint64_t a, uint64_t b, uint64_t carry) {
    uint64_t sum = a + carry;
    uint64_t carry1 = (sum < a) ? 1 : 0;
    sum += b;
    uint64_t carry2 = (sum < b) ? 1 : 0;
    *result = sum;
    return carry1 + carry2;
}

// 256-bit addition
__device__ __forceinline__ void uint256_add(uint256_t* result, const uint256_t* a, const uint256_t* b) {
    uint64_t carry = 0;
    carry = add_with_carry(&result->d[0], a->d[0], b->d[0], carry);
    carry = add_with_carry(&result->d[1], a->d[1], b->d[1], carry);
    carry = add_with_carry(&result->d[2], a->d[2], b->d[2], carry);
    add_with_carry(&result->d[3], a->d[3], b->d[3], carry);
}

// 256-bit subtraction
__device__ __forceinline__ uint64_t sub_with_borrow(uint64_t* result, uint64_t a, uint64_t b, uint64_t borrow) {
    uint64_t diff = a - borrow;
    uint64_t borrow1 = (diff > a) ? 1 : 0;
    uint64_t temp = diff;
    diff -= b;
    uint64_t borrow2 = (diff > temp) ? 1 : 0;
    *result = diff;
    return borrow1 + borrow2;
}

__device__ __forceinline__ void uint256_sub(uint256_t* result, const uint256_t* a, const uint256_t* b) {
    uint64_t borrow = 0;
    borrow = sub_with_borrow(&result->d[0], a->d[0], b->d[0], borrow);
    borrow = sub_with_borrow(&result->d[1], a->d[1], b->d[1], borrow);
    borrow = sub_with_borrow(&result->d[2], a->d[2], b->d[2], borrow);
    sub_with_borrow(&result->d[3], a->d[3], b->d[3], borrow);
}

// Compare: returns 1 if a > b, 0 if a == b, -1 if a < b
__device__ __forceinline__ int uint256_cmp(const uint256_t* a, const uint256_t* b) {
    for (int i = 3; i >= 0; i--) {
        if (a->d[i] > b->d[i]) return 1;
        if (a->d[i] < b->d[i]) return -1;
    }
    return 0;
}

// Check if zero
__device__ __forceinline__ bool uint256_is_zero(const uint256_t* a) {
    return (a->d[0] == 0 && a->d[1] == 0 && a->d[2] == 0 && a->d[3] == 0);
}

// Modular addition (a + b) mod m
__device__ __forceinline__ void uint256_mod_add(uint256_t* result, const uint256_t* a, const uint256_t* b, const uint256_t* m) {
    uint256_add(result, a, b);
    if (uint256_cmp(result, m) >= 0) {
        uint256_sub(result, result, m);
    }
}

// Modular subtraction (a - b) mod m
__device__ __forceinline__ void uint256_mod_sub(uint256_t* result, const uint256_t* a, const uint256_t* b, const uint256_t* m) {
    if (uint256_cmp(a, b) >= 0) {
        uint256_sub(result, a, b);
    } else {
        uint256_t temp;
        uint256_sub(&temp, m, b);
        uint256_add(result, a, &temp);
        if (uint256_cmp(result, m) >= 0) {
            uint256_sub(result, result, m);
        }
    }
}

// ============================================================================
// Fast Modular Multiplication for SECP256K1
// ============================================================================
// secp256k1 prime: p = 2^256 - 0x1000003D1 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// This special form allows very fast reduction without full Montgomery multiplication

__constant__ uint64_t SECP256K1_BETA = 0x1000003D1ULL;  // p = 2^256 - BETA

// Multiply 256-bit by 64-bit, result stored in 5 words (320 bits)
// result[0..4] = a[0..3] * b
__device__ __forceinline__ void mul256x64(uint64_t* result, const uint256_t* a, uint64_t b) {
    uint64_t carry = 0;
    
    for (int i = 0; i < 4; i++) {
        uint64_t lo = a->d[i] * b;
        uint64_t hi = __umul64hi(a->d[i], b);
        uint64_t sum = lo + carry;
        result[i] = sum;
        carry = hi + ((sum < lo) ? 1 : 0);
    }
    result[4] = carry;
}

// Fast modular multiplication for secp256k1: (a * b) mod p
// Uses the special form p = 2^256 - 0x1000003D1 for fast reduction
__device__ void uint256_mod_mul_secp256k1_fast(uint256_t* result, const uint256_t* a, const uint256_t* b, const uint256_t* p) {
    uint64_t r512[8];
    uint64_t t[5];
    
    // Initialize high words to zero
    r512[4] = 0;
    r512[5] = 0;
    r512[6] = 0;
    r512[7] = 0;
    
    // 256x256 multiplication - compute a * b[0]
    mul256x64(r512, a, b->d[0]);
    
    // a * b[1], add to r512[1..5]
    mul256x64(t, a, b->d[1]);
    uint64_t carry = 0;
    carry = add_with_carry(&r512[1], r512[1], t[0], carry);
    carry = add_with_carry(&r512[2], r512[2], t[1], carry);
    carry = add_with_carry(&r512[3], r512[3], t[2], carry);
    carry = add_with_carry(&r512[4], r512[4], t[3], carry);
    carry = add_with_carry(&r512[5], r512[5], t[4], carry);
    
    // a * b[2], add to r512[2..6]
    mul256x64(t, a, b->d[2]);
    carry = 0;
    carry = add_with_carry(&r512[2], r512[2], t[0], carry);
    carry = add_with_carry(&r512[3], r512[3], t[1], carry);
    carry = add_with_carry(&r512[4], r512[4], t[2], carry);
    carry = add_with_carry(&r512[5], r512[5], t[3], carry);
    carry = add_with_carry(&r512[6], r512[6], t[4], carry);
    
    // a * b[3], add to r512[3..7]
    mul256x64(t, a, b->d[3]);
    carry = 0;
    carry = add_with_carry(&r512[3], r512[3], t[0], carry);
    carry = add_with_carry(&r512[4], r512[4], t[1], carry);
    carry = add_with_carry(&r512[5], r512[5], t[2], carry);
    carry = add_with_carry(&r512[6], r512[6], t[3], carry);
    carry = add_with_carry(&r512[7], r512[7], t[4], carry);
    
    // Now r512[0..7] contains the 512-bit product a * b
    // Reduce from 512 to 320 bits using p = 2^256 - 0x1000003D1
    // High part: r512[4..7] * 0x1000003D1 is added to low part r512[0..3]
    
    uint256_t high_part;
    high_part.d[0] = r512[4];
    high_part.d[1] = r512[5];
    high_part.d[2] = r512[6];
    high_part.d[3] = r512[7];
    
    mul256x64(t, &high_part, SECP256K1_BETA);
    carry = 0;
    carry = add_with_carry(&r512[0], r512[0], t[0], carry);
    carry = add_with_carry(&r512[1], r512[1], t[1], carry);
    carry = add_with_carry(&r512[2], r512[2], t[2], carry);
    carry = add_with_carry(&r512[3], r512[3], t[3], carry);
    
    // Reduce from 320 to 256 bits
    // t[4] + carry is the overflow, multiply by 0x1000003D1 and add
    uint64_t overflow = t[4] + carry;
    uint64_t lo = overflow * SECP256K1_BETA;
    uint64_t hi = __umul64hi(overflow, SECP256K1_BETA);
    
    carry = 0;
    carry = add_with_carry(&result->d[0], r512[0], lo, carry);
    carry = add_with_carry(&result->d[1], r512[1], hi, carry);
    carry = add_with_carry(&result->d[2], r512[2], 0, carry);
    carry = add_with_carry(&result->d[3], r512[3], 0, carry);
    
    // Final reduction if result >= p (very rare)
    if (uint256_cmp(result, p) >= 0) {
        uint256_sub(result, result, p);
    }
}

// Fast modular squaring for secp256k1: (a * a) mod p
__device__ __forceinline__ void uint256_mod_sqr_secp256k1_fast(uint256_t* result, const uint256_t* a, const uint256_t* p) {
    uint256_mod_mul_secp256k1_fast(result, a, a, p);
}

#endif // CUDA_UINT256_CUH
