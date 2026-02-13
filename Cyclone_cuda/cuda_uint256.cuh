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

#endif // CUDA_UINT256_CUH
