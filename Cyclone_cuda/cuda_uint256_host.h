#ifndef CUDA_UINT256_HOST_H
#define CUDA_UINT256_HOST_H

#include <stdint.h>
#include "cuda_uint256.cuh"

// Host versions of uint256 functions that can be called from CPU code
// These are equivalent to the __device__ versions in cuda_uint256.cuh

// Add with carry (host version)
inline uint64_t add_with_carry_host(uint64_t* result, uint64_t a, uint64_t b, uint64_t carry) {
    uint64_t sum = a + carry;
    uint64_t carry1 = (sum < a) ? 1 : 0;
    sum += b;
    uint64_t carry2 = (sum < b) ? 1 : 0;
    *result = sum;
    return carry1 + carry2;
}

// 256-bit addition (host version)
inline void uint256_add_host(uint256_t* result, const uint256_t* a, const uint256_t* b) {
    uint64_t carry = 0;
    carry = add_with_carry_host(&result->d[0], a->d[0], b->d[0], carry);
    carry = add_with_carry_host(&result->d[1], a->d[1], b->d[1], carry);
    carry = add_with_carry_host(&result->d[2], a->d[2], b->d[2], carry);
    add_with_carry_host(&result->d[3], a->d[3], b->d[3], carry);
}

// Subtract with borrow (host version)
inline uint64_t sub_with_borrow_host(uint64_t* result, uint64_t a, uint64_t b, uint64_t borrow) {
    uint64_t diff = a - borrow;
    uint64_t borrow1 = (diff > a) ? 1 : 0;
    uint64_t temp = diff;
    diff -= b;
    uint64_t borrow2 = (diff > temp) ? 1 : 0;
    *result = diff;
    return borrow1 + borrow2;
}

// 256-bit subtraction (host version)
inline void uint256_sub_host(uint256_t* result, const uint256_t* a, const uint256_t* b) {
    uint64_t borrow = 0;
    borrow = sub_with_borrow_host(&result->d[0], a->d[0], b->d[0], borrow);
    borrow = sub_with_borrow_host(&result->d[1], a->d[1], b->d[1], borrow);
    borrow = sub_with_borrow_host(&result->d[2], a->d[2], b->d[2], borrow);
    sub_with_borrow_host(&result->d[3], a->d[3], b->d[3], borrow);
}

// Set uint256 from 64-bit value (host version)
inline void uint256_set_u64_host(uint256_t* a, uint64_t val) {
    a->d[0] = val;
    a->d[1] = 0;
    a->d[2] = 0;
    a->d[3] = 0;
}

#endif // CUDA_UINT256_HOST_H
