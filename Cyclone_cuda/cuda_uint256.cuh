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

// ============================================================================
// Montgomery Multiplication for SECP256K1
// ============================================================================
// Montgomery representation: a is represented as aR mod N, where R = 2^256
// This allows fast modular multiplication without division
//
// For secp256k1, we combine Montgomery multiplication with the special form
// p = 2^256 - 0x1000003D1 for maximum performance

// Montgomery constant: R = 2^256 mod p (precomputed)
__constant__ uint256_t MONT_R = {{0x1000003D1ULL, 0x0000000000000001ULL, 0, 0}};

// R^2 mod p (precomputed for conversion to Montgomery form)
// R^2 = (2^256)^2 mod p = 2^512 mod p
__constant__ uint256_t MONT_R2 = {{0x000E90A1ULL, 0x000007A2ULL, 0x0000000000000001ULL, 0}};

// Inverse of p modulo 2^64 (precomputed, used in Montgomery reduction)
// p_inv = -p^(-1) mod 2^64
__constant__ uint64_t MONT_P_INV = 0xD838091DD2253531ULL;

// Montgomery reduction: converts T*R^(-1) mod p
// Input: T (512 bits in t[0..7]), Output: T*R^(-1) mod p (256 bits)
// Uses CIOS (Coarsely Integrated Operand Scanning) algorithm
__device__ void mont_reduce(uint256_t* result, const uint64_t* t) {
    uint64_t c[8];
    
    // Copy input to working array
    for (int i = 0; i < 8; i++) {
        c[i] = t[i];
    }
    
    // secp256k1 prime p in 64-bit words (little-endian)
    const uint64_t p[4] = {
        0xFFFFFFFEFFFFFC2FULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL
    };
    
    // CIOS Montgomery reduction
    for (int i = 0; i < 4; i++) {
        // m = c[i] * p_inv mod 2^64
        uint64_t m = c[i] * MONT_P_INV;
        
        // Add m * p to c, starting at position i
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            // Multiply m * p[j]
            uint64_t lo = m * p[j];
            uint64_t hi = __umul64hi(m, p[j]);
            
            // Add to c[i+j]
            uint64_t sum = c[i+j] + lo + carry;
            carry = hi;
            
            // Handle carries
            if (sum < c[i+j]) carry++;
            if (sum < lo) carry++;
            
            c[i+j] = sum;
        }
        
        // Propagate final carry
        int idx = i + 4;
        while (carry && idx < 8) {
            uint64_t sum = c[idx] + carry;
            carry = (sum < c[idx]) ? 1 : 0;
            c[idx] = sum;
            idx++;
        }
    }
    
    // Result is in c[4..7]
    result->d[0] = c[4];
    result->d[1] = c[5];
    result->d[2] = c[6];
    result->d[3] = c[7];
    
    // Final conditional subtraction if result >= p
    uint256_t p_local = {{p[0], p[1], p[2], p[3]}};
    if (uint256_cmp(result, &p_local) >= 0) {
        uint256_sub(result, result, &p_local);
    }
}

// Montgomery multiplication: (a * b * R^(-1)) mod p
// Both a and b should be in Montgomery form
__device__ void mont_mul(uint256_t* result, const uint256_t* a, const uint256_t* b) {
    uint64_t t[8];
    
    // Initialize to zero
    for (int i = 0; i < 8; i++) {
        t[i] = 0;
    }
    
    // Standard 256x256 multiplication to get 512-bit result
    // This is the same as in the fast implementation
    uint64_t temp[5];
    
    mul256x64(t, a, b->d[0]);
    
    mul256x64(temp, a, b->d[1]);
    uint64_t carry = 0;
    carry = add_with_carry(&t[1], t[1], temp[0], carry);
    carry = add_with_carry(&t[2], t[2], temp[1], carry);
    carry = add_with_carry(&t[3], t[3], temp[2], carry);
    carry = add_with_carry(&t[4], t[4], temp[3], carry);
    carry = add_with_carry(&t[5], t[5], temp[4], carry);
    
    mul256x64(temp, a, b->d[2]);
    carry = 0;
    carry = add_with_carry(&t[2], t[2], temp[0], carry);
    carry = add_with_carry(&t[3], t[3], temp[1], carry);
    carry = add_with_carry(&t[4], t[4], temp[2], carry);
    carry = add_with_carry(&t[5], t[5], temp[3], carry);
    carry = add_with_carry(&t[6], t[6], temp[4], carry);
    
    mul256x64(temp, a, b->d[3]);
    carry = 0;
    carry = add_with_carry(&t[3], t[3], temp[0], carry);
    carry = add_with_carry(&t[4], t[4], temp[1], carry);
    carry = add_with_carry(&t[5], t[5], temp[2], carry);
    carry = add_with_carry(&t[6], t[6], temp[3], carry);
    carry = add_with_carry(&t[7], t[7], temp[4], carry);
    
    // Montgomery reduction
    mont_reduce(result, t);
}

// Convert to Montgomery form: a -> aR mod p
__device__ void to_montgomery(uint256_t* result, const uint256_t* a) {
    // To convert to Montgomery form, multiply by R^2 and reduce
    // aR mod p = a * R^2 * R^(-1) mod p
    mont_mul(result, a, &MONT_R2);
}

// Convert from Montgomery form: aR -> a
__device__ void from_montgomery(uint256_t* result, const uint256_t* a) {
    // To convert from Montgomery form, multiply by 1
    // a * 1 * R^(-1) mod p = a * R^(-1) mod p = a (since input is aR)
    uint256_t one = {{1, 0, 0, 0}};
    mont_mul(result, a, &one);
}

// Montgomery squaring: (a * a * R^(-1)) mod p
__device__ void mont_sqr(uint256_t* result, const uint256_t* a) {
    mont_mul(result, a, a);
}

// Modular inverse using Fermat's little theorem with Montgomery arithmetic
// For prime p: a^(-1) = a^(p-2) mod p
__device__ void uint256_mod_inv_mont(uint256_t* result, const uint256_t* a, const uint256_t* mod) {
    // Convert exponent p-2
    uint256_t exp, two;
    uint256_set(&exp, mod);
    uint256_set_u64(&two, 2);
    uint256_sub(&exp, &exp, &two);
    
    // Convert a to Montgomery form
    uint256_t a_mont, base, temp;
    to_montgomery(&a_mont, a);
    uint256_set(&base, &a_mont);
    
    // Start with 1 in Montgomery form (which is R mod p)
    uint256_set(result, &MONT_R);
    
    // Binary exponentiation in Montgomery space
    for (int i = 0; i < 256; i++) {
        int word_idx = i / 64;
        int bit_idx = i % 64;
        
        if (word_idx < 4) {
            uint64_t bit = (exp.d[word_idx] >> bit_idx) & 1;
            if (bit) {
                mont_mul(&temp, result, &base);
                uint256_set(result, &temp);
            }
        }
        
        if (i < 255) {
            mont_sqr(&temp, &base);
            uint256_set(&base, &temp);
        }
    }
    
    // Convert result back from Montgomery form
    from_montgomery(&temp, result);
    uint256_set(result, &temp);
}

#endif // CUDA_UINT256_CUH
