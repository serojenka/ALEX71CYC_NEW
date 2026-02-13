#ifndef CUDA_SECP256K1_CUH
#define CUDA_SECP256K1_CUH

#include "cuda_uint256.cuh"

// SECP256K1 curve parameters
// p = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F (field prime)
// n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141 (order)
// G = 0479BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

// Point structure for affine coordinates (X, Y, Z)
// In affine coordinates, Z is always 1 for valid points or 0 for point at infinity
typedef struct {
    uint256_t x;
    uint256_t y;
    uint256_t z;  // Always 1 for valid points, 0 for point at infinity
} point_t;

// SECP256k1 prime p (field modulus)
__constant__ uint256_t secp256k1_p = {{0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL}};

// SECP256k1 order n
__constant__ uint256_t secp256k1_n = {{0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFEULL}};

// Generator point G (in affine coordinates)
__constant__ uint256_t secp256k1_gx = {{0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL}};
__constant__ uint256_t secp256k1_gy = {{0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL}};

// Forward declarations
__device__ void uint256_mod_mul(uint256_t* result, const uint256_t* a, const uint256_t* b, const uint256_t* mod);
__device__ void uint256_mod_sqr(uint256_t* result, const uint256_t* a, const uint256_t* mod);
__device__ void uint256_mod_mul_secp256k1_fast(uint256_t* result, const uint256_t* a, const uint256_t* b, const uint256_t* p);
__device__ void uint256_mod_sqr_secp256k1_fast(uint256_t* result, const uint256_t* a, const uint256_t* p);
__device__ void mont_mul(uint256_t* result, const uint256_t* a, const uint256_t* b);
__device__ void mont_sqr(uint256_t* result, const uint256_t* a);
__device__ void to_montgomery(uint256_t* result, const uint256_t* a);
__device__ void from_montgomery(uint256_t* result, const uint256_t* a);
__device__ void uint256_mod_inv_mont(uint256_t* result, const uint256_t* a, const uint256_t* mod);

// Configuration: Use Montgomery multiplication (1) or fast secp256k1 (0)
// For secp256k1, the fast special-form reduction is typically better
// Montgomery is provided as an alternative for testing and comparison
#ifndef USE_MONTGOMERY
#define USE_MONTGOMERY 0  // Default to fast secp256k1 method
#endif

// Modular inverse using Fermat's little theorem: a^(-1) = a^(p-2) mod p
// Uses fast secp256k1 multiplication or Montgomery based on configuration
__device__ void uint256_mod_inv(uint256_t* result, const uint256_t* a, const uint256_t* mod) {
#if USE_MONTGOMERY
    uint256_mod_inv_mont(result, a, mod);
#else
    // Calculate p-2
    uint256_t exp, two;
    uint256_set(&exp, mod);
    uint256_set_u64(&two, 2);
    uint256_sub(&exp, &exp, &two);
    
    // Binary exponentiation with fast multiplication
    uint256_t base, temp;
    uint256_set(&base, a);
    uint256_set_u64(result, 1);
    
    for (int i = 0; i < 256; i++) {
        int word_idx = i / 64;
        int bit_idx = i % 64;
        
        if (word_idx < 4) {
            uint64_t bit = (exp.d[word_idx] >> bit_idx) & 1;
            if (bit) {
                uint256_mod_mul_secp256k1_fast(&temp, result, &base, mod);
                uint256_set(result, &temp);
            }
        }
        
        if (i < 255) {
            uint256_mod_sqr_secp256k1_fast(&temp, &base, mod);
            uint256_set(&base, &temp);
        }
    }
#endif
}

// Modular multiplication - uses Montgomery or fast secp256k1-optimized version
__device__ void uint256_mod_mul(uint256_t* result, const uint256_t* a, const uint256_t* b, const uint256_t* mod) {
#if USE_MONTGOMERY
    // Note: This implementation converts to/from Montgomery form for each operation
    // which adds overhead. For optimal performance in EC operations, point coordinates
    // should remain in Montgomery form throughout multi-operation sequences.
    // This per-operation conversion is provided for correctness and simplicity.
    // Future optimization: Keep EC point coordinates in Montgomery space.
    uint256_t a_mont, b_mont, result_mont;
    to_montgomery(&a_mont, a);
    to_montgomery(&b_mont, b);
    mont_mul(&result_mont, &a_mont, &b_mont);
    from_montgomery(result, &result_mont);
#else
    uint256_mod_mul_secp256k1_fast(result, a, b, mod);
#endif
}

// Modular squaring
__device__ __forceinline__ void uint256_mod_sqr(uint256_t* result, const uint256_t* a, const uint256_t* mod) {
#if USE_MONTGOMERY
    uint256_t a_mont, result_mont;
    to_montgomery(&a_mont, a);
    mont_sqr(&result_mont, &a_mont);
    from_montgomery(result, &result_mont);
#else
    uint256_mod_mul(result, a, a, mod);
#endif
}

// Point doubling in affine coordinates
// Uses the standard affine doubling formula
__device__ void point_double(point_t* result, const point_t* p) {
    // Handle point at infinity
    if (uint256_is_zero(&p->z)) {
        uint256_set_zero(&result->x);
        uint256_set_zero(&result->y);
        uint256_set_zero(&result->z);
        return;
    }
    
    // Check if y = 0 (point at infinity case)
    if (uint256_is_zero(&p->y)) {
        uint256_set_zero(&result->x);
        uint256_set_zero(&result->y);
        uint256_set_zero(&result->z);
        return;
    }
    
    /*
      Affine doubling formula:
      s = (3 * x^2 + a) * (2 * y)^-1 % p
      x' = s^2 - 2 * x % p
      y' = s * (x - x') - y % p
      z' = 1
      
      For secp256k1, a = 0, so:
      s = (3 * x^2) * (2 * y)^-1 % p
    */
    
    uint256_t x2, _3x2, _2y, _2y_inv, s, s2, _2x, x_sub_xr, temp;
    
    // x2 = x^2
    uint256_mod_sqr(&x2, &p->x, &secp256k1_p);
    
    // _3x2 = 3 * x^2 (since a = 0 for secp256k1)
    uint256_mod_add(&_3x2, &x2, &x2, &secp256k1_p);
    uint256_mod_add(&_3x2, &_3x2, &x2, &secp256k1_p);
    
    // _2y = 2 * y
    uint256_mod_add(&_2y, &p->y, &p->y, &secp256k1_p);
    
    // _2y_inv = (2 * y)^-1 mod p
    uint256_mod_inv(&_2y_inv, &_2y, &secp256k1_p);
    
    // s = 3 * x^2 * (2 * y)^-1 mod p
    uint256_mod_mul(&s, &_3x2, &_2y_inv, &secp256k1_p);
    
    // s2 = s^2
    uint256_mod_sqr(&s2, &s, &secp256k1_p);
    
    // _2x = 2 * x
    uint256_mod_add(&_2x, &p->x, &p->x, &secp256k1_p);
    
    // x' = s^2 - 2 * x % p
    uint256_mod_sub(&result->x, &s2, &_2x, &secp256k1_p);
    
    // x_sub_xr = x - x'
    uint256_mod_sub(&x_sub_xr, &p->x, &result->x, &secp256k1_p);
    
    // temp = s * (x - x')
    uint256_mod_mul(&temp, &s, &x_sub_xr, &secp256k1_p);
    
    // y' = s * (x - x') - y % p
    uint256_mod_sub(&result->y, &temp, &p->y, &secp256k1_p);
    
    // z' = 1 (affine coordinates)
    uint256_set_u64(&result->z, 1);
}

// Point addition in affine coordinates
__device__ void point_add(point_t* result, const point_t* p1, const point_t* p2) {
    // Handle special cases - point at infinity
    if (uint256_is_zero(&p1->z)) {
        *result = *p2;
        return;
    }
    if (uint256_is_zero(&p2->z)) {
        *result = *p1;
        return;
    }
    
    /*
      Affine addition formula:
      h = x2 - x1 % p
      r = y2 - y1 % p
      s = r * h^-1 % p
      x' = s^2 - x1 - x2 % p
      y' = s * (x1 - x') - y1 % p
      z' = 1
    */
    
    uint256_t h, r, h_inv, s, s2, x1_plus_x2, x1_sub_xr, temp;
    
    // h = x2 - x1
    uint256_mod_sub(&h, &p2->x, &p1->x, &secp256k1_p);
    
    // r = y2 - y1
    uint256_mod_sub(&r, &p2->y, &p1->y, &secp256k1_p);
    
    // Check if points are equal (h == 0)
    if (uint256_is_zero(&h)) {
        if (uint256_is_zero(&r)) {
            // Same point, use point doubling
            point_double(result, p1);
        } else {
            // Points are negatives of each other, result is point at infinity
            uint256_set_zero(&result->x);
            uint256_set_zero(&result->y);
            uint256_set_zero(&result->z);
        }
        return;
    }
    
    // h_inv = h^-1 mod p
    uint256_mod_inv(&h_inv, &h, &secp256k1_p);
    
    // s = r * h^-1 mod p
    uint256_mod_mul(&s, &r, &h_inv, &secp256k1_p);
    
    // s2 = s^2
    uint256_mod_sqr(&s2, &s, &secp256k1_p);
    
    // x1_plus_x2 = x1 + x2
    uint256_mod_add(&x1_plus_x2, &p1->x, &p2->x, &secp256k1_p);
    
    // x' = s^2 - x1 - x2 % p
    uint256_mod_sub(&result->x, &s2, &x1_plus_x2, &secp256k1_p);
    
    // x1_sub_xr = x1 - x'
    uint256_mod_sub(&x1_sub_xr, &p1->x, &result->x, &secp256k1_p);
    
    // temp = s * (x1 - x')
    uint256_mod_mul(&temp, &s, &x1_sub_xr, &secp256k1_p);
    
    // y' = s * (x1 - x') - y1 % p
    uint256_mod_sub(&result->y, &temp, &p1->y, &secp256k1_p);
    
    // z' = 1 (affine coordinates)
    uint256_set_u64(&result->z, 1);
}

// No-op function: points are already in affine form (z=1) or at infinity (z=0)
// This function is kept for API compatibility with code that expects point_to_affine
__device__ void point_to_affine(point_t* p) {
    if (uint256_is_zero(&p->z)) {
        // Point at infinity, nothing to do
        return;
    }
    // Point is already in affine form with z=1, nothing to do
}

#endif // CUDA_SECP256K1_CUH
