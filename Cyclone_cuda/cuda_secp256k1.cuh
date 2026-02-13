#ifndef CUDA_SECP256K1_CUH
#define CUDA_SECP256K1_CUH

#include "cuda_uint256.cuh"

// SECP256K1 curve parameters
// p = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F (field prime)
// n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141 (order)
// G = 0479BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

// Point structure for Jacobian coordinates (X, Y, Z)
typedef struct {
    uint256_t x;
    uint256_t y;
    uint256_t z;
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

// Point doubling in Jacobian coordinates
__device__ void point_double(point_t* result, const point_t* p) {
    if (uint256_is_zero(&p->y)) {
        uint256_set_zero(&result->x);
        uint256_set_zero(&result->y);
        uint256_set_zero(&result->z);
        return;
    }
    
    uint256_t s, m, t, y2;
    
    // s = 4*x*y^2
    uint256_mod_sqr(&y2, &p->y, &secp256k1_p);
    uint256_mod_mul(&s, &p->x, &y2, &secp256k1_p);
    uint256_mod_add(&s, &s, &s, &secp256k1_p);
    uint256_mod_add(&s, &s, &s, &secp256k1_p);
    
    // m = 3*x^2 (since a=0 for secp256k1)
    uint256_mod_sqr(&m, &p->x, &secp256k1_p);
    uint256_t m3;
    uint256_mod_add(&m3, &m, &m, &secp256k1_p);
    uint256_mod_add(&m, &m3, &m, &secp256k1_p);
    
    // x' = m^2 - 2*s
    uint256_mod_sqr(&result->x, &m, &secp256k1_p);
    uint256_t s2;
    uint256_mod_add(&s2, &s, &s, &secp256k1_p);
    uint256_mod_sub(&result->x, &result->x, &s2, &secp256k1_p);
    
    // y' = m*(s - x') - 8*y^4
    uint256_t y4;
    uint256_mod_sqr(&y4, &y2, &secp256k1_p);
    uint256_mod_add(&t, &y4, &y4, &secp256k1_p);
    uint256_mod_add(&t, &t, &t, &secp256k1_p);
    uint256_mod_add(&t, &t, &t, &secp256k1_p);
    
    uint256_mod_sub(&y2, &s, &result->x, &secp256k1_p);
    uint256_mod_mul(&result->y, &m, &y2, &secp256k1_p);
    uint256_mod_sub(&result->y, &result->y, &t, &secp256k1_p);
    
    // z' = 2*y*z
    uint256_mod_mul(&result->z, &p->y, &p->z, &secp256k1_p);
    uint256_mod_add(&result->z, &result->z, &result->z, &secp256k1_p);
}

// Point addition in Jacobian coordinates
__device__ void point_add(point_t* result, const point_t* p1, const point_t* p2) {
    // Handle special cases
    if (uint256_is_zero(&p1->z)) {
        *result = *p2;
        return;
    }
    if (uint256_is_zero(&p2->z)) {
        *result = *p1;
        return;
    }
    
    uint256_t u1, u2, s1, s2, h, r;
    
    // u1 = x1*z2^2, u2 = x2*z1^2
    uint256_t z1_2, z2_2;
    uint256_mod_sqr(&z1_2, &p1->z, &secp256k1_p);
    uint256_mod_sqr(&z2_2, &p2->z, &secp256k1_p);
    uint256_mod_mul(&u1, &p1->x, &z2_2, &secp256k1_p);
    uint256_mod_mul(&u2, &p2->x, &z1_2, &secp256k1_p);
    
    // s1 = y1*z2^3, s2 = y2*z1^3
    uint256_t z1_3, z2_3;
    uint256_mod_mul(&z1_3, &z1_2, &p1->z, &secp256k1_p);
    uint256_mod_mul(&z2_3, &z2_2, &p2->z, &secp256k1_p);
    uint256_mod_mul(&s1, &p1->y, &z2_3, &secp256k1_p);
    uint256_mod_mul(&s2, &p2->y, &z1_3, &secp256k1_p);
    
    // h = u2 - u1
    uint256_mod_sub(&h, &u2, &u1, &secp256k1_p);
    
    // r = s2 - s1
    uint256_mod_sub(&r, &s2, &s1, &secp256k1_p);
    
    // Check if points are equal
    if (uint256_is_zero(&h)) {
        if (uint256_is_zero(&r)) {
            point_double(result, p1);
        } else {
            uint256_set_zero(&result->x);
            uint256_set_zero(&result->y);
            uint256_set_zero(&result->z);
        }
        return;
    }
    
    // x3 = r^2 - h^3 - 2*u1*h^2
    uint256_t h2, h3, u1h2;
    uint256_mod_sqr(&h2, &h, &secp256k1_p);
    uint256_mod_mul(&h3, &h2, &h, &secp256k1_p);
    uint256_mod_mul(&u1h2, &u1, &h2, &secp256k1_p);
    
    uint256_t r2;
    uint256_mod_sqr(&r2, &r, &secp256k1_p);
    uint256_mod_sub(&result->x, &r2, &h3, &secp256k1_p);
    uint256_t u1h2_2;
    uint256_mod_add(&u1h2_2, &u1h2, &u1h2, &secp256k1_p);
    uint256_mod_sub(&result->x, &result->x, &u1h2_2, &secp256k1_p);
    
    // y3 = r*(u1*h^2 - x3) - s1*h^3
    uint256_t temp;
    uint256_mod_sub(&temp, &u1h2, &result->x, &secp256k1_p);
    uint256_mod_mul(&result->y, &r, &temp, &secp256k1_p);
    uint256_mod_mul(&temp, &s1, &h3, &secp256k1_p);
    uint256_mod_sub(&result->y, &result->y, &temp, &secp256k1_p);
    
    // z3 = h*z1*z2
    uint256_mod_mul(&temp, &p1->z, &p2->z, &secp256k1_p);
    uint256_mod_mul(&result->z, &h, &temp, &secp256k1_p);
}

// Convert Jacobian to affine coordinates
__device__ void point_to_affine(point_t* p) {
    if (uint256_is_zero(&p->z)) {
        return;
    }
    
    uint256_t z_inv, z_inv2;
    uint256_mod_inv(&z_inv, &p->z, &secp256k1_p);
    uint256_mod_sqr(&z_inv2, &z_inv, &secp256k1_p);
    
    uint256_mod_mul(&p->x, &p->x, &z_inv2, &secp256k1_p);
    uint256_mod_mul(&z_inv2, &z_inv2, &z_inv, &secp256k1_p);
    uint256_mod_mul(&p->y, &p->y, &z_inv2, &secp256k1_p);
    uint256_set_u64(&p->z, 1);
}

#endif // CUDA_SECP256K1_CUH
