/**
 * Test program for affine coordinate ECC operations
 * Tests with known private key 0x6AC3875
 * Expected public key: 031A864BAE3922F351F1B57CFDD827C25B7E093CB9C88A72C1CD893D9F90F44ECE
 * Expected address: 128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Simplified uint256 structure for testing
typedef struct {
    uint64_t d[4];
} uint256_t;

typedef struct {
    uint256_t x;
    uint256_t y;
    uint256_t z;
} point_t;

// Basic uint256 operations
void uint256_set_zero(uint256_t* a) {
    a->d[0] = 0; a->d[1] = 0; a->d[2] = 0; a->d[3] = 0;
}

void uint256_set_u64(uint256_t* a, uint64_t val) {
    a->d[0] = val; a->d[1] = 0; a->d[2] = 0; a->d[3] = 0;
}

void uint256_set_hex(uint256_t* a, const char* hex) {
    sscanf(hex, "%016llx%016llx%016llx%016llx",
           (unsigned long long*)&a->d[3],
           (unsigned long long*)&a->d[2],
           (unsigned long long*)&a->d[1],
           (unsigned long long*)&a->d[0]);
}

void uint256_print(const char* label, const uint256_t* a) {
    printf("%s: %016llx%016llx%016llx%016llx\n", label,
           (unsigned long long)a->d[3],
           (unsigned long long)a->d[2],
           (unsigned long long)a->d[1],
           (unsigned long long)a->d[0]);
}

void uint256_set(uint256_t* dest, const uint256_t* src) {
    dest->d[0] = src->d[0];
    dest->d[1] = src->d[1];
    dest->d[2] = src->d[2];
    dest->d[3] = src->d[3];
}

int uint256_cmp(const uint256_t* a, const uint256_t* b) {
    for (int i = 3; i >= 0; i--) {
        if (a->d[i] > b->d[i]) return 1;
        if (a->d[i] < b->d[i]) return -1;
    }
    return 0;
}

bool uint256_equal(const uint256_t* a, const uint256_t* b) {
    return (a->d[0] == b->d[0] && a->d[1] == b->d[1] && 
            a->d[2] == b->d[2] && a->d[3] == b->d[3]);
}

bool uint256_is_zero(const uint256_t* a) {
    return (a->d[0] == 0 && a->d[1] == 0 && a->d[2] == 0 && a->d[3] == 0);
}

uint64_t add_with_carry(uint64_t* result, uint64_t a, uint64_t b, uint64_t carry) {
    uint64_t sum = a + carry;
    uint64_t carry1 = (sum < a) ? 1 : 0;
    sum += b;
    uint64_t carry2 = (sum < b) ? 1 : 0;
    *result = sum;
    return carry1 + carry2;
}

void uint256_add(uint256_t* result, const uint256_t* a, const uint256_t* b) {
    uint64_t carry = 0;
    carry = add_with_carry(&result->d[0], a->d[0], b->d[0], carry);
    carry = add_with_carry(&result->d[1], a->d[1], b->d[1], carry);
    carry = add_with_carry(&result->d[2], a->d[2], b->d[2], carry);
    add_with_carry(&result->d[3], a->d[3], b->d[3], carry);
}

uint64_t sub_with_borrow(uint64_t* result, uint64_t a, uint64_t b, uint64_t borrow) {
    uint64_t diff = a - borrow;
    uint64_t borrow1 = (diff > a) ? 1 : 0;
    uint64_t temp = diff;
    diff -= b;
    uint64_t borrow2 = (diff > temp) ? 1 : 0;
    *result = diff;
    return borrow1 + borrow2;
}

void uint256_sub(uint256_t* result, const uint256_t* a, const uint256_t* b) {
    uint64_t borrow = 0;
    borrow = sub_with_borrow(&result->d[0], a->d[0], b->d[0], borrow);
    borrow = sub_with_borrow(&result->d[1], a->d[1], b->d[1], borrow);
    borrow = sub_with_borrow(&result->d[2], a->d[2], b->d[2], borrow);
    sub_with_borrow(&result->d[3], a->d[3], b->d[3], borrow);
}

void uint256_mod_add(uint256_t* result, const uint256_t* a, const uint256_t* b, const uint256_t* m) {
    uint256_add(result, a, b);
    if (uint256_cmp(result, m) >= 0) {
        uint256_sub(result, result, m);
    }
}

void uint256_mod_sub(uint256_t* result, const uint256_t* a, const uint256_t* b, const uint256_t* m) {
    if (uint256_cmp(a, b) >= 0) {
        uint256_sub(result, a, b);
    } else {
        // When a < b: we need to compute (a - b) mod m = a + m - b
        // To avoid issues, compute: m - b + a = m - (b - a)
        uint256_t b_minus_a, m_minus_diff;
        uint256_sub(&b_minus_a, b, a);          // b - a (this is positive since b > a)
        uint256_sub(&m_minus_diff, m, &b_minus_a);  // m - (b - a)
        uint256_set(result, &m_minus_diff);
    }
}

void umul128(uint64_t a, uint64_t b, uint64_t* lo, uint64_t* hi) {
    __uint128_t product = (__uint128_t)a * b;
    *lo = (uint64_t)product;
    *hi = (uint64_t)(product >> 64);
}

void mul256x64(uint64_t* result, const uint256_t* a, uint64_t b) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t lo, hi;
        umul128(a->d[i], b, &lo, &hi);
        uint64_t sum = lo + carry;
        result[i] = sum;
        carry = hi + ((sum < lo) ? 1 : 0);
    }
    result[4] = carry;
}

void uint256_mod_mul_secp256k1_fast(uint256_t* result, const uint256_t* a, const uint256_t* b, const uint256_t* p) {
    const uint64_t SECP256K1_BETA = 0x1000003D1ULL;
    
    uint64_t r512[8];
    uint64_t t[5];
    
    r512[4] = 0; r512[5] = 0; r512[6] = 0; r512[7] = 0;
    
    mul256x64(r512, a, b->d[0]);
    
    mul256x64(t, a, b->d[1]);
    uint64_t carry = 0;
    carry = add_with_carry(&r512[1], r512[1], t[0], carry);
    carry = add_with_carry(&r512[2], r512[2], t[1], carry);
    carry = add_with_carry(&r512[3], r512[3], t[2], carry);
    carry = add_with_carry(&r512[4], r512[4], t[3], carry);
    carry = add_with_carry(&r512[5], r512[5], t[4], carry);
    
    mul256x64(t, a, b->d[2]);
    carry = 0;
    carry = add_with_carry(&r512[2], r512[2], t[0], carry);
    carry = add_with_carry(&r512[3], r512[3], t[1], carry);
    carry = add_with_carry(&r512[4], r512[4], t[2], carry);
    carry = add_with_carry(&r512[5], r512[5], t[3], carry);
    carry = add_with_carry(&r512[6], r512[6], t[4], carry);
    
    mul256x64(t, a, b->d[3]);
    carry = 0;
    carry = add_with_carry(&r512[3], r512[3], t[0], carry);
    carry = add_with_carry(&r512[4], r512[4], t[1], carry);
    carry = add_with_carry(&r512[5], r512[5], t[2], carry);
    carry = add_with_carry(&r512[6], r512[6], t[3], carry);
    carry = add_with_carry(&r512[7], r512[7], t[4], carry);
    
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
    
    uint64_t overflow = t[4] + carry;
    uint64_t lo, hi;
    umul128(overflow, SECP256K1_BETA, &lo, &hi);
    
    carry = 0;
    carry = add_with_carry(&result->d[0], r512[0], lo, carry);
    carry = add_with_carry(&result->d[1], r512[1], hi, carry);
    carry = add_with_carry(&result->d[2], r512[2], 0, carry);
    carry = add_with_carry(&result->d[3], r512[3], 0, carry);
    
    if (uint256_cmp(result, p) >= 0) {
        uint256_sub(result, result, p);
    }
}

void uint256_mod_sqr(uint256_t* result, const uint256_t* a, const uint256_t* p) {
    uint256_mod_mul_secp256k1_fast(result, a, a, p);
}

void uint256_mod_mul(uint256_t* result, const uint256_t* a, const uint256_t* b, const uint256_t* p) {
    uint256_mod_mul_secp256k1_fast(result, a, b, p);
}

// Modular inverse using Fermat's little theorem: a^(-1) = a^(p-2) mod p
void uint256_mod_inv(uint256_t* result, const uint256_t* a, const uint256_t* mod) {
    // Calculate p-2
    uint256_t exp, two;
    uint256_set(&exp, mod);
    uint256_set_u64(&two, 2);
    uint256_sub(&exp, &exp, &two);
    
    // Binary exponentiation
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
            uint256_mod_sqr(&temp, &base, mod);
            uint256_set(&base, &temp);
        }
    }
}

// Affine point doubling
void point_double_affine(point_t* result, const point_t* p, const uint256_t* secp256k1_p) {
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
    
    uint256_t x2, _3x2, _2y, _2y_inv, s, s2, _2x, x_sub_xr, temp;
    
    // x2 = x^2
    uint256_mod_sqr(&x2, &p->x, secp256k1_p);
    
    // _3x2 = 3 * x^2 (since a = 0 for secp256k1)
    uint256_mod_add(&_3x2, &x2, &x2, secp256k1_p);
    uint256_mod_add(&_3x2, &_3x2, &x2, secp256k1_p);
    
    // _2y = 2 * y
    uint256_mod_add(&_2y, &p->y, &p->y, secp256k1_p);
    
    // _2y_inv = (2 * y)^-1 mod p
    uint256_mod_inv(&_2y_inv, &_2y, secp256k1_p);
    
    // s = 3 * x^2 * (2 * y)^-1 mod p
    uint256_mod_mul(&s, &_3x2, &_2y_inv, secp256k1_p);
    
    // s2 = s^2
    uint256_mod_sqr(&s2, &s, secp256k1_p);
    
    // _2x = 2 * x
    uint256_mod_add(&_2x, &p->x, &p->x, secp256k1_p);
    
    // x' = s^2 - 2 * x % p
    uint256_mod_sub(&result->x, &s2, &_2x, secp256k1_p);
    
    // x_sub_xr = x - x'
    uint256_mod_sub(&x_sub_xr, &p->x, &result->x, secp256k1_p);
    
    // temp = s * (x - x')
    uint256_mod_mul(&temp, &s, &x_sub_xr, secp256k1_p);
    
    // y' = s * (x - x') - y % p
    uint256_mod_sub(&result->y, &temp, &p->y, secp256k1_p);
    
    // z' = 1 (affine coordinates)
    uint256_set_u64(&result->z, 1);
}

// Affine point addition
void point_add_affine(point_t* result, const point_t* p1, const point_t* p2, const uint256_t* secp256k1_p) {
    // Handle special cases - point at infinity
    if (uint256_is_zero(&p1->z)) {
        *result = *p2;
        return;
    }
    if (uint256_is_zero(&p2->z)) {
        *result = *p1;
        return;
    }
    
    uint256_t h, r, h_inv, s, s2, x1_plus_x2, x1_sub_xr, temp;
    
    // h = x2 - x1
    uint256_mod_sub(&h, &p2->x, &p1->x, secp256k1_p);
    
    // r = y2 - y1
    uint256_mod_sub(&r, &p2->y, &p1->y, secp256k1_p);
    
    // Check if points are equal (h == 0)
    if (uint256_is_zero(&h)) {
        if (uint256_is_zero(&r)) {
            // Same point, use point doubling
            point_double_affine(result, p1, secp256k1_p);
        } else {
            // Points are negatives of each other, result is point at infinity
            uint256_set_zero(&result->x);
            uint256_set_zero(&result->y);
            uint256_set_zero(&result->z);
        }
        return;
    }
    
    // h_inv = h^-1 mod p
    uint256_mod_inv(&h_inv, &h, secp256k1_p);
    
    // s = r * h^-1 mod p
    uint256_mod_mul(&s, &r, &h_inv, secp256k1_p);
    
    // s2 = s^2
    uint256_mod_sqr(&s2, &s, secp256k1_p);
    
    // x1_plus_x2 = x1 + x2
    uint256_mod_add(&x1_plus_x2, &p1->x, &p2->x, secp256k1_p);
    
    // x' = s^2 - x1 - x2 % p
    uint256_mod_sub(&result->x, &s2, &x1_plus_x2, secp256k1_p);
    
    // x1_sub_xr = x1 - x'
    uint256_mod_sub(&x1_sub_xr, &p1->x, &result->x, secp256k1_p);
    
    // temp = s * (x1 - x')
    uint256_mod_mul(&temp, &s, &x1_sub_xr, secp256k1_p);
    
    // y' = s * (x1 - x') - y1 % p
    uint256_mod_sub(&result->y, &temp, &p1->y, secp256k1_p);
    
    // z' = 1 (affine coordinates)
    uint256_set_u64(&result->z, 1);
}

// Scalar multiplication using double-and-add
void point_mul_affine(point_t* result, const uint256_t* scalar, const uint256_t* secp256k1_p, const uint256_t* secp256k1_gx, const uint256_t* secp256k1_gy) {
    point_t temp, acc;
    uint256_set_zero(&acc.x);
    uint256_set_zero(&acc.y);
    uint256_set_zero(&acc.z);
    
    // Set temp to generator G
    uint256_set(&temp.x, secp256k1_gx);
    uint256_set(&temp.y, secp256k1_gy);
    uint256_set_u64(&temp.z, 1);
    
    // Double-and-add algorithm
    for (int i = 0; i < 256; i++) {
        int word_idx = i / 64;
        int bit_idx = i % 64;
        
        if (word_idx < 4) {
            uint64_t bit = (scalar->d[word_idx] >> bit_idx) & 1;
            
            if (bit) {
                if (uint256_is_zero(&acc.z)) {
                    acc = temp;
                } else {
                    point_add_affine(&acc, &acc, &temp, secp256k1_p);
                }
            }
        }
        
        if (i < 255) {
            point_double_affine(&temp, &temp, secp256k1_p);
        }
    }
    
    *result = acc;
}

int main() {
    printf("Testing Affine Coordinate ECC Implementation\n");
    printf("=============================================\n\n");
    
    uint256_t secp256k1_p;
    uint256_set_hex(&secp256k1_p, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    
    uint256_t secp256k1_gx, secp256k1_gy;
    uint256_set_hex(&secp256k1_gx, "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    uint256_set_hex(&secp256k1_gy, "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
    
    // Test with known private key 0x6AC3875
    uint256_t privkey;
    uint256_set_hex(&privkey, "0000000000000000000000000000000000000000000000000000000006AC3875");
    
    printf("Private key: 0x6AC3875\n");
    uint256_print("", &privkey);
    printf("\n");
    
    // Expected public key (compressed)
    uint256_t expected_pubkey_x;
    uint256_set_hex(&expected_pubkey_x, "1A864BAE3922F351F1B57CFDD827C25B7E093CB9C88A72C1CD893D9F90F44ECE");
    
    printf("Expected public key X:\n");
    uint256_print("", &expected_pubkey_x);
    printf("\n");
    
    // Compute public key using scalar multiplication
    printf("Computing public key using affine coordinate scalar multiplication...\n");
    point_t pubkey;
    point_mul_affine(&pubkey, &privkey, &secp256k1_p, &secp256k1_gx, &secp256k1_gy);
    
    printf("\nComputed public key:\n");
    uint256_print("  x", &pubkey.x);
    uint256_print("  y", &pubkey.y);
    uint256_print("  z", &pubkey.z);
    printf("\n");
    
    // Verify result
    if (uint256_equal(&pubkey.x, &expected_pubkey_x)) {
        printf("✓✓✓ SUCCESS! ✓✓✓\n");
        printf("Computed public key X matches expected value!\n");
        printf("Affine coordinate implementation is CORRECT.\n");
    } else {
        printf("✗ FAIL: Computed public key X does not match expected value\n");
    }
    
    // Test point doubling with generator
    printf("\n\nTest: Point Doubling (2*G)\n");
    printf("===========================\n");
    point_t G, G2;
    uint256_set(&G.x, &secp256k1_gx);
    uint256_set(&G.y, &secp256k1_gy);
    uint256_set_u64(&G.z, 1);
    
    point_double_affine(&G2, &G, &secp256k1_p);
    
    uint256_t expected_2G_x, expected_2G_y;
    uint256_set_hex(&expected_2G_x, "C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5");
    uint256_set_hex(&expected_2G_y, "1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A");
    
    printf("Computed 2*G:\n");
    uint256_print("  x", &G2.x);
    uint256_print("  y", &G2.y);
    
    printf("\nExpected 2*G:\n");
    uint256_print("  x", &expected_2G_x);
    uint256_print("  y", &expected_2G_y);
    
    if (uint256_equal(&G2.x, &expected_2G_x) && uint256_equal(&G2.y, &expected_2G_y)) {
        printf("\n✓ Point doubling is CORRECT\n");
    } else {
        printf("\n✗ Point doubling FAILED\n");
    }
    
    return 0;
}
