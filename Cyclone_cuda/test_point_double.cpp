/**
 * Test program to verify point doubling and public key computation
 * Tests with known private key 0x6AC3875
 * Expected public key: 031A864BAE3922F351F1B57CFDD827C25B7E093CB9C88A72C1CD893D9F90F44ECE
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
        uint256_t temp;
        uint256_sub(&temp, a, b);
        uint256_add(result, &temp, m);
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

// Convert Jacobian to affine coordinates
void point_to_affine(point_t* p, const uint256_t* secp256k1_p) {
    if (uint256_is_zero(&p->z)) {
        return;
    }
    
    uint256_t z_inv, z_inv2;
    uint256_mod_inv(&z_inv, &p->z, secp256k1_p);
    uint256_mod_sqr(&z_inv2, &z_inv, secp256k1_p);
    
    uint256_mod_mul(&p->x, &p->x, &z_inv2, secp256k1_p);
    uint256_mod_mul(&z_inv2, &z_inv2, &z_inv, secp256k1_p);
    uint256_mod_mul(&p->y, &p->y, &z_inv2, secp256k1_p);
    uint256_set_u64(&p->z, 1);
}

// CURRENT (possibly buggy) point doubling implementation
void point_double_current(point_t* result, const point_t* p, const uint256_t* secp256k1_p) {
    if (uint256_is_zero(&p->y)) {
        uint256_set_zero(&result->x);
        uint256_set_zero(&result->y);
        uint256_set_zero(&result->z);
        return;
    }
    
    uint256_t s, m, t, y2;
    
    // s = 4*x*y^2
    uint256_mod_sqr(&y2, &p->y, secp256k1_p);
    uint256_mod_mul(&s, &p->x, &y2, secp256k1_p);
    uint256_mod_add(&s, &s, &s, secp256k1_p);
    uint256_mod_add(&s, &s, &s, secp256k1_p);
    
    // m = 3*x^2 (since a=0 for secp256k1)
    uint256_mod_sqr(&m, &p->x, secp256k1_p);
    uint256_t m3;
    uint256_mod_add(&m3, &m, &m, secp256k1_p);
    uint256_mod_add(&m, &m3, &m, secp256k1_p);
    
    // x' = m^2 - 2*s
    uint256_mod_sqr(&result->x, &m, secp256k1_p);
    uint256_t s2;
    uint256_mod_add(&s2, &s, &s, secp256k1_p);
    uint256_mod_sub(&result->x, &result->x, &s2, secp256k1_p);
    
    // y' = m*(s - x') - 8*y^4
    uint256_t y4;
    uint256_mod_sqr(&y4, &y2, secp256k1_p);
    uint256_mod_add(&t, &y4, &y4, secp256k1_p);
    uint256_mod_add(&t, &t, &t, secp256k1_p);
    uint256_mod_add(&t, &t, &t, secp256k1_p);
    
    uint256_mod_sub(&y2, &s, &result->x, secp256k1_p);
    uint256_mod_mul(&result->y, &m, &y2, secp256k1_p);
    uint256_mod_sub(&result->y, &result->y, &t, secp256k1_p);
    
    // z' = 2*y*z
    uint256_mod_mul(&result->z, &p->y, &p->z, secp256k1_p);
    uint256_mod_add(&result->z, &result->z, &result->z, secp256k1_p);
}

// NEW (correct formula from problem statement) point doubling
void point_double_problem_formula(point_t* result, const point_t* p, const uint256_t* secp256k1_p) {
    if (uint256_is_zero(&p->y)) {
        uint256_set_zero(&result->x);
        uint256_set_zero(&result->y);
        uint256_set_zero(&result->z);
        return;
    }
    
    uint256_t s, m, t, y2;
    
    // s = 4*x*y^2
    uint256_mod_sqr(&y2, &p->y, secp256k1_p);
    uint256_mod_mul(&s, &p->x, &y2, secp256k1_p);
    uint256_mod_add(&s, &s, &s, secp256k1_p);
    uint256_mod_add(&s, &s, &s, secp256k1_p);
    
    // m = 3*x^2 (since a=0 for secp256k1)
    uint256_mod_sqr(&m, &p->x, secp256k1_p);
    uint256_t m3;
    uint256_mod_add(&m3, &m, &m, secp256k1_p);
    uint256_mod_add(&m, &m3, &m, secp256k1_p);
    
    // x' = m^2 - 2*s
    uint256_mod_sqr(&result->x, &m, secp256k1_p);
    uint256_t s2;
    uint256_mod_add(&s2, &s, &s, secp256k1_p);
    uint256_mod_sub(&result->x, &result->x, &s2, secp256k1_p);
    
    // y' = m*(s - x') - 8*y^4*x  ← NOTE THE *x AT THE END!
    uint256_t y4;
    uint256_mod_sqr(&y4, &y2, secp256k1_p);
    uint256_mod_add(&t, &y4, &y4, secp256k1_p);
    uint256_mod_add(&t, &t, &t, secp256k1_p);
    uint256_mod_add(&t, &t, &t, secp256k1_p);
    // Multiply by x
    uint256_mod_mul(&t, &t, &p->x, secp256k1_p);
    
    uint256_mod_sub(&y2, &s, &result->x, secp256k1_p);
    uint256_mod_mul(&result->y, &m, &y2, secp256k1_p);
    uint256_mod_sub(&result->y, &result->y, &t, secp256k1_p);
    
    // z' = 2*y*z
    uint256_mod_mul(&result->z, &p->y, &p->z, secp256k1_p);
    uint256_mod_add(&result->z, &result->z, &result->z, secp256k1_p);
}
void point_double_fixed(point_t* result, const point_t* p, const uint256_t* secp256k1_p) {
    if (uint256_is_zero(&p->y)) {
        uint256_set_zero(&result->x);
        uint256_set_zero(&result->y);
        uint256_set_zero(&result->z);
        return;
    }
    
    /*
      Jacobian doubling formula from AVX2 implementation:
      W = a * Z^2 + 3 * X^2  (a=0 for secp256k1, so W = 3*X^2)
      S = Y * Z
      B = X * Y * S
      H = W^2 - 8*B
      X' = 2*H*S
      Y' = W*(4*B - H) - 8*Y^2*S^2
      Z' = 8*S^3
    */
    
    uint256_t z2, x2, _3x2, w, s, s2, b, _8b, _8y2s2, y2, h;
    
    // Line 269: z2 = Z^2
    uint256_mod_sqr(&z2, &p->z, secp256k1_p);
    // Line 270: z2 = 0 (since a=0)
    uint256_set_zero(&z2);
    
    // Line 271: x2 = X^2
    uint256_mod_sqr(&x2, &p->x, secp256k1_p);
    
    // Lines 272-273: _3x2 = 3*X^2
    uint256_mod_add(&_3x2, &x2, &x2, secp256k1_p);  // _3x2 = 2*x2
    uint256_mod_add(&_3x2, &_3x2, &x2, secp256k1_p);  // _3x2 = 3*x2
    
    // Line 274: w = z2 + _3x2
    uint256_mod_add(&w, &z2, &_3x2, secp256k1_p);
    
    // Line 275: s = Y * Z
    uint256_mod_mul(&s, &p->y, &p->z, secp256k1_p);
    
    // Lines 276-277: b = X * Y * S
    uint256_mod_mul(&b, &p->y, &s, secp256k1_p);  // b = Y * S
    uint256_mod_mul(&b, &b, &p->x, secp256k1_p);   // b = b * X
    
    // Line 278: h = W^2
    uint256_mod_sqr(&h, &w, secp256k1_p);
    
    // Lines 279-282: _8b = 8*B, then h = h - _8b
    uint256_mod_add(&_8b, &b, &b, secp256k1_p);  // _8b = 2*b
    uint256_mod_add(&_8b, &_8b, &_8b, secp256k1_p);  // _8b = 4*b
    uint256_mod_add(&_8b, &_8b, &_8b, secp256k1_p);  // _8b = 8*b
    uint256_mod_sub(&h, &h, &_8b, secp256k1_p);  // h = h - _8b
    
    // Lines 284-285: r.x = 2*H*S
    uint256_mod_mul(&result->x, &h, &s, secp256k1_p);  // result->x = H * S
    uint256_mod_add(&result->x, &result->x, &result->x, secp256k1_p);  // result->x = 2 * result->x
    
    // Line 287: s2 = S^2
    uint256_mod_sqr(&s2, &s, secp256k1_p);
    
    // Line 288: y2 = Y^2
    uint256_mod_sqr(&y2, &p->y, secp256k1_p);
    
    // Lines 289-292: _8y2s2 = 8*Y^2*S^2
    uint256_mod_mul(&_8y2s2, &y2, &s2, secp256k1_p);  // _8y2s2 = Y^2 * S^2
    uint256_mod_add(&_8y2s2, &_8y2s2, &_8y2s2, secp256k1_p);  // _8y2s2 = 2*_8y2s2
    uint256_mod_add(&_8y2s2, &_8y2s2, &_8y2s2, secp256k1_p);  // _8y2s2 = 4*_8y2s2
    uint256_mod_add(&_8y2s2, &_8y2s2, &_8y2s2, secp256k1_p);  // _8y2s2 = 8*_8y2s2
    
    // Lines 294-298: r.y = W*(4*B - H) - 8*Y^2*S^2
    uint256_t _4b;
    uint256_mod_add(&_4b, &b, &b, secp256k1_p);  // _4b = 2*b
    uint256_mod_add(&_4b, &_4b, &_4b, secp256k1_p);  // _4b = 4*b
    uint256_mod_sub(&result->y, &_4b, &h, secp256k1_p);  // result->y = 4*b - h
    uint256_mod_mul(&result->y, &result->y, &w, secp256k1_p);  // result->y = result->y * w
    uint256_mod_sub(&result->y, &result->y, &_8y2s2, secp256k1_p);  // result->y = result->y - _8y2s2
    
    // Lines 300-303: r.z = 8*S^3
    uint256_mod_mul(&result->z, &s2, &s, secp256k1_p);  // result->z = S^2 * S = S^3
    uint256_mod_add(&result->z, &result->z, &result->z, secp256k1_p);  // result->z = 2*result->z
    uint256_mod_add(&result->z, &result->z, &result->z, secp256k1_p);  // result->z = 4*result->z
    uint256_mod_add(&result->z, &result->z, &result->z, secp256k1_p);  // result->z = 8*result->z
}

// Scalar multiplication using double-and-add
void point_mul(point_t* result, const uint256_t* scalar, const uint256_t* secp256k1_p, const uint256_t* secp256k1_gx, const uint256_t* secp256k1_gy) {
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
                // TODO: Need point_add - for now just mark this
                printf("  Bit %d is set\n", i);
            }
        }
        
        if (i < 255) {
            point_double_current(&temp, &temp, secp256k1_p);
        }
    }
    
    // For this test, we'll use a simplified approach
    *result = temp;
}

int main() {
    printf("Testing Point Doubling with Known Private Key\n");
    printf("==============================================\n\n");
    
    uint256_t secp256k1_p;
    uint256_set_hex(&secp256k1_p, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    
    uint256_t secp256k1_gx, secp256k1_gy;
    uint256_set_hex(&secp256k1_gx, "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    uint256_set_hex(&secp256k1_gy, "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
    
    // Test private key
    uint256_t privkey;
    uint256_set_hex(&privkey, "00000000000000000000000000000000000000000000000000000000006AC3875");
    
    printf("Private key: ");
    uint256_print("", &privkey);
    printf("\n");
    
    // Expected public key (compressed)
    uint256_t expected_pubkey_x;
    uint256_set_hex(&expected_pubkey_x, "1A864BAE3922F351F1B57CFDD827C25B7E093CB9C88A72C1CD893D9F90F44ECE");
    
    printf("Expected public key X: ");
    uint256_print("", &expected_pubkey_x);
    printf("\n");
    
    // Test point doubling with generator
    point_t G, G2, G4;
    uint256_set(&G.x, &secp256k1_gx);
    uint256_set(&G.y, &secp256k1_gy);
    uint256_set_u64(&G.z, 1);
    
    printf("Generator G:\n");
    uint256_print("  x", &G.x);
    uint256_print("  y", &G.y);
    uint256_print("  z", &G.z);
    printf("\n");
    
    // Compute 2*G
    point_double_current(&G2, &G, &secp256k1_p);
    printf("2*G (using current formula):\n");
    uint256_print("  x", &G2.x);
    uint256_print("  y", &G2.y);
    uint256_print("  z", &G2.z);
    
    // Expected 2*G
    uint256_t expected_2G_x, expected_2G_y;
    uint256_set_hex(&expected_2G_x, "C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5");
    uint256_set_hex(&expected_2G_y, "1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A");
    printf("\nExpected 2*G:\n");
    uint256_print("  x", &expected_2G_x);
    uint256_print("  y", &expected_2G_y);
    
    // Convert to affine and compare (z should be 2 for first doubling since z'=2*y*z and input z=1)
    // For comparison, let's check if z is correct
    uint256_t expected_z;
    uint256_set_u64(&expected_z, 2);
    uint256_mod_mul(&expected_z, &expected_z, &secp256k1_gy, &secp256k1_p); // z' = 2*y (since input z=1)
    
    printf("\nExpected z for 2*G: ");
    uint256_print("", &expected_z);
    
    uint256_t one;
    uint256_set_u64(&one, 1);
    if (uint256_equal(&G2.z, &one)) {
        printf("\nz=1, comparing directly\n");
        if (uint256_equal(&G2.x, &expected_2G_x) && uint256_equal(&G2.y, &expected_2G_y)) {
            printf("✓ PASS: 2*G matches expected value\n");
        } else {
            printf("✗ FAIL: 2*G does not match expected value\n");
        }
    } else {
        printf("\nz≠1, converting to affine for comparison\n");
        point_t G2_affine = G2;
        point_to_affine(&G2_affine, &secp256k1_p);
        
        printf("2*G (affine):\n");
        uint256_print("  x", &G2_affine.x);
        uint256_print("  y", &G2_affine.y);
        uint256_print("  z", &G2_affine.z);
        printf("\n");
        
        if (uint256_equal(&G2_affine.x, &expected_2G_x) && uint256_equal(&G2_affine.y, &expected_2G_y)) {
            printf("✓ PASS (CURRENT): 2*G matches expected value after affine conversion\n");
        } else {
            printf("✗ FAIL (CURRENT): 2*G does not match expected value\n");
        }
    }
    
    printf("\n===========================================\n");
    printf("Testing PROBLEM STATEMENT formula (with *x term)\n");
    printf("===========================================\n\n");
    
    // Test with problem statement formula
    point_t G2_problem;
    point_double_problem_formula(&G2_problem, &G, &secp256k1_p);
    printf("2*G (using PROBLEM STATEMENT formula):\n");
    uint256_print("  x", &G2_problem.x);
    uint256_print("  y", &G2_problem.y);
    uint256_print("  z", &G2_problem.z);
    
    if (uint256_equal(&G2_problem.z, &one)) {
        printf("\nz=1, comparing directly\n");
        if (uint256_equal(&G2_problem.x, &expected_2G_x) && uint256_equal(&G2_problem.y, &expected_2G_y)) {
            printf("✓ PASS (PROBLEM): 2*G matches expected value\n");
        } else {
            printf("✗ FAIL (PROBLEM): 2*G does not match expected value\n");
        }
    } else {
        printf("\nz≠1, converting to affine for comparison\n");
        point_t G2_problem_affine = G2_problem;
        point_to_affine(&G2_problem_affine, &secp256k1_p);
        
        printf("2*G (PROBLEM, affine):\n");
        uint256_print("  x", &G2_problem_affine.x);
        uint256_print("  y", &G2_problem_affine.y);
        uint256_print("  z", &G2_problem_affine.z);
        printf("\n");
        
        if (uint256_equal(&G2_problem_affine.x, &expected_2G_x) && uint256_equal(&G2_problem_affine.y, &expected_2G_y)) {
            printf("✓✓✓ PASS (PROBLEM): 2*G matches expected value after affine conversion ✓✓✓\n");
        } else {
            printf("✗ FAIL (PROBLEM): 2*G does not match expected value\n");
        }
    }
    
    printf("\n===========================================\n");
    printf("Testing FIXED formula (AVX2)\n");
    printf("===========================================\n\n");
    
    // Test with fixed formula
    point_t G2_fixed;
    point_double_fixed(&G2_fixed, &G, &secp256k1_p);
    printf("2*G (using FIXED formula):\n");
    uint256_print("  x", &G2_fixed.x);
    uint256_print("  y", &G2_fixed.y);
    uint256_print("  z", &G2_fixed.z);
    
    if (uint256_equal(&G2_fixed.z, &one)) {
        printf("\nz=1, comparing directly\n");
        if (uint256_equal(&G2_fixed.x, &expected_2G_x) && uint256_equal(&G2_fixed.y, &expected_2G_y)) {
            printf("✓ PASS (FIXED): 2*G matches expected value\n");
        } else {
            printf("✗ FAIL (FIXED): 2*G does not match expected value\n");
        }
    } else {
        printf("\nz≠1, converting to affine for comparison\n");
        point_t G2_fixed_affine = G2_fixed;
        point_to_affine(&G2_fixed_affine, &secp256k1_p);
        
        printf("2*G (FIXED, affine):\n");
        uint256_print("  x", &G2_fixed_affine.x);
        uint256_print("  y", &G2_fixed_affine.y);
        uint256_print("  z", &G2_fixed_affine.z);
        printf("\n");
        
        if (uint256_equal(&G2_fixed_affine.x, &expected_2G_x) && uint256_equal(&G2_fixed_affine.y, &expected_2G_y)) {
            printf("✓✓✓ PASS (FIXED): 2*G matches expected value after affine conversion ✓✓✓\n");
        } else {
            printf("✗ FAIL (FIXED): 2*G does not match expected value\n");
        }
    }
    
    printf("\nTest completed. Full scalar multiplication requires point addition.\n");
    
    return 0;
}
