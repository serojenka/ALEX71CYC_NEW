/**
 * Test program for fast secp256k1 modular multiplication
 * This runs on CPU to verify the algorithm correctness
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Simplified uint256 structure for testing
typedef struct {
    uint64_t d[4];
} uint256_t;

// Helper functions
void uint256_set_zero(uint256_t* a) {
    a->d[0] = 0; a->d[1] = 0; a->d[2] = 0; a->d[3] = 0;
}

void uint256_set_u64(uint256_t* a, uint64_t val) {
    a->d[0] = val; a->d[1] = 0; a->d[2] = 0; a->d[3] = 0;
}

void uint256_set_hex(uint256_t* a, const char* hex) {
    // Parse hex string (without 0x prefix)
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

// Add with carry
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

// 64-bit multiply returning 128-bit result
void umul128(uint64_t a, uint64_t b, uint64_t* lo, uint64_t* hi) {
    __uint128_t product = (__uint128_t)a * b;
    *lo = (uint64_t)product;
    *hi = (uint64_t)(product >> 64);
}

// Multiply 256-bit by 64-bit
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

// Fast modular multiplication for secp256k1
void uint256_mod_mul_secp256k1_fast(uint256_t* result, const uint256_t* a, const uint256_t* b, const uint256_t* p) {
    const uint64_t SECP256K1_BETA = 0x1000003D1ULL;
    
    uint64_t r512[8];
    uint64_t t[5];
    
    // Initialize
    r512[4] = 0; r512[5] = 0; r512[6] = 0; r512[7] = 0;
    
    // 256x256 multiplication
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
    
    // Reduce from 512 to 320 bits
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
    uint64_t overflow = t[4] + carry;
    uint64_t lo, hi;
    umul128(overflow, SECP256K1_BETA, &lo, &hi);
    
    carry = 0;
    carry = add_with_carry(&result->d[0], r512[0], lo, carry);
    carry = add_with_carry(&result->d[1], r512[1], hi, carry);
    carry = add_with_carry(&result->d[2], r512[2], 0, carry);
    carry = add_with_carry(&result->d[3], r512[3], 0, carry);
    
    // Final reduction if result >= p
    if (uint256_cmp(result, p) >= 0) {
        uint256_sub(result, result, p);
    }
}

int main() {
    printf("Testing fast secp256k1 modular multiplication\n");
    printf("==============================================\n\n");
    
    uint256_t p;
    uint256_set_hex(&p, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    uint256_print("secp256k1 p", &p);
    printf("\n");
    
    // Test 1: 2 * 3 = 6
    {
        printf("Test 1: 2 * 3 mod p\n");
        uint256_t a, b, result, expected;
        uint256_set_u64(&a, 2);
        uint256_set_u64(&b, 3);
        uint256_set_u64(&expected, 6);
        
        uint256_mod_mul_secp256k1_fast(&result, &a, &b, &p);
        
        uint256_print("  a", &a);
        uint256_print("  b", &b);
        uint256_print("  result", &result);
        uint256_print("  expected", &expected);
        printf("  %s\n\n", uint256_equal(&result, &expected) ? "PASS" : "FAIL");
    }
    
    // Test 2: Large number multiplication
    {
        printf("Test 2: Large number multiplication\n");
        uint256_t a, b, result;
        uint256_set_hex(&a, "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
        uint256_set_hex(&b, "0000000000000000000000000000000000000000000000000000000000000002");
        
        uint256_mod_mul_secp256k1_fast(&result, &a, &b, &p);
        
        uint256_print("  a", &a);
        uint256_print("  b", &b);
        uint256_print("  result", &result);
        
        // result should be 2*a mod p
        uint256_t expected;
        uint256_add(&expected, &a, &a);
        if (uint256_cmp(&expected, &p) >= 0) {
            uint256_sub(&expected, &expected, &p);
        }
        uint256_print("  expected (2*a)", &expected);
        printf("  %s\n\n", uint256_equal(&result, &expected) ? "PASS" : "FAIL");
    }
    
    // Test 3: Squaring
    {
        printf("Test 3: Squaring a number\n");
        uint256_t a, result;
        uint256_set_u64(&a, 12345);
        
        uint256_mod_mul_secp256k1_fast(&result, &a, &a, &p);
        
        uint256_print("  a", &a);
        uint256_print("  a^2 mod p", &result);
        
        uint256_t expected;
        uint256_set_u64(&expected, 12345ULL * 12345ULL);
        printf("  %s\n\n", uint256_equal(&result, &expected) ? "PASS" : "FAIL");
    }
    
    // Test 4: Identity test - a * 1 = a
    {
        printf("Test 4: Identity - a * 1 = a\n");
        uint256_t a, one, result;
        uint256_set_hex(&a, "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
        uint256_set_u64(&one, 1);
        
        uint256_mod_mul_secp256k1_fast(&result, &a, &one, &p);
        
        uint256_print("  a", &a);
        uint256_print("  result", &result);
        printf("  %s\n\n", uint256_equal(&result, &a) ? "PASS" : "FAIL");
    }
    
    printf("Tests completed!\n");
    return 0;
}
