/**
 * Test program specifically for uint256_mod_sub fix
 * Tests both cases: a >= b and a < b
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Simplified uint256 structure for testing
typedef struct {
    uint64_t d[4];
} uint256_t;

void uint256_set_zero(uint256_t* a) {
    a->d[0] = a->d[1] = a->d[2] = a->d[3] = 0;
}

void uint256_set_u64(uint256_t* a, uint64_t val) {
    a->d[0] = val;
    a->d[1] = a->d[2] = a->d[3] = 0;
}

void uint256_print(const char* label, const uint256_t* a) {
    printf("%s: %016llx%016llx%016llx%016llx\n", label, 
           (unsigned long long)a->d[3], (unsigned long long)a->d[2], 
           (unsigned long long)a->d[1], (unsigned long long)a->d[0]);
}

int uint256_cmp(const uint256_t* a, const uint256_t* b) {
    for (int i = 3; i >= 0; i--) {
        if (a->d[i] > b->d[i]) return 1;
        if (a->d[i] < b->d[i]) return -1;
    }
    return 0;
}

bool uint256_equal(const uint256_t* a, const uint256_t* b) {
    return uint256_cmp(a, b) == 0;
}

uint64_t add_with_carry(uint64_t* result, uint64_t a, uint64_t b, uint64_t carry) {
    __uint128_t sum = (__uint128_t)a + b + carry;
    *result = (uint64_t)sum;
    return (uint64_t)(sum >> 64);
}

void uint256_add(uint256_t* result, const uint256_t* a, const uint256_t* b) {
    uint64_t carry = 0;
    carry = add_with_carry(&result->d[0], a->d[0], b->d[0], carry);
    carry = add_with_carry(&result->d[1], a->d[1], b->d[1], carry);
    carry = add_with_carry(&result->d[2], a->d[2], b->d[2], carry);
    add_with_carry(&result->d[3], a->d[3], b->d[3], carry);
}

uint64_t sub_with_borrow(uint64_t* result, uint64_t a, uint64_t b, uint64_t borrow) {
    __uint128_t diff = (__uint128_t)a - b - borrow;
    *result = (uint64_t)diff;
    return (diff >> 64) & 1;
}

void uint256_sub(uint256_t* result, const uint256_t* a, const uint256_t* b) {
    uint64_t borrow = 0;
    borrow = sub_with_borrow(&result->d[0], a->d[0], b->d[0], borrow);
    borrow = sub_with_borrow(&result->d[1], a->d[1], b->d[1], borrow);
    borrow = sub_with_borrow(&result->d[2], a->d[2], b->d[2], borrow);
    sub_with_borrow(&result->d[3], a->d[3], b->d[3], borrow);
}

// Fixed implementation
void uint256_mod_sub(uint256_t* result, const uint256_t* a, const uint256_t* b, const uint256_t* m) {
    if (uint256_cmp(a, b) >= 0) {
        uint256_sub(result, a, b);
    } else {
        uint256_t temp;
        uint256_sub(&temp, a, b);
        uint256_add(result, &temp, m);
    }
}

int main() {
    printf("Testing uint256_mod_sub fix\n");
    printf("============================\n\n");
    
    // Test 1: Simple case where a >= b
    printf("Test 1: a >= b (10 - 3) mod 13 = 7\n");
    uint256_t a, b, m, result;
    uint256_set_u64(&a, 10);
    uint256_set_u64(&b, 3);
    uint256_set_u64(&m, 13);
    uint256_mod_sub(&result, &a, &b, &m);
    uint256_print("  Result", &result);
    if (result.d[0] == 7 && result.d[1] == 0 && result.d[2] == 0 && result.d[3] == 0) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL: Expected 7\n\n");
    }
    
    // Test 2: Case where a < b (modular wraparound)
    printf("Test 2: a < b (3 - 10) mod 13 = 6\n");
    printf("  Expected: 3 - 10 ≡ -7 ≡ 6 (mod 13)\n");
    uint256_set_u64(&a, 3);
    uint256_set_u64(&b, 10);
    uint256_set_u64(&m, 13);
    uint256_mod_sub(&result, &a, &b, &m);
    uint256_print("  Result", &result);
    if (result.d[0] == 6 && result.d[1] == 0 && result.d[2] == 0 && result.d[3] == 0) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL: Expected 6\n\n");
    }
    
    // Test 3: Larger numbers with secp256k1 prime
    printf("Test 3: Large numbers with secp256k1 prime\n");
    // secp256k1 prime: 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    m.d[0] = 0xFFFFFFFEFFFFFC2FULL;
    m.d[1] = 0xFFFFFFFFFFFFFFFFULL;
    m.d[2] = 0xFFFFFFFFFFFFFFFFULL;
    m.d[3] = 0xFFFFFFFFFFFFFFFFULL;
    
    // Let a = 5, b = 10
    uint256_set_u64(&a, 5);
    uint256_set_u64(&b, 10);
    uint256_mod_sub(&result, &a, &b, &m);
    
    // Expected: 5 - 10 = -5 ≡ p - 5 (mod p)
    uint256_t expected;
    expected.d[0] = 0xFFFFFFFEFFFFFC2AULL;
    expected.d[1] = 0xFFFFFFFFFFFFFFFFULL;
    expected.d[2] = 0xFFFFFFFFFFFFFFFFULL;
    expected.d[3] = 0xFFFFFFFFFFFFFFFFULL;
    
    uint256_print("  Result  ", &result);
    uint256_print("  Expected", &expected);
    if (uint256_equal(&result, &expected)) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
    }
    
    // Test 4: Edge case - subtract from 0
    printf("Test 4: Edge case - (0 - 5) mod 13 = 8\n");
    uint256_set_u64(&a, 0);
    uint256_set_u64(&b, 5);
    uint256_set_u64(&m, 13);
    uint256_mod_sub(&result, &a, &b, &m);
    uint256_print("  Result", &result);
    if (result.d[0] == 8 && result.d[1] == 0 && result.d[2] == 0 && result.d[3] == 0) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL: Expected 8\n\n");
    }
    
    printf("All tests completed!\n");
    return 0;
}
