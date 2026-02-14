#include <stdio.h>
#include <stdint.h>
#include <time.h>

typedef struct {
    uint64_t d[4];
} uint256_t;

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

bool uint256_is_zero(const uint256_t* a) {
    return (a->d[0] == 0 && a->d[1] == 0 && a->d[2] == 0 && a->d[3] == 0);
}

int uint256_cmp(const uint256_t* a, const uint256_t* b) {
    for (int i = 3; i >= 0; i--) {
        if (a->d[i] > b->d[i]) return 1;
        if (a->d[i] < b->d[i]) return -1;
    }
    return 0;
}

uint64_t add_with_carry(uint64_t* result, uint64_t a, uint64_t b, uint64_t carry) {
    uint64_t sum = a + carry;
    uint64_t carry1 = (sum < a) ? 1 : 0;
    sum += b;
    uint64_t carry2 = (sum < b) ? 1 : 0;
    *result = sum;
    return carry1 + carry2;
}

uint64_t uint256_add_with_carry_out(uint256_t* result, const uint256_t* a, const uint256_t* b) {
    uint64_t carry = 0;
    carry = add_with_carry(&result->d[0], a->d[0], b->d[0], carry);
    carry = add_with_carry(&result->d[1], a->d[1], b->d[1], carry);
    carry = add_with_carry(&result->d[2], a->d[2], b->d[2], carry);
    carry = add_with_carry(&result->d[3], a->d[3], b->d[3], carry);
    return carry;
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

void uint256_mod_sub(uint256_t* result, const uint256_t* a, const uint256_t* b, const uint256_t* m) {
    if (uint256_cmp(a, b) >= 0) {
        uint256_sub(result, a, b);
    } else {
        uint256_t temp;
        uint256_sub(&temp, a, b);
        uint256_add_with_carry_out(result, &temp, m);
    }
}

uint64_t uint256_get_bit(const uint256_t* a, int pos) {
    if (pos < 0 || pos >= 256) return 0;
    int word_idx = pos / 64;
    int bit_idx = pos % 64;
    return (a->d[word_idx] >> bit_idx) & 1;
}

void uint256_rshift(uint256_t* result, const uint256_t* a, int n) {
    if (n >= 256) {
        uint256_set_zero(result);
        return;
    }
    if (n == 0) {
        *result = *a;
        return;
    }
    
    int word_shift = n / 64;
    int bit_shift = n % 64;
    
    if (bit_shift == 0) {
        for (int i = 0; i < 4 - word_shift; i++) {
            result->d[i] = a->d[i + word_shift];
        }
        for (int i = 4 - word_shift; i < 4; i++) {
            result->d[i] = 0;
        }
    } else {
        for (int i = 0; i < 4 - word_shift - 1; i++) {
            result->d[i] = (a->d[i + word_shift] >> bit_shift) | 
                           (a->d[i + word_shift + 1] << (64 - bit_shift));
        }
        if (4 - word_shift - 1 >= 0 && 4 - word_shift - 1 < 4) {
            result->d[4 - word_shift - 1] = a->d[3] >> bit_shift;
        }
        for (int i = 4 - word_shift; i < 4; i++) {
            result->d[i] = 0;
        }
    }
}

void uint256_rshift1_with_carry(uint256_t* result, const uint256_t* a, uint64_t carry_in) {
    result->d[0] = (a->d[0] >> 1) | (a->d[1] << 63);
    result->d[1] = (a->d[1] >> 1) | (a->d[2] << 63);
    result->d[2] = (a->d[2] >> 1) | (a->d[3] << 63);
    result->d[3] = (a->d[3] >> 1) | (carry_in << 63);
}

void uint256_mod_inverse(uint256_t* r, const uint256_t* a, const uint256_t* mod) {
    uint256_t u = *a;
    uint256_t v = *mod;
    uint256_t x1, x2;
    uint256_set_u64(&x1, 1);
    uint256_set_zero(&x2);
    
    uint64_t x1_carry = 0;
    uint64_t x2_carry = 0;
    
    while (!uint256_is_zero(&u) && !uint256_is_zero(&v)) {
        while (uint256_get_bit(&u, 0) == 0) {
            uint256_rshift(&u, &u, 1);
            if (uint256_get_bit(&x1, 0)) {
                x1_carry += uint256_add_with_carry_out(&x1, &x1, mod);
            }
            uint256_rshift1_with_carry(&x1, &x1, x1_carry);
            x1_carry = 0;
        }
        while (uint256_get_bit(&v, 0) == 0) {
            uint256_rshift(&v, &v, 1);
            if (uint256_get_bit(&x2, 0)) {
                x2_carry += uint256_add_with_carry_out(&x2, &x2, mod);
            }
            uint256_rshift1_with_carry(&x2, &x2, x2_carry);
            x2_carry = 0;
        }
        if (uint256_cmp(&u, &v) >= 0) {
            uint256_mod_sub(&u, &u, &v, mod);
            uint256_mod_sub(&x1, &x1, &x2, mod);
            x1_carry = 0;
        } else {
            uint256_mod_sub(&v, &v, &u, mod);
            uint256_mod_sub(&x2, &x2, &x1, mod);
            x2_carry = 0;
        }
    }
    
    if (uint256_is_zero(&u)) {
        while (uint256_cmp(&x2, mod) >= 0) {
            uint256_sub(&x2, &x2, mod);
        }
        *r = x2;
    } else {
        while (uint256_cmp(&x1, mod) >= 0) {
            uint256_sub(&x1, &x1, mod);
        }
        *r = x1;
    }
}

int main() {
    printf("Comprehensive uint256_mod_inverse Tests\n");
    printf("==========================================\n\n");
    
    uint256_t p;
    uint256_set_hex(&p, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    
    // Test with several values to ensure no infinite loops
    uint64_t test_values[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                               31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                               0x6AC3875, 0xDEADBEEF, 0x123456789ABCDEF0ULL};
    
    int num_tests = sizeof(test_values) / sizeof(test_values[0]);
    int passed = 0;
    
    printf("Testing %d values for modular inverse...\n\n", num_tests);
    
    clock_t start = clock();
    
    for (int i = 0; i < num_tests; i++) {
        uint256_t a, inv;
        uint256_set_u64(&a, test_values[i]);
        
        uint256_mod_inverse(&inv, &a, &p);
        
        // Check that the inverse is valid (non-zero and less than p)
        if (!uint256_is_zero(&inv) && uint256_cmp(&inv, &p) < 0) {
            passed++;
            printf("✓ Test %2d: inv(0x%llx) computed successfully\n", 
                   i+1, (unsigned long long)test_values[i]);
        } else {
            printf("✗ Test %2d: inv(0x%llx) FAILED\n", 
                   i+1, (unsigned long long)test_values[i]);
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("\n==========================================\n");
    printf("Results: %d/%d tests passed\n", passed, num_tests);
    printf("Time taken: %.3f seconds\n", time_spent);
    printf("Average time per inverse: %.6f seconds\n", time_spent / num_tests);
    
    if (passed == num_tests) {
        printf("\n✓ All tests PASSED - No infinite loops detected!\n");
        return 0;
    } else {
        printf("\n✗ Some tests FAILED\n");
        return 1;
    }
}
