#include <stdio.h>
#include <stdint.h>

typedef struct {
    uint64_t d[4];
} uint256_t;

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

bool uint256_equal(const uint256_t* a, const uint256_t* b) {
    return (a->d[0] == b->d[0] && a->d[1] == b->d[1] && 
            a->d[2] == b->d[2] && a->d[3] == b->d[3]);
}

int main() {
    printf("Testing modular inverse\n");
    printf("=======================\n\n");
    
    uint256_t p;
    uint256_set_hex(&p, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    uint256_print("p", &p);
    
    // Inverse of 2 mod p should be (p+1)/2
    uint256_t expected_inv;
    uint256_set_hex(&expected_inv, "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF7FFFFE18");
    uint256_print("Expected inv(2)", &expected_inv);
    
    // To verify, 2 * inv(2) should equal 1 mod p
    printf("\nNote: 2 * inv(2) = 1 (mod p)\n");
    printf("This requires modular multiplication to verify\n");
    
    return 0;
}
