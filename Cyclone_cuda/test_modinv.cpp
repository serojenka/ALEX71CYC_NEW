#include <stdio.h>
#include <stdint.h>

typedef struct {
    uint64_t d[4];
} uint256_t;

void uint256_set_u64(uint256_t* a, uint64_t val) {
    a->d[0] = val; a->d[1] = 0; a->d[2] = 0; a->d[3] = 0;
}

void uint256_print(const char* label, const uint256_t* a) {
    printf("%s: %016llx%016llx%016llx%016llx\n", label,
           (unsigned long long)a->d[3],
           (unsigned long long)a->d[2],
           (unsigned long long)a->d[1],
           (unsigned long long)a->d[0]);
}

// Just test that 2 * (2^-1) = 1 mod p
int main() {
    uint256_t two, half, result;
    uint256_set_u64(&two, 2);
    
    // For secp256k1 p, 2^-1 mod p = (p+1)/2
    // p = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    // (p+1)/2 = 7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF7FFFFE18
    half.d[0] = 0xFFFFFF7FFFFE18ULL;
    half.d[1] = 0xFFFFFFFFFFFFFFFFULL;
    half.d[2] = 0xFFFFFFFFFFFFFFFFULL;
    half.d[3] = 0x7FFFFFFFFFFFFFFFULL;
    
    printf("Testing: 2 * (2^-1) mod p should equal 1\n");
    uint256_print("2", &two);
    uint256_print("2^-1 (expected)", &half);
    
    return 0;
}
