#include <stdio.h>
#include <stdint.h>

typedef struct {
    uint64_t d[4];
} uint256_t;

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

int main() {
    uint256_t privkey;
    uint256_set_hex(&privkey, "00000000000000000000000000000000000000000000000000000000006AC3875");
    printf("Private key should be 0x6AC3875 = decimal 112138357\n");
    uint256_print("Parsed", &privkey);
    printf("d[0] = %llu (decimal), should be 112138357\n", (unsigned long long)privkey.d[0]);
    return 0;
}
