#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>

void print_hex(const char* label, const uint8_t* data, int len) {
    printf("%s: ", label);
    for (int i = 0; i < len; i++) {
        printf("%02x", data[i]);
    }
    printf("\n");
}

int main() {
    // Test public key: 031A864BAE3922F351F1B57CFDD827C25B7E093CB9C88A72C1CD893D9F90F44ECE
    uint8_t pubkey[33] = {
        0x03, 0x1A, 0x86, 0x4B, 0xAE, 0x39, 0x22, 0xF3, 0x51, 0xF1, 0xB5, 0x7C, 0xFD, 0xD8, 0x27, 0xC2,
        0x5B, 0x7E, 0x09, 0x3C, 0xB9, 0xC8, 0x8A, 0x72, 0xC1, 0xCD, 0x89, 0x3D, 0x9F, 0x90, 0xF4, 0x4E,
        0xCE
    };
    
    printf("Testing hash160 (SHA256 + RIPEMD160)\n");
    printf("=====================================\n\n");
    
    // Step 1: SHA256
    uint8_t sha_hash[32];
    SHA256(pubkey, 33, sha_hash);
    print_hex("SHA256    ", sha_hash, 32);
    
    // Step 2: RIPEMD160
    uint8_t ripemd_hash[20];
    RIPEMD160(sha_hash, 32, ripemd_hash);
    print_hex("RIPEMD160 ", ripemd_hash, 20);
    
    // Expected result
    uint8_t expected[20] = {
        0x0c, 0x7a, 0xaf, 0x6c, 0xaa, 0x7e, 0x54, 0x24, 0xb6, 0x3d, 
        0x31, 0x7f, 0x0f, 0x8f, 0x1f, 0x9f, 0xa4, 0x0d, 0x55, 0x60
    };
    print_hex("Expected  ", expected, 20);
    
    bool match = memcmp(ripemd_hash, expected, 20) == 0;
    printf("\nResult: %s\n", match ? "PASS" : "FAIL");
    
    return match ? 0 : 1;
}
