#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "cuda_hash.cuh"

void print_hex(const char* label, const uint8_t* data, int len) {
    printf("%s: ", label);
    for (int i = 0; i < len; i++) {
        printf("%02x", data[i]);
    }
    printf("\n");
}

__global__ void test_sha256_kernel(uint8_t* result) {
    uint8_t pubkey[33] = {
        0x03, 0x1A, 0x86, 0x4B, 0xAE, 0x39, 0x22, 0xF3, 0x51, 0xF1, 0xB5, 0x7C, 0xFD, 0xD8, 0x27, 0xC2,
        0x5B, 0x7E, 0x09, 0x3C, 0xB9, 0xC8, 0x8A, 0x72, 0xC1, 0xCD, 0x89, 0x3D, 0x9F, 0x90, 0xF4, 0x4E,
        0xCE
    };
    
    uint8_t hash[32];
    sha256(pubkey, 33, hash);
    
    for (int i = 0; i < 32; i++) {
        result[i] = hash[i];
    }
}

int main() {
    printf("Testing CUDA SHA256 implementation for 33-byte input\n");
    printf("====================================================\n\n");
    
    uint8_t expected[32] = {
        0x8a, 0x89, 0x04, 0xbe, 0x5c, 0xb8, 0xe8, 0xd9, 0x90, 0x7d, 0xe7, 0xab, 0xd3, 0x37, 0x81, 0xc4,
        0x27, 0x81, 0xe4, 0x34, 0x08, 0x05, 0x7d, 0xbe, 0x62, 0xbc, 0x5a, 0xfa, 0xce, 0x9a, 0x58, 0x75
    };
    
    print_hex("Expected SHA256", expected, 32);
    
    uint8_t* d_result;
    cudaMalloc(&d_result, 32);
    
    test_sha256_kernel<<<1, 1>>>(d_result);
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    uint8_t h_result[32];
    cudaMemcpy(h_result, d_result, 32, cudaMemcpyDeviceToHost);
    
    print_hex("CUDA SHA256    ", h_result, 32);
    
    bool match = (memcmp(h_result, expected, 32) == 0);
    printf("\n");
    if (match) {
        printf("✓ TEST PASSED: SHA256 matches expected value\n");
    } else {
        printf("✗ TEST FAILED: SHA256 does not match expected value\n");
        printf("\nDifferences:\n");
        for (int i = 0; i < 32; i++) {
            if (h_result[i] != expected[i]) {
                printf("  Byte %2d: got 0x%02x, expected 0x%02x\n", i, h_result[i], expected[i]);
            }
        }
    }
    
    cudaFree(d_result);
    return match ? 0 : 1;
}
