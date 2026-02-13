#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "cuda_hash.cuh"

// Host version for verification
void print_hex(const char* label, const uint8_t* data, int len) {
    printf("%s: ", label);
    for (int i = 0; i < len; i++) {
        printf("%02x", data[i]);
    }
    printf("\n");
}

__global__ void test_hash160_kernel(uint8_t* result) {
    // Test public key: 031A864BAE3922F351F1B57CFDD827C25B7E093CB9C88A72C1CD893D9F90F44ECE
    uint8_t pubkey[33] = {
        0x03, 0x1A, 0x86, 0x4B, 0xAE, 0x39, 0x22, 0xF3, 0x51, 0xF1, 0xB5, 0x7C, 0xFD, 0xD8, 0x27, 0xC2,
        0x5B, 0x7E, 0x09, 0x3C, 0xB9, 0xC8, 0x8A, 0x72, 0xC1, 0xCD, 0x89, 0x3D, 0x9F, 0x90, 0xF4, 0x4E,
        0xCE
    };
    
    uint8_t h160[20];
    hash160(pubkey, 33, h160);
    
    // Copy result
    for (int i = 0; i < 20; i++) {
        result[i] = h160[i];
    }
}

int main() {
    printf("Testing CUDA hash160 implementation\n");
    printf("====================================\n\n");
    
    // Expected values
    uint8_t expected_sha256[32] = {
        0x8a, 0x89, 0x04, 0xbe, 0x5c, 0xb8, 0xe8, 0xd9, 0x90, 0x7d, 0xe7, 0xab, 0xd3, 0x37, 0x81, 0xc4,
        0x27, 0x81, 0xe4, 0x34, 0x08, 0x05, 0x7d, 0xbe, 0x62, 0xbc, 0x5a, 0xfa, 0xce, 0x9a, 0x58, 0x75
    };
    
    uint8_t expected_ripemd160[20] = {
        0x0c, 0x7a, 0xaf, 0x6c, 0xaa, 0x7e, 0x54, 0x24, 0xb6, 0x3d, 0x31, 0x7f, 0x0f, 0x8f, 0x1f, 0x9f,
        0xa4, 0x0d, 0x55, 0x60
    };
    
    printf("Test case: Key 0x6AC3875 -> Address 128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k\n");
    printf("Public key: 031A864BAE3922F351F1B57CFDD827C25B7E093CB9C88A72C1CD893D9F90F44ECE\n\n");
    
    print_hex("Expected SHA256   ", expected_sha256, 32);
    print_hex("Expected RIPEMD160", expected_ripemd160, 20);
    printf("\n");
    
    // Allocate device memory
    uint8_t* d_result;
    cudaMalloc(&d_result, 20);
    
    // Run kernel
    test_hash160_kernel<<<1, 1>>>(d_result);
    cudaDeviceSynchronize();
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Copy result back
    uint8_t h_result[20];
    cudaMemcpy(h_result, d_result, 20, cudaMemcpyDeviceToHost);
    
    // Print result
    print_hex("CUDA Result       ", h_result, 20);
    
    // Compare
    bool match = true;
    for (int i = 0; i < 20; i++) {
        if (h_result[i] != expected_ripemd160[i]) {
            match = false;
            break;
        }
    }
    
    printf("\n");
    if (match) {
        printf("✓ TEST PASSED: Hash160 matches expected value\n");
    } else {
        printf("✗ TEST FAILED: Hash160 does not match expected value\n");
        printf("\nDifferences:\n");
        for (int i = 0; i < 20; i++) {
            if (h_result[i] != expected_ripemd160[i]) {
                printf("  Byte %2d: got 0x%02x, expected 0x%02x\n", i, h_result[i], expected_ripemd160[i]);
            }
        }
    }
    
    cudaFree(d_result);
    return match ? 0 : 1;
}
