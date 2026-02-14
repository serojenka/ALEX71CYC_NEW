#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include "cuda_uint256.cuh"

// Host helper function to set uint256 from uint64
void uint256_set_u64_host(uint256_t* a, uint64_t val) {
    a->d[0] = val;
    a->d[1] = 0;
    a->d[2] = 0;
    a->d[3] = 0;
}

// Host helper function to set uint256 from hex string
void uint256_set_hex_host(uint256_t* a, const char* hex) {
    sscanf(hex, "%016llx%016llx%016llx%016llx",
           (unsigned long long*)&a->d[3],
           (unsigned long long*)&a->d[2],
           (unsigned long long*)&a->d[1],
           (unsigned long long*)&a->d[0]);
}

// Host helper function to print uint256
void uint256_print_host(const char* label, const uint256_t* a) {
    printf("%s: 0x%016llx%016llx%016llx%016llx\n", label,
           (unsigned long long)a->d[3],
           (unsigned long long)a->d[2],
           (unsigned long long)a->d[1],
           (unsigned long long)a->d[0]);
}

// Kernel to test modular inverse
__global__ void test_mod_inverse_kernel(uint256_t* input, uint256_t* mod, uint256_t* result) {
    uint256_mod_inverse(result, input, mod);
}

// Kernel to test modular multiplication for verification
__global__ void test_mod_mul_kernel(uint256_t* a, uint256_t* b, uint256_t* mod, uint256_t* result) {
    uint256_mod_mul_secp256k1_fast(result, a, b, mod);
}

int main() {
    printf("Testing uint256_mod_inverse with extended Euclidean algorithm\n");
    printf("=============================================================\n\n");
    
    // SECP256K1 prime p
    uint256_t p_host;
    uint256_set_hex_host(&p_host, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    uint256_print_host("secp256k1 prime p", &p_host);
    printf("\n");
    
    // Test case 1: Inverse of 2 mod p
    printf("Test 1: Inverse of 2 mod p\n");
    printf("---------------------------\n");
    uint256_t two_host;
    uint256_set_u64_host(&two_host, 2);
    uint256_print_host("Input (2)", &two_host);
    
    // Allocate device memory
    uint256_t *d_input, *d_mod, *d_result;
    cudaMalloc(&d_input, sizeof(uint256_t));
    cudaMalloc(&d_mod, sizeof(uint256_t));
    cudaMalloc(&d_result, sizeof(uint256_t));
    
    // Copy to device
    cudaMemcpy(d_input, &two_host, sizeof(uint256_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mod, &p_host, sizeof(uint256_t), cudaMemcpyHostToDevice);
    
    // Run kernel
    test_mod_inverse_kernel<<<1, 1>>>(d_input, d_mod, d_result);
    cudaDeviceSynchronize();
    
    // Copy result back
    uint256_t result_host;
    cudaMemcpy(&result_host, d_result, sizeof(uint256_t), cudaMemcpyDeviceToHost);
    uint256_print_host("Result inv(2)", &result_host);
    
    // Expected inverse of 2 mod p is (p+1)/2
    uint256_t expected_inv2;
    uint256_set_hex_host(&expected_inv2, "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF7FFFFE18");
    uint256_print_host("Expected inv(2)", &expected_inv2);
    
    // Verify: 2 * inv(2) should equal 1 mod p
    uint256_t *d_verify;
    cudaMalloc(&d_verify, sizeof(uint256_t));
    test_mod_mul_kernel<<<1, 1>>>(d_input, d_result, d_mod, d_verify);
    cudaDeviceSynchronize();
    
    uint256_t verify_host;
    cudaMemcpy(&verify_host, d_verify, sizeof(uint256_t), cudaMemcpyDeviceToHost);
    uint256_print_host("Verification: 2 * inv(2) mod p", &verify_host);
    
    bool test1_pass = (verify_host.d[0] == 1 && verify_host.d[1] == 0 && 
                       verify_host.d[2] == 0 && verify_host.d[3] == 0);
    printf("Test 1: %s\n\n", test1_pass ? "PASS" : "FAIL");
    
    // Test case 2: Inverse of 0x6AC3875 (the key mentioned in the problem)
    printf("Test 2: Inverse of 0x6AC3875 mod p\n");
    printf("------------------------------------\n");
    uint256_t key_host;
    uint256_set_u64_host(&key_host, 0x6AC3875);
    uint256_print_host("Input (0x6AC3875)", &key_host);
    
    cudaMemcpy(d_input, &key_host, sizeof(uint256_t), cudaMemcpyHostToDevice);
    test_mod_inverse_kernel<<<1, 1>>>(d_input, d_mod, d_result);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&result_host, d_result, sizeof(uint256_t), cudaMemcpyDeviceToHost);
    uint256_print_host("Result inv(0x6AC3875)", &result_host);
    
    // Verify: 0x6AC3875 * inv(0x6AC3875) should equal 1 mod p
    test_mod_mul_kernel<<<1, 1>>>(d_input, d_result, d_mod, d_verify);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&verify_host, d_verify, sizeof(uint256_t), cudaMemcpyDeviceToHost);
    uint256_print_host("Verification: 0x6AC3875 * inv(0x6AC3875) mod p", &verify_host);
    
    bool test2_pass = (verify_host.d[0] == 1 && verify_host.d[1] == 0 && 
                       verify_host.d[2] == 0 && verify_host.d[3] == 0);
    printf("Test 2: %s\n\n", test2_pass ? "PASS" : "FAIL");
    
    // Test case 3: Inverse of 3 mod p
    printf("Test 3: Inverse of 3 mod p\n");
    printf("---------------------------\n");
    uint256_t three_host;
    uint256_set_u64_host(&three_host, 3);
    uint256_print_host("Input (3)", &three_host);
    
    cudaMemcpy(d_input, &three_host, sizeof(uint256_t), cudaMemcpyHostToDevice);
    test_mod_inverse_kernel<<<1, 1>>>(d_input, d_mod, d_result);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&result_host, d_result, sizeof(uint256_t), cudaMemcpyDeviceToHost);
    uint256_print_host("Result inv(3)", &result_host);
    
    // Verify: 3 * inv(3) should equal 1 mod p
    test_mod_mul_kernel<<<1, 1>>>(d_input, d_result, d_mod, d_verify);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&verify_host, d_verify, sizeof(uint256_t), cudaMemcpyDeviceToHost);
    uint256_print_host("Verification: 3 * inv(3) mod p", &verify_host);
    
    bool test3_pass = (verify_host.d[0] == 1 && verify_host.d[1] == 0 && 
                       verify_host.d[2] == 0 && verify_host.d[3] == 0);
    printf("Test 3: %s\n\n", test3_pass ? "PASS" : "FAIL");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_mod);
    cudaFree(d_result);
    cudaFree(d_verify);
    
    // Summary
    printf("=====================\n");
    printf("Summary:\n");
    printf("  Test 1 (inv(2)): %s\n", test1_pass ? "PASS" : "FAIL");
    printf("  Test 2 (inv(0x6AC3875)): %s\n", test2_pass ? "PASS" : "FAIL");
    printf("  Test 3 (inv(3)): %s\n", test3_pass ? "PASS" : "FAIL");
    printf("  Overall: %s\n", (test1_pass && test2_pass && test3_pass) ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    
    return (test1_pass && test2_pass && test3_pass) ? 0 : 1;
}
