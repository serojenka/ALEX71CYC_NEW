/**
 * Test program for Montgomery multiplication on CUDA
 * Compares Montgomery vs fast secp256k1 implementations
 */

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "cuda_uint256.cuh"
#include "cuda_secp256k1.cuh"

// Helper to print uint256
void print_uint256(const char* label, const uint256_t* a) {
    printf("%s: %016llx%016llx%016llx%016llx\n", label,
           (unsigned long long)a->d[3],
           (unsigned long long)a->d[2],
           (unsigned long long)a->d[1],
           (unsigned long long)a->d[0]);
}

// Test kernel
__global__ void test_multiplication_kernel(uint256_t* results, int* status) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint256_t a, b, result_fast, result_mont;
        
        // Test 1: Simple multiplication 2 * 3
        uint256_set_u64(&a, 2);
        uint256_set_u64(&b, 3);
        
        uint256_mod_mul_secp256k1_fast(&result_fast, &a, &b, &secp256k1_p);
        
        // Also test Montgomery path
        uint256_t a_mont, b_mont, result_mont_form;
        to_montgomery(&a_mont, &a);
        to_montgomery(&b_mont, &b);
        mont_mul(&result_mont_form, &a_mont, &b_mont);
        from_montgomery(&result_mont, &result_mont_form);
        
        results[0] = result_fast;
        results[1] = result_mont;
        
        // Test 2: Squaring
        uint256_set_u64(&a, 12345);
        uint256_mod_sqr_secp256k1_fast(&result_fast, &a, &secp256k1_p);
        
        to_montgomery(&a_mont, &a);
        mont_sqr(&result_mont_form, &a_mont);
        from_montgomery(&result_mont, &result_mont_form);
        
        results[2] = result_fast;
        results[3] = result_mont;
        
        // Test 3: Large number multiplication
        a.d[0] = 0x59F2815B16F81798ULL;
        a.d[1] = 0x029BFCDB2DCE28D9ULL;
        a.d[2] = 0x55A06295CE870B07ULL;
        a.d[3] = 0x79BE667EF9DCBBACULL;
        
        uint256_set_u64(&b, 2);
        
        uint256_mod_mul_secp256k1_fast(&result_fast, &a, &b, &secp256k1_p);
        
        to_montgomery(&a_mont, &a);
        to_montgomery(&b_mont, &b);
        mont_mul(&result_mont_form, &a_mont, &b_mont);
        from_montgomery(&result_mont, &result_mont_form);
        
        results[4] = result_fast;
        results[5] = result_mont;
        
        *status = 1; // Success
    }
}

// Test EC operations
__global__ void test_ec_operations_kernel(point_t* results, int* status) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Test point doubling
        point_t g, result;
        g.x = secp256k1_gx;
        g.y = secp256k1_gy;
        uint256_set_u64(&g.z, 1);
        
        point_double(&result, &g);
        results[0] = result;
        
        // Test point addition (G + G should equal 2G)
        point_add(&result, &g, &g);
        results[1] = result;
        
        *status = 1; // Success
    }
}

int main() {
    printf("Testing Montgomery Multiplication vs Fast secp256k1\n");
    printf("===================================================\n\n");
    
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("Error: No CUDA devices found!\n");
        return 1;
    }
    
    printf("Found %d CUDA device(s)\n", device_count);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n\n", prop.major, prop.minor);
    
    // Allocate device memory
    uint256_t* d_results;
    int* d_status;
    point_t* d_ec_results;
    
    cudaMalloc(&d_results, 6 * sizeof(uint256_t));
    cudaMalloc(&d_status, sizeof(int));
    cudaMalloc(&d_ec_results, 2 * sizeof(point_t));
    
    int h_status = 0;
    cudaMemcpy(d_status, &h_status, sizeof(int), cudaMemcpyHostToDevice);
    
    // Run multiplication tests
    printf("Running multiplication tests...\n");
    test_multiplication_kernel<<<1, 1>>>(d_results, d_status);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Get results
    uint256_t h_results[6];
    cudaMemcpy(h_results, d_results, 6 * sizeof(uint256_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_status, d_status, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (h_status != 1) {
        printf("Kernel reported failure!\n");
        return 1;
    }
    
    printf("\nTest 1: 2 * 3 mod p\n");
    print_uint256("  Fast result ", &h_results[0]);
    print_uint256("  Mont result ", &h_results[1]);
    printf("  Match: %s\n\n", 
           (h_results[0].d[0] == h_results[1].d[0] &&
            h_results[0].d[1] == h_results[1].d[1] &&
            h_results[0].d[2] == h_results[1].d[2] &&
            h_results[0].d[3] == h_results[1].d[3]) ? "YES" : "NO");
    
    printf("Test 2: 12345^2 mod p\n");
    print_uint256("  Fast result ", &h_results[2]);
    print_uint256("  Mont result ", &h_results[3]);
    printf("  Match: %s\n\n",
           (h_results[2].d[0] == h_results[3].d[0] &&
            h_results[2].d[1] == h_results[3].d[1] &&
            h_results[2].d[2] == h_results[3].d[2] &&
            h_results[2].d[3] == h_results[3].d[3]) ? "YES" : "NO");
    
    printf("Test 3: G_x * 2 mod p\n");
    print_uint256("  Fast result ", &h_results[4]);
    print_uint256("  Mont result ", &h_results[5]);
    printf("  Match: %s\n\n",
           (h_results[4].d[0] == h_results[5].d[0] &&
            h_results[4].d[1] == h_results[5].d[1] &&
            h_results[4].d[2] == h_results[5].d[2] &&
            h_results[4].d[3] == h_results[5].d[3]) ? "YES" : "NO");
    
    // Run EC operation tests
    printf("Running elliptic curve operation tests...\n");
    h_status = 0;
    cudaMemcpy(d_status, &h_status, sizeof(int), cudaMemcpyHostToDevice);
    
    test_ec_operations_kernel<<<1, 1>>>(d_ec_results, d_status);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("EC kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("EC kernel execution error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    point_t h_ec_results[2];
    cudaMemcpy(h_ec_results, d_ec_results, 2 * sizeof(point_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_status, d_status, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (h_status != 1) {
        printf("EC kernel reported failure!\n");
        return 1;
    }
    
    printf("\nTest 4: Point doubling (2*G)\n");
    print_uint256("  X coord", &h_ec_results[0].x);
    print_uint256("  Y coord", &h_ec_results[0].y);
    print_uint256("  Z coord", &h_ec_results[0].z);
    
    printf("\nTest 5: Point addition (G+G, should match 2*G)\n");
    print_uint256("  X coord", &h_ec_results[1].x);
    print_uint256("  Y coord", &h_ec_results[1].y);
    print_uint256("  Z coord", &h_ec_results[1].z);
    
    // Cleanup
    cudaFree(d_results);
    cudaFree(d_status);
    cudaFree(d_ec_results);
    
    printf("\nAll tests completed successfully!\n");
    return 0;
}
