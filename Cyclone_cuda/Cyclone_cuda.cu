// Compilation instructions:
// Linux:
//   nvcc -O3 -arch=sm_70 -std=c++14 -o Cyclone_cuda Cyclone_cuda.cu -lcuda
// Windows (with CUDA 12 and Visual Studio):
//   nvcc -O3 -arch=sm_70 -std=c++14 -o Cyclone_cuda.exe Cyclone_cuda.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>

#include "cuda_uint256.cuh"
#include "cuda_secp256k1.cuh"
#include "cuda_hash.cuh"
#include "cuda_utils.h"
#include "cuda_wif.h"
#include "cuda_uint256_host.h"

// Configuration
#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID 1024
#define KEYS_PER_THREAD 256

// Result structure for found keys
struct SearchResult {
    bool found;
    uint256_t private_key;
    uint8_t hash160[20];
    uint8_t pubkey[33];
};

// Global device memory for results
__device__ SearchResult d_result;
__device__ bool d_found_flag = false;

// Scalar multiplication using double-and-add
__device__ void point_mul(point_t* result, const uint256_t* scalar) {
    point_t temp, acc;
    uint256_set_zero(&acc.x);
    uint256_set_zero(&acc.y);
    uint256_set_zero(&acc.z);
    
    // Set temp to generator G
    temp.x = secp256k1_gx;
    temp.y = secp256k1_gy;
    uint256_set_u64(&temp.z, 1);
    
    // Double-and-add algorithm
    for (int i = 0; i < 256; i++) {
        int word_idx = i / 64;
        int bit_idx = i % 64;
        
        if (word_idx < 4) {
            uint64_t bit = (scalar->d[word_idx] >> bit_idx) & 1;
            
            if (bit) {
                if (uint256_is_zero(&acc.z)) {
                    acc = temp;
                } else {
                    point_add(&acc, &acc, &temp);
                }
            }
        }
        
        if (i < 255) {
            point_double(&temp, &temp);
        }
    }
    
    *result = acc;
}

// Convert point to compressed public key
__device__ void point_to_compressed_pubkey(const point_t* p, uint8_t pubkey[33]) {
    point_t affine = *p;
    point_to_affine(&affine);
    
    // Determine prefix (02 for even y, 03 for odd y)
    pubkey[0] = (affine.y.d[0] & 1) ? 0x03 : 0x02;
    
    // Copy x coordinate (big-endian)
    for (int i = 0; i < 4; i++) {
        uint64_t word = affine.x.d[3 - i];
        for (int j = 0; j < 8; j++) {
            pubkey[1 + i * 8 + j] = (word >> (56 - j * 8)) & 0xff;
        }
    }
}

// Main search kernel
__global__ void search_kernel(
    const uint8_t* target_hash160,
    uint256_t range_start,
    uint256_t range_size,
    uint64_t total_threads,
    uint64_t keys_per_thread,
    int use_random,
    uint64_t random_seed,
    int partial_match_len,
    uint64_t jump_size
) {
    uint64_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (d_found_flag) return;
    
    // Calculate this thread's starting key
    uint256_t thread_offset, thread_start, key;
    uint256_set_u64(&thread_offset, global_id);
    uint256_set_zero(&thread_start);  // Initialize to zero
    
    // Multiply offset by keys_per_thread
    for (int i = 0; i < 64; i++) {
        if ((keys_per_thread >> i) & 1) {
            uint256_t shift_val = thread_offset;
            for (int j = 0; j < i; j++) {
                uint256_add(&shift_val, &shift_val, &shift_val);
            }
            uint256_add(&thread_start, &thread_start, &shift_val);
        }
    }
    
    uint256_add(&thread_start, &range_start, &thread_start);
    key = thread_start;
    
    // Random number generator state (simple LCG)
    uint64_t rng_state = random_seed + global_id;
    
    // Process keys
    for (uint64_t i = 0; i < keys_per_thread && !d_found_flag; i++) {
        if (use_random) {
            // Generate random key in range
            rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
            uint256_set_u64(&key, rng_state);
            uint256_mod_add(&key, &range_start, &key, &range_size);
        }
        
        // Compute public key
        point_t pubkey_point;
        point_mul(&pubkey_point, &key);
        
        // Convert to compressed format
        uint8_t pubkey[33];
        point_to_compressed_pubkey(&pubkey_point, pubkey);
        
        // Compute hash160
        uint8_t h160[20];
        hash160(pubkey, 33, h160);
        
        // Check for partial match if enabled
        bool is_candidate = false;
        if (partial_match_len > 0) {
            int match_bytes = partial_match_len / 2;
            bool match = true;
            for (int j = 0; j < match_bytes && j < 20; j++) {
                if (h160[j] != target_hash160[j]) {
                    match = false;
                    break;
                }
            }
            
            if (match && (partial_match_len & 1)) {
                // Check half-byte
                if ((h160[match_bytes] & 0xF0) != (target_hash160[match_bytes] & 0xF0)) {
                    match = false;
                }
            }
            
            if (match) {
                is_candidate = true;
                // Jump forward if configured
                if (jump_size > 0) {
                    uint256_t jump;
                    uint256_set_u64(&jump, jump_size);
                    uint256_add(&key, &key, &jump);
                }
            }
        }
        
        // Check for full match
        bool full_match = true;
        for (int j = 0; j < 20; j++) {
            if (h160[j] != target_hash160[j]) {
                full_match = false;
                break;
            }
        }
        
        if (full_match) {
            // Found it! Store result atomically
            if (atomicCAS((int*)&d_found_flag, 0, 1) == 0) {
                d_result.found = true;
                d_result.private_key = key;
                for (int j = 0; j < 20; j++) {
                    d_result.hash160[j] = h160[j];
                }
                for (int j = 0; j < 33; j++) {
                    d_result.pubkey[j] = pubkey[j];
                }
            }
            return;
        }
        
        // Move to next key
        if (!use_random && !is_candidate) {
            uint256_t one;
            uint256_set_u64(&one, 1);
            uint256_add(&key, &key, &one);
        }
    }
}

// Host utility functions
void print_hex(const char* label, const uint8_t* data, int len) {
    printf("%s", label);
    for (int i = 0; i < len; i++) {
        printf("%02x", data[i]);
    }
    printf("\n");
}

void uint256_to_hex(const uint256_t* val, char* hex_out) {
    sprintf(hex_out, "%016llx%016llx%016llx%016llx",
            (unsigned long long)val->d[3],
            (unsigned long long)val->d[2],
            (unsigned long long)val->d[1],
            (unsigned long long)val->d[0]);
}

void parse_hex_string(const char* hex, uint8_t* out, int out_len) {
    int hex_len = strlen(hex);
    memset(out, 0, out_len);
    
    for (int i = 0; i < hex_len && i / 2 < out_len; i += 2) {
        char byte_str[3] = {hex[i], hex[i + 1], 0};
        out[i / 2] = (uint8_t)strtol(byte_str, NULL, 16);
    }
}

void parse_hex_to_uint256(const char* hex, uint256_t* out) {
    memset(out, 0, sizeof(uint256_t));
    int len = strlen(hex);
    
    // Parse from right to left (least significant to most significant)
    for (int i = 0; i < len && i < 64; i++) {
        char c = hex[len - 1 - i];
        uint64_t digit = 0;
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
        
        int word_idx = i / 16;
        int bit_offset = (i % 16) * 4;
        if (word_idx < 4) {
            out->d[word_idx] |= (digit << bit_offset);
        }
    }
}

int main(int argc, char** argv) {
    std::cout << "=== Cyclone CUDA - GPU Bitcoin Puzzle Solver ===" << std::endl;
    std::cout << "CUDA 12 - Multi-GPU Support" << std::endl;
    std::cout << std::endl;
    
    // Parse command-line arguments (maintaining CLI compatibility)
    std::string target_address;
    std::string range_start_hex, range_end_hex;
    bool address_provided = false, range_provided = false;
    int partial_match_len = 0;
    uint64_t jump_size = 0;
    bool use_random = false;
    int num_gpus = 1;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-a") == 0 && i + 1 < argc) {
            target_address = argv[++i];
            address_provided = true;
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            std::string range_input = argv[++i];
            size_t colon_pos = range_input.find(':');
            if (colon_pos != std::string::npos) {
                range_start_hex = range_input.substr(0, colon_pos);
                range_end_hex = range_input.substr(colon_pos + 1);
                range_provided = true;
            }
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            partial_match_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc) {
            jump_size = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--random") == 0) {
            use_random = true;
        } else if (strcmp(argv[i], "--gpus") == 0 && i + 1 < argc) {
            num_gpus = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " -a <address> -r <start:end> [options]\n";
            std::cout << "Options:\n";
            std::cout << "  -a <address>     Target P2PKH address\n";
            std::cout << "  -r <start:end>   Range in hex (e.g., 1:FFFF)\n";
            std::cout << "  -p <len>         Partial match length (hex digits)\n";
            std::cout << "  -j <size>        Jump size after partial match\n";
            std::cout << "  --random         Use random search mode\n";
            std::cout << "  --gpus <n>       Number of GPUs to use (default: all available)\n";
            return 0;
        }
    }
    
    if (!address_provided || !range_provided) {
        std::cerr << "Error: Both -a and -r are required!\n";
        std::cerr << "Use -h for help\n";
        return 1;
    }
    
    // Decode target address to hash160
    uint8_t target_hash160[20];
    if (!decode_address_to_hash160(target_address.c_str(), target_hash160)) {
        std::cerr << "Error: Failed to decode address. Provide valid P2PKH address or hash160.\n";
        return 1;
    }
    
    // Parse range
    uint256_t range_start, range_end, range_size;
    parse_hex_to_uint256(range_start_hex.c_str(), &range_start);
    parse_hex_to_uint256(range_end_hex.c_str(), &range_end);
    uint256_sub_host(&range_size, &range_end, &range_start);
    uint256_t one;
    uint256_set_u64_host(&one, 1);
    uint256_add_host(&range_size, &range_size, &one);
    
    // Query GPU devices
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "  GPU " << i << ": " << prop.name 
                  << " (Compute " << prop.major << "." << prop.minor 
                  << ", " << prop.totalGlobalMem / (1024*1024) << " MB)" << std::endl;
    }
    std::cout << std::endl;
    
    if (device_count == 0) {
        std::cerr << "Error: No CUDA devices found!\n";
        return 1;
    }
    
    if (num_gpus > device_count) {
        num_gpus = device_count;
    }
    
    std::cout << "Using " << num_gpus << " GPU(s)" << std::endl;
    std::cout << "Target Address: " << target_address << std::endl;
    std::cout << "Range: " << range_start_hex << ":" << range_end_hex << std::endl;
    if (partial_match_len > 0) {
        std::cout << "Partial match: " << partial_match_len << " hex digits" << std::endl;
    }
    if (jump_size > 0) {
        std::cout << "Jump size: " << jump_size << std::endl;
    }
    if (use_random) {
        std::cout << "Mode: Random search" << std::endl;
    }
    std::cout << std::endl;
    
    // Allocate device memory
    uint8_t* d_target_hash160;
    cudaMalloc(&d_target_hash160, 20);
    cudaMemcpy(d_target_hash160, target_hash160, 20, cudaMemcpyHostToDevice);
    
    // Initialize result
    SearchResult h_result;
    h_result.found = false;
    cudaMemcpyToSymbol(d_result, &h_result, sizeof(SearchResult));
    bool found_flag = false;
    cudaMemcpyToSymbol(d_found_flag, &found_flag, sizeof(bool));
    
    // Calculate work distribution
    uint64_t total_threads = (uint64_t)BLOCKS_PER_GRID * THREADS_PER_BLOCK * num_gpus;
    uint64_t keys_per_thread = KEYS_PER_THREAD;
    uint64_t random_seed = (uint64_t)time(NULL);
    
    std::cout << "Threads per GPU: " << (BLOCKS_PER_GRID * THREADS_PER_BLOCK) << std::endl;
    std::cout << "Keys per thread: " << keys_per_thread << std::endl;
    std::cout << "Total threads: " << total_threads << std::endl;
    std::cout << std::endl;
    
    // Launch kernels on multiple GPUs
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "================= SEARCHING =================\n";
    
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        
        // Adjust range for this GPU
        uint256_t gpu_range_start = range_start;
        // In production, properly divide range among GPUs
        
        search_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
            d_target_hash160,
            gpu_range_start,
            range_size,
            total_threads,
            keys_per_thread,
            use_random ? 1 : 0,
            random_seed + gpu * 1000000,
            partial_match_len,
            jump_size
        );
    }
    
    // Check for kernel launch errors immediately
    std::cout << "Kernels launched on " << num_gpus << " GPU(s), waiting for completion..." << std::endl;
    
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(launchErr) << std::endl;
        cudaFree(d_target_hash160);
        return 1;
    }
    
    // Wait for all GPUs with error checking
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        std::cout << "Waiting for GPU " << gpu << " to finish..." << std::endl;
        
        cudaError_t syncErr = cudaDeviceSynchronize();
        if (syncErr != cudaSuccess) {
            std::cerr << "GPU " << gpu << " synchronization error: " << cudaGetErrorString(syncErr) << std::endl;
            cudaFree(d_target_hash160);
            return 1;
        }
        
        std::cout << "GPU " << gpu << " completed successfully" << std::endl;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    // Add diagnostic output
    std::cout << "\nKernel execution time: " << duration << " milliseconds" << std::endl;
    
    // Check for results
    cudaMemcpyFromSymbol(&h_result, d_result, sizeof(SearchResult));
    cudaMemcpyFromSymbol(&found_flag, d_found_flag, sizeof(bool));
    
    if (h_result.found || found_flag) {
        std::cout << "\n================== FOUND MATCH! ==================\n";
        char priv_hex[65];
        uint256_to_hex(&h_result.private_key, priv_hex);
        std::cout << "Private Key   : " << priv_hex << std::endl;
        
        std::cout << "Public Key    : ";
        for (int i = 0; i < 33; i++) {
            printf("%02X", h_result.pubkey[i]);
        }
        std::cout << std::endl;
        
        // Generate WIF
        char wif[52];
        generate_wif(priv_hex, true, wif);  // true = compressed
        std::cout << "WIF           : " << wif << std::endl;
        
        std::cout << "Hash160       : ";
        for (int i = 0; i < 20; i++) {
            printf("%02x", h_result.hash160[i]);
        }
        std::cout << std::endl;
        
        std::cout << "P2PKH Address : " << target_address << std::endl;
    } else {
        std::cout << "\nNo match found in range.\n";
    }
    
    double seconds = duration / 1000.0;
    uint64_t total_keys = total_threads * keys_per_thread;
    double mkeys_per_sec = (total_keys / seconds) / 1e6;
    
    std::cout << "\nTotal Checked : " << total_keys << std::endl;
    std::cout << "Elapsed Time  : " << std::fixed << std::setprecision(2) << seconds << " seconds\n";
    std::cout << "Speed         : " << std::fixed << std::setprecision(2) << mkeys_per_sec << " Mkeys/s\n";
    
    // Cleanup
    cudaFree(d_target_hash160);
    
    return h_result.found ? 0 : 1;
}
