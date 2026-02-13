#include <iostream>
#include <iomanip>
#include <cstring>
#include "cuda_utils.h"

void print_hex(const char* label, const uint8_t* data, int len) {
    std::cout << label;
    for (int i = 0; i < len; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)data[i];
    }
    std::cout << std::dec << std::endl;
}

bool test_address(const char* address, const char* expected_hash160_hex, bool should_succeed) {
    std::cout << "Testing: " << address << std::endl;
    
    uint8_t hash160[20];
    bool result = decode_address_to_hash160(address, hash160);
    
    if (should_succeed) {
        if (!result) {
            std::cout << "  ✗ FAIL: Expected success but got failure" << std::endl;
            return false;
        }
        
        print_hex("  Hash160: ", hash160, 20);
        std::cout << "  Expected: " << expected_hash160_hex << std::endl;
        
        // Parse expected hash160
        uint8_t expected[20];
        for (int i = 0; i < 20; i++) {
            char byte_str[3] = {expected_hash160_hex[i*2], expected_hash160_hex[i*2+1], 0};
            expected[i] = (uint8_t)strtol(byte_str, NULL, 16);
        }
        
        if (memcmp(hash160, expected, 20) == 0) {
            std::cout << "  ✓ PASS: Hash160 matches" << std::endl;
            return true;
        } else {
            std::cout << "  ✗ FAIL: Hash160 mismatch" << std::endl;
            return false;
        }
    } else {
        if (result) {
            std::cout << "  ✗ FAIL: Expected failure but got success" << std::endl;
            return false;
        } else {
            std::cout << "  ✓ PASS: Correctly rejected invalid address" << std::endl;
            return true;
        }
    }
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "End-to-End Base58 Decoding Test" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << std::endl;
    
    int passed = 0;
    int total = 0;
    
    // Test 1: Known valid address from problem statement
    std::cout << "Test 1: Known valid address" << std::endl;
    total++;
    if (test_address("128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k", 
                     "0c7aaf6caa7e5424b63d317f0f8f1f9fa40d5560", 
                     true)) {
        passed++;
    }
    std::cout << std::endl;
    
    // Test 2: Another known Bitcoin address (1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa - Satoshi's address)
    std::cout << "Test 2: Satoshi's Genesis Block address" << std::endl;
    total++;
    if (test_address("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", 
                     "62e907b15cbf27d5425399ebf6f0fb50ebb88f18", 
                     true)) {
        passed++;
    }
    std::cout << std::endl;
    
    // Test 3: Invalid address (corrupted checksum)
    std::cout << "Test 3: Invalid address (bad checksum)" << std::endl;
    total++;
    if (test_address("128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86l", 
                     "", 
                     false)) {
        passed++;
    }
    std::cout << std::endl;
    
    // Test 4: Invalid address (invalid Base58 character)
    std::cout << "Test 4: Invalid address (invalid character)" << std::endl;
    total++;
    if (test_address("128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86O", // 'O' is not in Base58
                     "", 
                     false)) {
        passed++;
    }
    std::cout << std::endl;
    
    // Summary
    std::cout << "==================================================" << std::endl;
    std::cout << "Test Results: " << passed << "/" << total << " passed" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    if (passed == total) {
        std::cout << "✓ All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "✗ Some tests FAILED!" << std::endl;
        return 1;
    }
}
