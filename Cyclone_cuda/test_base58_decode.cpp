#include <iostream>
#include <iomanip>
#include "cuda_utils.h"

int main() {
    const char* test_address = "128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k";
    const char* expected_hash160 = "0c7aaf6caa7e5424b63d317f0f8f1f9fa40d5560";
    
    uint8_t hash160[20];
    
    std::cout << "Testing Base58 decoding for: " << test_address << std::endl;
    std::cout << "Expected hash160: " << expected_hash160 << std::endl;
    
    if (decode_address_to_hash160(test_address, hash160)) {
        std::cout << "Decoded hash160:  ";
        for (int i = 0; i < 20; i++) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)hash160[i];
        }
        std::cout << std::endl;
        
        // Check if it matches
        bool matches = true;
        for (int i = 0; i < 20; i++) {
            uint8_t expected_byte = 0;
            char byte_str[3] = {expected_hash160[i*2], expected_hash160[i*2+1], 0};
            expected_byte = (uint8_t)strtol(byte_str, NULL, 16);
            if (hash160[i] != expected_byte) {
                matches = false;
                break;
            }
        }
        
        if (matches) {
            std::cout << "\n✓ TEST PASSED: Base58 decoding is correct!" << std::endl;
            return 0;
        } else {
            std::cout << "\n✗ TEST FAILED: Base58 decoding is incorrect!" << std::endl;
            return 1;
        }
    } else {
        std::cout << "✗ Failed to decode address!" << std::endl;
        return 1;
    }
}
