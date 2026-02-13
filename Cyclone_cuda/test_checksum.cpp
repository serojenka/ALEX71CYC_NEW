#include <iostream>
#include <iomanip>
#include "cuda_utils.h"

void print_hex(const char* label, const uint8_t* data, int len) {
    std::cout << label;
    for (int i = 0; i < len; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)data[i];
    }
    std::cout << std::dec << std::endl;
}

int main() {
    std::cout << "Testing Base58 decoding with checksum verification" << std::endl;
    std::cout << std::endl;
    
    // Test valid address
    const char* valid_address = "128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k";
    const char* expected_hash160 = "0c7aaf6caa7e5424b63d317f0f8f1f9fa40d5560";
    
    std::cout << "Testing valid address: " << valid_address << std::endl;
    uint8_t hash160[20];
    if (decode_address_to_hash160(valid_address, hash160)) {
        std::cout << "✓ Decoding successful" << std::endl;
        print_hex("  Hash160: ", hash160, 20);
        std::cout << "  Expected: " << expected_hash160 << std::endl;
    } else {
        std::cout << "✗ Decoding failed!" << std::endl;
        return 1;
    }
    
    std::cout << std::endl;
    
    // Test invalid address (corrupted checksum)
    const char* invalid_address = "128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86l"; // Changed last char
    std::cout << "Testing invalid address (bad checksum): " << invalid_address << std::endl;
    if (decode_address_to_hash160(invalid_address, hash160)) {
        std::cout << "✗ Should have failed checksum validation!" << std::endl;
        return 1;
    } else {
        std::cout << "✓ Correctly rejected invalid address" << std::endl;
    }
    
    std::cout << std::endl << "All tests passed!" << std::endl;
    return 0;
}
