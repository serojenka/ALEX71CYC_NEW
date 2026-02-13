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

int main() {
    const char* test_address = "128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k";
    
    std::cout << "Testing Base58 decoding for: " << test_address << std::endl;
    std::cout << std::endl;
    
    // First test: decode the full 25 bytes
    uint8_t decoded[25];
    size_t decoded_len = 25;
    
    if (base58_decode(test_address, decoded, &decoded_len)) {
        std::cout << "Successfully decoded " << decoded_len << " bytes" << std::endl;
        print_hex("Full decoded: ", decoded, decoded_len);
        std::cout << std::endl;
        
        if (decoded_len == 25) {
            std::cout << "Version byte: " << std::hex << (int)decoded[0] << std::dec << std::endl;
            print_hex("Hash160:      ", decoded + 1, 20);
            print_hex("Checksum:     ", decoded + 21, 4);
            std::cout << std::endl;
        }
    } else {
        std::cout << "Failed to decode!" << std::endl;
    }
    
    // Second test: use decode_address_to_hash160
    uint8_t hash160[20];
    if (decode_address_to_hash160(test_address, hash160)) {
        print_hex("Hash160 via decode_address_to_hash160: ", hash160, 20);
    } else {
        std::cout << "decode_address_to_hash160 failed!" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Expected hash160: 0c7aaf6caa7e5424b63d317f0f8f1f9fa40d5560" << std::endl;
    
    return 0;
}
