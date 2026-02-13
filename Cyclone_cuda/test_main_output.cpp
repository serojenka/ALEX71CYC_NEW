#include <iostream>
#include <iomanip>
#include <string>
#include "cuda_utils.h"

int main() {
    std::string target_address = "128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k";
    
    std::cout << "=== Cyclone CUDA - GPU Bitcoin Puzzle Solver ===" << std::endl;
    std::cout << "CUDA 12 - Multi-GPU Support" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Target Address: " << target_address << std::endl;
    
    // Decode target address to hash160
    uint8_t target_hash160[20];
    if (!decode_address_to_hash160(target_address.c_str(), target_hash160)) {
        std::cerr << "Error: Failed to decode address. Provide valid P2PKH address or hash160.\n";
        return 1;
    }
    
    // Print target hash160 for verification
    std::cout << "Target Hash160: ";
    for (int i = 0; i < 20; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)target_hash160[i];
    }
    std::cout << std::dec << std::endl;
    
    std::cout << std::endl;
    std::cout << "✓ Address decoded successfully!" << std::endl;
    std::cout << "✓ Checksum verified!" << std::endl;
    
    return 0;
}
