#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cstring>
#include <stdint.h>

// Base58 alphabet for Bitcoin addresses
static const char* BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// Decode a base58 string to binary
static bool base58_decode(const char* input, uint8_t* output, size_t* output_len) {
    size_t input_len = strlen(input);
    
    // Count leading '1's (which represent leading zeros)
    size_t leading_zeros = 0;
    while (leading_zeros < input_len && input[leading_zeros] == '1') {
        leading_zeros++;
    }
    
    // Allocate temporary space for the result
    size_t capacity = input_len * 733 / 1000 + 1; // log(58) / log(256), rounded up
    uint8_t* temp = new uint8_t[capacity];
    memset(temp, 0, capacity);
    
    size_t temp_len = 0;
    
    // Process each base58 character
    for (size_t i = leading_zeros; i < input_len; i++) {
        // Find character in alphabet
        const char* ptr = strchr(BASE58_ALPHABET, input[i]);
        if (!ptr) {
            delete[] temp;
            return false; // Invalid character
        }
        
        int carry = ptr - BASE58_ALPHABET;
        
        // Multiply temp by 58 and add carry
        for (size_t j = 0; j < capacity; j++) {
            carry += 58 * temp[capacity - 1 - j];
            temp[capacity - 1 - j] = carry % 256;
            carry /= 256;
        }
        
        // Update length
        while (temp_len < capacity && temp[capacity - 1 - temp_len] != 0) {
            temp_len++;
        }
    }
    
    // Add leading zeros
    size_t result_len = leading_zeros + temp_len;
    if (result_len > *output_len) {
        delete[] temp;
        return false; // Output buffer too small
    }
    
    // Copy to output
    memset(output, 0, leading_zeros);
    memcpy(output + leading_zeros, temp + capacity - temp_len, temp_len);
    *output_len = result_len;
    
    delete[] temp;
    return true;
}

// Decode Bitcoin P2PKH address to hash160
static bool decode_address_to_hash160(const char* address, uint8_t hash160[20]) {
    uint8_t decoded[25]; // Version (1) + Hash160 (20) + Checksum (4)
    size_t decoded_len = 25;
    
    if (!base58_decode(address, decoded, &decoded_len)) {
        return false;
    }
    
    if (decoded_len != 25) {
        return false;
    }
    
    // Verify checksum (optional, but recommended)
    // ... SHA256(SHA256(decoded[0:21])) should match decoded[21:25]
    
    // Extract hash160 (skip version byte)
    memcpy(hash160, decoded + 1, 20);
    return true;
}

// Convert hex string to binary
static void hex_to_bytes(const char* hex, uint8_t* bytes, size_t byte_len) {
    size_t hex_len = strlen(hex);
    memset(bytes, 0, byte_len);
    
    for (size_t i = 0; i < hex_len && i / 2 < byte_len; i += 2) {
        char byte_str[3];
        byte_str[0] = hex[i];
        byte_str[1] = (i + 1 < hex_len) ? hex[i + 1] : '0';
        byte_str[2] = '\0';
        bytes[i / 2] = (uint8_t)strtol(byte_str, nullptr, 16);
    }
}

// Convert bytes to hex string
static void bytes_to_hex(const uint8_t* bytes, size_t byte_len, char* hex) {
    for (size_t i = 0; i < byte_len; i++) {
        sprintf(hex + i * 2, "%02x", bytes[i]);
    }
    hex[byte_len * 2] = '\0';
}

#endif // CUDA_UTILS_H
