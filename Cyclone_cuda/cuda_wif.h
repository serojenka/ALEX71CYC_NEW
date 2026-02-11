#ifndef CUDA_WIF_H
#define CUDA_WIF_H

#include <cstring>
#include <stdint.h>

// Base58 encoding (for WIF generation)
static void base58_encode(const uint8_t* input, size_t input_len, char* output) {
    static const char* BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    
    // Count leading zeros
    size_t leading_zeros = 0;
    while (leading_zeros < input_len && input[leading_zeros] == 0) {
        leading_zeros++;
    }
    
    // Allocate temporary space for encoding
    size_t capacity = input_len * 138 / 100 + 1; // log(256) / log(58), rounded up
    uint8_t* temp = new uint8_t[capacity];
    memset(temp, 0, capacity);
    
    // Convert to base58
    for (size_t i = leading_zeros; i < input_len; i++) {
        int carry = input[i];
        for (size_t j = 0; j < capacity; j++) {
            carry += 256 * temp[capacity - 1 - j];
            temp[capacity - 1 - j] = carry % 58;
            carry /= 58;
        }
    }
    
    // Skip leading zeros in temp
    size_t start_idx = 0;
    while (start_idx < capacity && temp[start_idx] == 0) {
        start_idx++;
    }
    
    // Generate output string
    size_t output_idx = 0;
    
    // Add '1's for leading zeros
    for (size_t i = 0; i < leading_zeros; i++) {
        output[output_idx++] = '1';
    }
    
    // Add base58 digits
    for (size_t i = start_idx; i < capacity; i++) {
        output[output_idx++] = BASE58_ALPHABET[temp[i]];
    }
    
    output[output_idx] = '\0';
    delete[] temp;
}

// SHA256 double hash (for checksum) - placeholder version
// This is used only by the old generate_wif if OpenSSL is not available
static void sha256_double(const uint8_t* input, size_t len, uint8_t output[32]) {
    // Simple placeholder checksum (NOT cryptographically secure)
    // In production, use proper SHA256 implementation or OpenSSL
    memset(output, 0, 32);
    for (size_t i = 0; i < len; i++) {
        output[i % 32] ^= input[i];
    }
}

// Simple SHA256 implementation for host (basic version for checksums)
// Conditional OpenSSL support
#ifdef USE_OPENSSL
#include <openssl/sha.h>

static void sha256_double_impl(const uint8_t* input, size_t len, uint8_t output[32]) {
    uint8_t temp[32];
    SHA256(input, len, temp);
    SHA256(temp, 32, output);
}
#else
// Fallback: simple checksum (NOT cryptographically secure - for demo only)
static void sha256_double_impl(const uint8_t* input, size_t len, uint8_t output[32]) {
    memset(output, 0, 32);
    for (size_t i = 0; i < len; i++) {
        output[i % 32] ^= input[i];
        output[(i + 1) % 32] ^= (input[i] >> 4);
    }
}
#endif

// Update generate_wif to use the implementation
static void generate_wif(const char* privkey_hex, bool compressed, char* wif_output) {
    // Decode hex private key
    uint8_t privkey[32];
    for (int i = 0; i < 32; i++) {
        char byte_str[3] = {privkey_hex[i*2], privkey_hex[i*2+1], 0};
        privkey[i] = (uint8_t)strtol(byte_str, nullptr, 16);
    }
    
    // Build WIF payload
    uint8_t payload[38];
    size_t payload_len = 0;
    
    payload[payload_len++] = 0x80;
    memcpy(payload + payload_len, privkey, 32);
    payload_len += 32;
    
    if (compressed) {
        payload[payload_len++] = 0x01;
    }
    
    // Calculate checksum
    uint8_t hash[32];
    sha256_double_impl(payload, payload_len, hash);
    
    // Append checksum
    memcpy(payload + payload_len, hash, 4);
    payload_len += 4;
    
    // Encode to base58
    base58_encode(payload, payload_len, wif_output);
}

#endif // CUDA_WIF_H
