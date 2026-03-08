#ifndef P2PKH_DECODER_H
#define P2PKH_DECODER_H

#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>

namespace P2PKHDecoder {

// Decode a P2PKH Bitcoin address to its 20-byte Hash160 payload.
std::vector<uint8_t> getHash160(const std::string& p2pkh_address);

// Generate a WIF-encoded private key from a 64-char hex string.
std::string compute_wif(const std::string& private_key_hex, bool compressed);

} // namespace P2PKHDecoder

#endif // P2PKH_DECODER_H
