// Standalone P2PKH address decoder and WIF generator.
// Uses a self-contained SHA-256 implementation (no AVX2, no OpenSSL required).

#include "p2pkh_decoder.h"
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace P2PKHDecoder {

// ============================================================
// Standalone SHA-256 (FIPS 180-4)
// ============================================================

static const uint32_t SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
    0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
    0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
    0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
    0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
    0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static inline uint32_t rotr32(uint32_t x, int n) { return (x>>n)|(x<<(32-n)); }
static inline uint32_t sha256_ch(uint32_t x,uint32_t y,uint32_t z){return(x&y)^(~x&z);}
static inline uint32_t sha256_maj(uint32_t x,uint32_t y,uint32_t z){return(x&y)^(x&z)^(y&z);}
static inline uint32_t sha256_Sigma0(uint32_t x){return rotr32(x,2)^rotr32(x,13)^rotr32(x,22);}
static inline uint32_t sha256_Sigma1(uint32_t x){return rotr32(x,6)^rotr32(x,11)^rotr32(x,25);}
static inline uint32_t sha256_sigma0(uint32_t x){return rotr32(x,7)^rotr32(x,18)^(x>>3);}
static inline uint32_t sha256_sigma1(uint32_t x){return rotr32(x,17)^rotr32(x,19)^(x>>10);}

static void sha256_compress(uint32_t s[8], const uint8_t blk[64]) {
    uint32_t W[64];
    for (int i=0;i<16;i++)
        W[i]=((uint32_t)blk[i*4]<<24)|((uint32_t)blk[i*4+1]<<16)|
             ((uint32_t)blk[i*4+2]<<8)|(uint32_t)blk[i*4+3];
    for (int i=16;i<64;i++)
        W[i]=sha256_sigma1(W[i-2])+W[i-7]+sha256_sigma0(W[i-15])+W[i-16];
    uint32_t a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
    for (int i=0;i<64;i++){
        uint32_t t1=h+sha256_Sigma1(e)+sha256_ch(e,f,g)+SHA256_K[i]+W[i];
        uint32_t t2=sha256_Sigma0(a)+sha256_maj(a,b,c);
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    s[0]+=a;s[1]+=b;s[2]+=c;s[3]+=d;s[4]+=e;s[5]+=f;s[6]+=g;s[7]+=h;
}

static void sha256_hash(const uint8_t *data, size_t len, uint8_t out[32]) {
    uint32_t s[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };
    uint8_t buf[64];
    size_t i = 0;
    while (i + 64 <= len) { sha256_compress(s, data+i); i += 64; }
    size_t rem = len - i;
    memcpy(buf, data+i, rem);
    buf[rem] = 0x80;
    if (rem >= 56) {
        memset(buf+rem+1, 0, 63-rem);
        sha256_compress(s, buf);
        memset(buf, 0, 56);
    } else {
        memset(buf+rem+1, 0, 55-rem);
    }
    uint64_t bits = (uint64_t)len * 8;
    for (int j=0;j<8;j++) buf[56+j]=(uint8_t)(bits>>(56-j*8));
    sha256_compress(s, buf);
    for (int j=0;j<8;j++){
        out[j*4]=(uint8_t)(s[j]>>24); out[j*4+1]=(uint8_t)(s[j]>>16);
        out[j*4+2]=(uint8_t)(s[j]>>8); out[j*4+3]=(uint8_t)(s[j]);
    }
}

// ============================================================
// Base58
// ============================================================

static const char* B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

static std::vector<uint8_t> base58_decode(const std::string& input) {
    std::vector<uint8_t> out;
    out.push_back(0);
    for (char c : input) {
        const char *p = strchr(B58, c);
        if (!p) throw std::invalid_argument("Invalid Base58 character");
        int carry = (int)(p - B58);
        for (auto &byte : out) {
            carry += 58 * byte;
            byte = (uint8_t)(carry & 0xFF);
            carry >>= 8;
        }
        while (carry > 0) { out.push_back((uint8_t)(carry&0xFF)); carry >>= 8; }
    }
    size_t lz = 0;
    for (char c : input) { if (c=='1') ++lz; else break; }
    std::reverse(out.begin(), out.end());
    while (!out.empty() && out[0]==0) out.erase(out.begin());
    std::vector<uint8_t> result(lz, 0);
    result.insert(result.end(), out.begin(), out.end());
    return result;
}

static std::string base58_encode(const std::vector<uint8_t>& input) {
    std::vector<int> digits(1, 0);
    for (uint8_t byte : input) {
        int carry = byte;
        for (auto &d : digits) { carry += d*256; d = carry%58; carry /= 58; }
        while (carry > 0) { digits.push_back(carry%58); carry /= 58; }
    }
    size_t lz = 0;
    for (uint8_t b : input) { if (b==0) ++lz; else break; }
    std::string encoded(lz, '1');
    for (auto it=digits.rbegin(); it!=digits.rend(); ++it)
        encoded += B58[*it];
    return encoded;
}

// ============================================================
// Public API
// ============================================================

std::vector<uint8_t> getHash160(const std::string& p2pkh_address) {
    std::vector<uint8_t> decoded = base58_decode(p2pkh_address);
    if (decoded.size() != 25)
        throw std::invalid_argument("Decoded address has wrong length");

    // Verify checksum: SHA256d(decoded[0:21]) first 4 bytes == decoded[21:25]
    uint8_t h1[32], h2[32];
    sha256_hash(decoded.data(), 21, h1);
    sha256_hash(h1, 32, h2);
    if (memcmp(h2, decoded.data()+21, 4) != 0)
        throw std::invalid_argument("Invalid address checksum");

    return std::vector<uint8_t>(decoded.begin()+1, decoded.begin()+21);
}

std::string compute_wif(const std::string& private_key_hex, bool compressed) {
    if (private_key_hex.size() != 64)
        throw std::invalid_argument("Private key must be 64 hex chars");

    std::vector<uint8_t> privkey(32);
    for (size_t i=0;i<32;i++) {
        char buf[3] = {private_key_hex[i*2], private_key_hex[i*2+1], 0};
        privkey[i] = (uint8_t)strtoul(buf, nullptr, 16);
    }

    std::vector<uint8_t> payload;
    payload.push_back(0x80);
    payload.insert(payload.end(), privkey.begin(), privkey.end());
    if (compressed) payload.push_back(0x01);

    uint8_t h1[32], h2[32];
    sha256_hash(payload.data(), payload.size(), h1);
    sha256_hash(h1, 32, h2);
    payload.insert(payload.end(), h2, h2+4);

    return base58_encode(payload);
}

} // namespace P2PKHDecoder
