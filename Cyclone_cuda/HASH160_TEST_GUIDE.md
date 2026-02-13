# Hash160 Implementation Test Guide

This document explains how to verify the CUDA RIPEMD160 and SHA256 implementations are correct.

## Test Case

Known Bitcoin key with confirmed address:

- **Private Key (hex)**: `0000000000000000000000000000000000000000000000000000000006AC3875`
- **Public Key (compressed)**: `031A864BAE3922F351F1B57CFDD827C25B7E093CB9C88A72C1CD893D9F90F44ECE`
- **Bitcoin Address**: `128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k`

## Expected Hash Values

### SHA256 of Public Key

**Input**: 33-byte compressed public key  
**Output**: 32-byte SHA256 hash

```
8a8904be5cb8e8d9907de7abd33781c42781e43408057dbe62bc5aface9a5875
```

### RIPEMD160 of SHA256

**Input**: 32-byte SHA256 hash (from above)  
**Output**: 20-byte RIPEMD160 hash (Hash160)

```
0c7aaf6caa7e5424b63d317f0f8f1f9fa40d5560
```

## Padding Details

### SHA256 Padding (FIPS 180-4)

For 33-byte compressed public key input:

```
Input:  33 bytes (public key)
        + 1 byte (0x80)
        + 22 bytes (zero padding)
        + 8 bytes (length = 264 bits as big-endian)
        = 64 bytes total (1 block)
```

Length encoding (big-endian):
```
0x0000000000000108 = 264 bits in big-endian
```

### RIPEMD160 Padding (RFC 1320)

For 32-byte SHA256 output:

```
Input:  32 bytes (SHA256 hash)
        + 1 byte (0x80)
        + 23 bytes (zero padding)
        + 8 bytes (length = 256 bits as little-endian)
        = 64 bytes total (1 block)
```

Length encoding (little-endian):
```
0x0001000000000000 = 256 bits in little-endian
```

## Verification Methods

### Method 1: Python Verification

Run the included Python script:

```bash
cd Cyclone_cuda
python3 verify_hash160.py
```

Expected output:
```
✓ TEST PASSED: Hash160 matches expected value
```

### Method 2: C++ with OpenSSL

Compile and run the CPU test:

```bash
cd Cyclone_cuda
g++ -o test_hash_cpu test_hash_cpu.cpp -lssl -lcrypto
./test_hash_cpu
```

Expected output:
```
Result: PASS
```

### Method 3: CUDA Test (requires NVIDIA GPU)

Compile and run the CUDA test:

```bash
cd Cyclone_cuda
nvcc -O3 -arch=sm_70 -std=c++14 -o test_hash160 test_hash160.cu
./test_hash160
```

Expected output:
```
✓ TEST PASSED: Hash160 matches expected value
```

### Method 4: Full Application Test

Run the full Cyclone CUDA application with the known key range:

```bash
cd Cyclone_cuda
make
./Cyclone_cuda -a 128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k -r 6AC3870:6AC3880
```

Expected output:
```
================== FOUND MATCH! ==================
Private Key   : 0000000000000000000000000000000000000000000000000000000006AC3875
Public Key    : 031A864BAE3922F351F1B57CFDD827C25B7E093CB9C88A72C1CD893D9F90F44ECE
P2PKH Address : 128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k
```

## Implementation Details

### SHA256 (cuda_hash.cuh lines 102-157)

- Follows FIPS 180-4 standard
- Uses big-endian byte order
- Length encoded as 64-bit big-endian
- Processes data in 64-byte (512-bit) blocks

### RIPEMD160 (cuda_hash.cuh lines 267-328)

- Follows RFC 1320 standard
- Uses little-endian byte order
- Length encoded as 64-bit little-endian
- Processes data in 64-byte (512-bit) blocks

### Hash160 (cuda_hash.cuh lines 331-341)

Pipeline:
1. SHA256(33-byte compressed public key) → 32 bytes
2. RIPEMD160(32-byte SHA256 output) → 20 bytes (Hash160)

## Common Issues and Solutions

### Issue: CUDA test fails to compile

**Solution**: Ensure CUDA Toolkit 12.x is installed:
```bash
nvcc --version  # Should show CUDA 12.x
```

### Issue: Wrong hash160 output

**Possible causes**:
1. Incorrect padding implementation
2. Wrong endianness in length encoding
3. Incorrect word loading (endianness)
4. Bug in transform functions

**Debug steps**:
1. Verify SHA256 output matches expected value
2. Verify RIPEMD160 output matches expected value
3. Check that RIPEMD160 receives 32 bytes (not 33)
4. Verify padding is applied correctly

### Issue: CPU test passes but CUDA test fails

**Possible causes**:
1. Different padding in CUDA vs CPU
2. Endianness issues on GPU
3. Integer overflow in CUDA code

**Debug steps**:
1. Add debug prints in CUDA kernel
2. Compare intermediate values with CPU version
3. Check sizeof() for types in CUDA vs CPU

## References

- **FIPS 180-4**: SHA-256 specification
- **RFC 1320**: RIPEMD-160 specification
- **Bitcoin Wiki**: Technical background on addresses

## Additional Test Cases

You can test with other known keys from puzzle challenges:

| Private Key | Public Key | Address |
|-------------|------------|---------|
| More test cases can be added here as needed |

## Automated Testing

To run all tests:

```bash
cd Cyclone_cuda
./run_all_tests.sh  # (create this script if needed)
```

## Conclusion

If all tests pass, the RIPEMD160 and SHA256 implementations are correct and match the standard Bitcoin address generation algorithm.
