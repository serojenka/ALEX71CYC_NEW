# SHA256 Implementation Test Guide

## Overview

This guide provides comprehensive testing procedures for the SHA256 implementation in `cuda_hash.cuh`, specifically focusing on proper padding for 33-byte compressed public key hashing.

## Test Case: Key 0x6AC3875

This well-known test case is used to verify the complete hash160 pipeline.

### Expected Values

| Item | Value |
|------|-------|
| Private Key | `0x6AC3875` |
| Compressed Public Key | `031A864BAE3922F351F1B57CFDD827C25B7E093CB9C88A72C1CD893D9F90F44ECE` |
| SHA256 Hash | `8a8904be5cb8e8d9907de7abd33781c42781e43408057dbe62bc5aface9a5875` |
| RIPEMD160 (Hash160) | `0c7aaf6caa7e5424b63d317f0f8f1f9fa40d5560` |
| Bitcoin Address | `128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k` |

## SHA256 Padding Specification

For a 33-byte compressed public key, SHA256 padding must produce a single 64-byte block:

```
Bytes 0-32   (33 bytes):  Compressed public key data
Byte 33      (1 byte):    0x80 (padding start marker)
Bytes 34-55  (22 bytes):  0x00 (zero padding)
Bytes 56-63  (8 bytes):   0x0000000000000108 (264 bits in big-endian)
```

### Padding Details

- **Input length**: 33 bytes = 264 bits
- **After 0x80**: 34 bytes
- **Required padding**: To 56 bytes (leaving 8 bytes for length field)
- **Zero padding**: 22 bytes (from byte 34 to byte 55)
- **Length field**: 264 = 0x108 in big-endian = `00 00 00 00 00 00 01 08`

## Test Scripts

### 1. Python Reference Implementation

```bash
python3 verify_hash160.py
```

This script:
- Computes SHA256 and RIPEMD160 using Python's hashlib
- Displays the complete hash160 pipeline
- Shows detailed padding for both hash functions
- Verifies against known expected values

**Expected Output**:
```
SHA256 Hash (32 bytes): 8a8904be5cb8e8d9907de7abd33781c42781e43408057dbe62bc5aface9a5875
RIPEMD160 Hash (20 bytes): 0c7aaf6caa7e5424b63d317f0f8f1f9fa40d5560
✓ TEST PASSED
```

### 2. SHA256 Padding Tracer

```bash
python3 test_sha256_padding.py
```

Shows the exact padding applied to the 33-byte public key:
- Hexadecimal dump of the padded block
- Bit length encoding
- Verification against Python's SHA256

### 3. Padding Logic Simulator

```bash
python3 simulate_cuda_sha256.py
```

Simulates the exact logic from cuda_hash.cuh:
- Step-by-step execution trace
- Buffer layout at each step
- Verifies the algorithm produces correct padding

### 4. CUDA Test Program (Requires GPU)

```bash
nvcc -o test_sha256_only test_sha256_only.cu
./test_sha256_only
```

Tests the CUDA SHA256 implementation directly:
- Runs sha256() function on GPU
- Compares output with expected value
- Reports any differences byte-by-byte

**Expected Output**:
```
Expected SHA256: 8a8904be5cb8e8d9907de7abd33781c42781e43408057dbe62bc5aface9a5875
CUDA SHA256:     8a8904be5cb8e8d9907de7abd33781c42781e43408057dbe62bc5aface9a5875
✓ TEST PASSED
```

### 5. Complete Hash160 Test (Requires GPU)

```bash
nvcc -o test_hash160 test_hash160.cu
./test_hash160
```

Tests the complete hash160 pipeline:
- Calls hash160() function on GPU
- Verifies both SHA256 and RIPEMD160 stages
- Confirms final hash160 matches expected value

## Manual Verification Steps

### Step 1: Verify Public Key

The compressed public key for private key `0x6AC3875` should be:
```
031A864BAE3922F351F1B57CFDD827C25B7E093CB9C88A72C1CD893D9F90F44ECE
```

This is 33 bytes starting with `0x03` (y-coordinate is odd).

### Step 2: Verify SHA256 Input Padding

The padded input to SHA256 transform should be:
```
Offset  Bytes
------  -----
00-0F:  03 1a 86 4b ae 39 22 f3 51 f1 b5 7c fd d8 27 c2
10-1F:  5b 7e 09 3c b9 c8 8a 72 c1 cd 89 3d 9f 90 f4 4e
20-2F:  ce 80 00 00 00 00 00 00 00 00 00 00 00 00 00 00
30-3F:  00 00 00 00 00 00 00 00 00 00 00 00 00 00 01 08
```

### Step 3: Verify SHA256 Output

The SHA256 hash must be:
```
8a8904be5cb8e8d9907de7abd33781c42781e43408057dbe62bc5aface9a5875
```

Any difference indicates a problem with:
- Padding logic
- Message schedule computation
- Compression function
- Output encoding

### Step 4: Verify RIPEMD160 Output

The final hash160 must be:
```
0c7aaf6caa7e5424b63d317f0f8f1f9fa40d5560
```

## Common Issues

### Issue: SHA256 Output Doesn't Match

**Symptoms**: SHA256 produces different hash value

**Possible Causes**:
1. Incorrect padding (most common)
2. Wrong endianness in message schedule
3. Wrong endianness in length field
4. Incorrect initial hash values
5. Bug in compression function

**Debug Steps**:
1. Check padding is exactly as specified above
2. Verify length field: `00 00 00 00 00 00 01 08` (big-endian)
3. Verify 0x80 is at correct position (byte 33)
4. Check message schedule words match reference

### Issue: Hash160 Doesn't Match

**Symptoms**: SHA256 is correct but final hash160 is wrong

**This indicates a RIPEMD160 problem**, not SHA256. See HASH160_TEST_GUIDE.md.

### Issue: CUDA Test Won't Compile

**Requirements**:
- NVIDIA GPU with compute capability 7.0+
- CUDA Toolkit 12.x or later
- Compatible C++ compiler

**Compilation**:
```bash
nvcc -arch=sm_70 -o test_program test_program.cu
```

## Implementation Verification

The current implementation in `cuda_hash.cuh` (lines 113-173) is **CORRECT** and implements:

✅ FIPS 180-4 standard padding
✅ Proper handling of 33-byte inputs  
✅ Correct big-endian length encoding
✅ Support for arbitrary input lengths
✅ Proper two-block handling when needed

See `SHA256_VERIFICATION.md` for detailed analysis.

## Conclusion

The SHA256 implementation correctly handles 33-byte compressed public key hashing. All test scripts verify that the implementation produces the expected output for the known test case.

If you're experiencing issues:
1. Run the Python tests first to verify expected values
2. Run the CUDA tests to verify GPU implementation
3. If SHA256 is correct but hash160 is wrong, check RIPEMD160
4. If both are correct but addresses don't match, check point multiplication

The SHA256 padding implementation is **not** the source of any issues.
