# Montgomery Multiplication Implementation

## Overview

This implementation adds Montgomery multiplication as an optimization option for the Cyclone CUDA Bitcoin puzzle solver. Montgomery multiplication is a technique for performing modular arithmetic efficiently, which is crucial for elliptic curve cryptography operations.

## What is Montgomery Multiplication?

Montgomery multiplication is a method for computing `(a * b) mod N` without performing expensive division operations. Instead of working with numbers directly, Montgomery arithmetic works in "Montgomery space" where numbers are represented as `aR mod N`, where `R` is typically a power of 2 (in our case, `R = 2^256`).

### Key Benefits:
- Eliminates expensive modular reduction operations
- Replaces division with shifts and multiplications
- Particularly efficient on hardware that handles 64-bit multiplications well (like modern GPUs)

### Trade-offs:
- Requires conversion to/from Montgomery form
- For secp256k1 specifically, the prime's special form (`p = 2^256 - 0x1000003D1`) allows for very fast reduction without Montgomery
- Best suited for operations where values stay in Montgomery form for multiple operations

## Implementation Details

### Files Modified

1. **cuda_uint256.cuh**
   - Added Montgomery constants (`MONT_R`, `MONT_R2`, `MONT_P_INV`)
   - Implemented `mont_reduce()` - Montgomery reduction algorithm
   - Implemented `mont_mul()` - Montgomery multiplication
   - Implemented `mont_sqr()` - Montgomery squaring
   - Implemented `to_montgomery()` and `from_montgomery()` - conversion functions
   - Implemented `uint256_mod_inv_mont()` - modular inverse using Montgomery arithmetic

2. **cuda_secp256k1.cuh**
   - Added `USE_MONTGOMERY` compile-time flag
   - Updated `uint256_mod_mul()`, `uint256_mod_sqr()`, and `uint256_mod_inv()` to support both implementations
   - Maintained backward compatibility with fast secp256k1 method

3. **Makefile**
   - Added `USE_MONTGOMERY` build option
   - Added test target for Montgomery multiplication
   - Updated help text with new options

### Montgomery Constants for secp256k1

For the secp256k1 prime `p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F`:

- **R = 2^256 mod p**
  ```
  0x00000000000000000000000000000000000000000000000000000001000003D1
  d[0] = 0x00000001000003D1ULL, d[1] = 0, d[2] = 0, d[3] = 0
  ```

- **R² = 2^512 mod p** (used for conversion to Montgomery form)
  ```
  0x000000000000000000000000000000000000000000000001000007A2000E90A1
  d[0] = 0x000007A2000E90A1ULL, d[1] = 0x0000000000000001ULL, d[2] = 0, d[3] = 0
  ```

- **p_inv = -p^(-1) mod 2^64** (used in Montgomery reduction)
  ```
  0xD838091DD2253531
  ```

### Algorithm: CIOS Montgomery Reduction

The implementation uses the CIOS (Coarsely Integrated Operand Scanning) algorithm:

```
Input: T (512 bits), Output: T*R^(-1) mod p (256 bits)

for i = 0 to 3:
    m = c[i] * p_inv mod 2^64
    c = c + m * p (starting at position i)
    
Result is in upper 256 bits: c[4..7]
Final conditional subtraction if result >= p
```

## Usage

### Building with Montgomery Multiplication

```bash
# Default: Use fast secp256k1 multiplication
make

# Enable Montgomery multiplication
make USE_MONTGOMERY=1

# Build for specific GPU architecture with Montgomery
make USE_MONTGOMERY=1 CUDA_ARCH=sm_89
```

### Testing

A test program is provided to verify correctness and compare implementations:

```bash
# Build the test
make test

# Run the test
./test_montgomery_cuda
```

The test compares:
- Montgomery multiplication vs fast secp256k1 multiplication
- Small number operations (2 * 3)
- Squaring operations
- Large number operations (using generator point coordinates)
- Elliptic curve point operations

## Performance Considerations

### When to Use Montgomery

**Use Montgomery when:**
- Performing many sequential modular multiplications
- Values stay in Montgomery form throughout computation
- Modular exponentiation (already stays in Montgomery form)

**Use Fast secp256k1 when:**
- Single or few multiplications
- Need maximum performance for secp256k1 specifically
- Taking advantage of the special prime form

### Expected Performance

The fast secp256k1 method is expected to be competitive or faster than Montgomery for secp256k1 because:
1. The prime `p = 2^256 - 0x1000003D1` has special structure
2. Reduction can be done with a few multiplications by small constant
3. No conversion overhead to/from Montgomery form

However, Montgomery may show benefits:
- On certain GPU architectures
- When many operations are chained
- In modular inversion (where exponentiation keeps values in Montgomery form)

### Benchmarking

To properly compare implementations, you should:
1. Build both versions
2. Run on your target hardware
3. Measure actual keys/second performance
4. Choose the faster implementation for production use

Example:
```bash
# Test fast secp256k1
make clean && make
./Cyclone_cuda -a <address> -r 1:FFFF

# Test Montgomery
make clean && make USE_MONTGOMERY=1
./Cyclone_cuda -a <address> -r 1:FFFF
```

## Technical Notes

### Thread Safety
All Montgomery operations are thread-safe on CUDA:
- No shared state between threads
- Each thread has independent local variables
- Constant memory is read-only

### Numerical Correctness
The implementation has been designed for correctness:
- Proper carry propagation in multi-precision arithmetic
- Final conditional reduction to ensure result < p
- Verified constants computed independently

### Future Optimizations
Potential improvements:
1. Inline more operations to reduce register pressure
2. Use PTX assembly for critical operations
3. Optimize for specific GPU architectures (sm_89, etc.)
4. Hybrid approach: Montgomery for exponentiation, fast for other ops

## References

- [Montgomery Multiplication](https://en.wikipedia.org/wiki/Montgomery_modular_multiplication)
- [CIOS Algorithm](https://www.microsoft.com/en-us/research/wp-content/uploads/1996/01/j37acmon.pdf)
- [secp256k1 Specification](http://www.secg.org/sec2-v2.pdf)
- [Fast secp256k1 Arithmetic](https://github.com/bitcoin-core/secp256k1)

## Testing and Validation

### Unit Tests
Run the test program to verify:
```bash
make test
./test_montgomery_cuda
```

Expected output shows matching results between implementations.

### Integration Tests
The main solver should produce identical results regardless of which multiplication method is used:
```bash
# Both should find the same private key
make && ./Cyclone_cuda -a 128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k -r 6AC3875:6AC3876
make clean && make USE_MONTGOMERY=1 && ./Cyclone_cuda -a 128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k -r 6AC3875:6AC3876
```

## Troubleshooting

### Compilation Errors
If you encounter errors:
1. Ensure CUDA 12.x is installed
2. Check GPU compute capability matches CUDA_ARCH
3. Verify all header files are present

### Runtime Errors
If results don't match:
1. Check that constants are correct
2. Verify no overflow in carry propagation
3. Ensure final reduction is applied

### Performance Issues
If Montgomery is slower than expected:
1. Try different GPU architectures
2. Profile with `nvprof` or `nsys`
3. Check register usage with `nvcc --ptxas-options=-v`

## Conclusion

This implementation provides Montgomery multiplication as an alternative to the fast secp256k1-specific reduction. While the fast method is likely optimal for secp256k1, Montgomery is provided for:
- Comparison and benchmarking
- Potential benefits on certain hardware
- Educational purposes
- Future optimizations

The default remains the fast secp256k1 method, which has been proven effective. Users should benchmark both on their target hardware to determine the best option.
