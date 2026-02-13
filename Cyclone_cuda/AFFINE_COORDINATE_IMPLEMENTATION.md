# Affine Coordinate ECC Implementation

## Overview

This document describes the replacement of Jacobian coordinate elliptic curve operations with affine coordinates in the CUDA secp256k1 implementation (`cuda_secp256k1.cuh`).

## Changes Made

### 1. Point Structure (`point_t`)

Updated to use affine coordinates:
- **Z coordinate**: Always `1` for valid points, `0` for point at infinity
- Simplified from Jacobian coordinates where Z could be any non-zero value

```c
typedef struct {
    uint256_t x;
    uint256_t y;
    uint256_t z;  // Always 1 for valid points, 0 for point at infinity
} point_t;
```

### 2. Point Doubling (`point_double`)

Replaced Jacobian doubling with standard affine doubling formula:

**Formula:**
- `s = (3 * x²) / (2 * y) mod p` (for secp256k1 where a=0)
- `x' = s² - 2*x mod p`
- `y' = s*(x - x') - y mod p`
- `z' = 1`

**Key features:**
- Uses modular inverse for division operations
- Handles point at infinity cases (z=0 or y=0)
- Simpler than Jacobian formula
- Each result has z=1 (affine form)

### 3. Point Addition (`point_add`)

Replaced Jacobian addition with standard affine addition formula:

**Formula:**
- `h = x₂ - x₁ mod p`
- `r = y₂ - y₁ mod p`
- `s = r / h mod p`
- `x' = s² - x₁ - x₂ mod p`
- `y' = s*(x₁ - x') - y₁ mod p`
- `z' = 1`

**Special cases:**
- If `p1` is point at infinity: return `p2`
- If `p2` is point at infinity: return `p1`
- If `h == 0 && r == 0`: points are equal, use point doubling
- If `h == 0 && r != 0`: points are negatives, return point at infinity

### 4. Point to Affine Conversion (`point_to_affine`)

Simplified to a no-op function since points are already in affine form:
- Points always have z=1 or z=0
- Kept for API compatibility with existing code
- No conversion needed

### 5. Scalar Multiplication (`point_mul`)

No changes required:
- Double-and-add algorithm works with both coordinate systems
- Uses `point_double` and `point_add` which are now affine
- Continues to work correctly with affine operations

## Mathematical Correctness

The affine formulas are mathematically equivalent to Jacobian formulas but simpler:

**Advantages:**
- Clearer mathematical correspondence to standard ECC textbooks
- Simpler to understand and verify
- Z always equals 1 for valid points

**Trade-offs:**
- Requires modular inverse operations (expensive but unavoidable in affine coordinates)
- The existing uint256_mod_inv implementation uses Fermat's little theorem: `a⁻¹ = a^(p-2) mod p`

## Implementation Details

### Modular Arithmetic Functions Used

All operations use tested functions from `cuda_uint256.cuh`:
- `uint256_mod_add()` - Modular addition
- `uint256_mod_sub()` - Modular subtraction (handles negative results correctly)
- `uint256_mod_mul()` - Modular multiplication (uses fast secp256k1-specific reduction)
- `uint256_mod_sqr()` - Modular squaring
- `uint256_mod_inv()` - Modular inverse (Fermat's little theorem)

### Edge Cases Handled

1. **Point at Infinity**:
   - Represented as z=0
   - Properly handled in all operations
   - Identity element for point addition

2. **Point Doubling Special Cases**:
   - Input is point at infinity (z=0): return point at infinity
   - y-coordinate is zero: return point at infinity (tangent is vertical)

3. **Point Addition Special Cases**:
   - Either input is point at infinity: return the other point
   - Points are equal: use point doubling
   - Points are negatives: return point at infinity

## Testing

### Test Program

Created `test_affine_ecc.cpp` for validation:
- Tests point doubling with generator G
- Tests scalar multiplication with known private key `0x6AC3875`
- Expected public key: `031A864BAE3922F351F1B57CFDD827C25B7E093CB9C88A72C1CD893D9F90F44ECE`
- Expected address: `128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k`

### Verification

The implementation:
- ✅ Uses mathematically correct affine formulas
- ✅ Handles all edge cases (point at infinity, equal points, negative points)
- ✅ Uses tested modular arithmetic functions
- ✅ Maintains API compatibility (point_to_affine kept as no-op)
- ✅ Documented with clear comments

## References

### Standard ECC Formulas

Affine point doubling for curve y² = x³ + ax + b:
- λ = (3x₁² + a) / (2y₁) mod p
- x₃ = λ² - 2x₁ mod p
- y₃ = λ(x₁ - x₃) - y₁ mod p

Affine point addition for P₁ ≠ ±P₂:
- λ = (y₂ - y₁) / (x₂ - x₁) mod p
- x₃ = λ² - x₁ - x₂ mod p
- y₃ = λ(x₁ - x₃) - y₁ mod p

For secp256k1: a = 0, b = 7

### Resources

- [secp256k1 Specification](https://www.secg.org/sec2-v2.pdf)
- [Bitcoin Wiki - Secp256k1](https://en.bitcoin.it/wiki/Secp256k1)
- [Elliptic Curve Cryptography Standards](http://www.secg.org/)

## Conclusion

The affine coordinate implementation provides a mathematically correct and simpler alternative to Jacobian coordinates for ECC operations. All formulas match standard references, edge cases are properly handled, and the implementation uses existing tested modular arithmetic functions.

The code is ready for CUDA compilation and testing with the known test key `0x6AC3875` to verify correct public key generation and address derivation.
