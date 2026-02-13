#!/usr/bin/env python3
"""
Test to verify the expected behavior of RIPEMD160 padding for hash160.

This script tests what the correct hash160 value should be for the known test case.
"""

import hashlib

def test_known_case():
    """Test the known case: key 0x6AC3875 -> address 128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k"""
    
    # Known test data
    public_key_hex = "031A864BAE3922F351F1B57CFDD827C25B7E093CB9C88A72C1CD893D9F90F44ECE"
    expected_hash160 = "0c7aaf6caa7e5424b63d317f0f8f1f9fa40d5560"
    
    # Convert to bytes
    pubkey = bytes.fromhex(public_key_hex)
    
    print("="*70)
    print("Testing Hash160 for Known Key 0x6AC3875")
    print("="*70)
    print(f"\nPublic Key (33 bytes): {public_key_hex}")
    print(f"Length: {len(pubkey)} bytes\n")
    
    # Step 1: SHA256 of public key
    sha256_hash = hashlib.sha256(pubkey).digest()
    print(f"SHA256 Hash (32 bytes): {sha256_hash.hex()}")
    
    # Step 2: RIPEMD160 of SHA256 hash
    ripemd160_hash = hashlib.new('ripemd160', sha256_hash).digest()
    print(f"RIPEMD160 Hash (20 bytes): {ripemd160_hash.hex()}")
    
    # Compare
    print(f"\nExpected Hash160: {expected_hash160}")
    print(f"Computed Hash160: {ripemd160_hash.hex()}")
    
    if ripemd160_hash.hex() == expected_hash160:
        print("\n✓ TEST PASSED: Hash160 matches expected value")
        return True
    else:
        print("\n✗ TEST FAILED: Hash160 does not match")
        return False

def trace_ripemd160_padding():
    """Trace the exact padding that should be used for RIPEMD160"""
    
    # The 32-byte SHA256 output
    sha256_hex = "8a8904be5cb8e8d9907de7abd33781c42781e43408057dbe62bc5aface9a5875"
    sha256_bytes = bytes.fromhex(sha256_hex)
    
    print("\n" + "="*70)
    print("RIPEMD160 Padding Details")
    print("="*70)
    print(f"\nInput (SHA256 output): {sha256_hex}")
    print(f"Input length: {len(sha256_bytes)} bytes")
    
    # Build the padded message
    padded = bytearray(sha256_bytes)
    
    # Append 0x80
    padded.append(0x80)
    print(f"\nAfter appending 0x80: {len(padded)} bytes")
    
    # Pad with zeros to 56 bytes
    while len(padded) < 56:
        padded.append(0x00)
    print(f"After zero padding: {len(padded)} bytes")
    
    # Append length in bits as 64-bit little-endian
    bit_length = len(sha256_bytes) * 8
    length_bytes = bit_length.to_bytes(8, 'little')
    padded.extend(length_bytes)
    
    print(f"After appending length: {len(padded)} bytes")
    print(f"Length value: {bit_length} bits = 0x{length_bytes.hex()}")
    
    # Display the padded block
    print("\nFinal 64-byte padded block:")
    print("-" * 70)
    for i in range(0, 64, 16):
        hex_str = ' '.join(f"{b:02x}" for b in padded[i:i+16])
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in padded[i:i+16])
        print(f"  {i:02d}: {hex_str}  |{ascii_str}|")
    
    # Verify
    result = hashlib.new('ripemd160', sha256_bytes).digest()
    print(f"\nRIPEMD160 result: {result.hex()}")

if __name__ == "__main__":
    success = test_known_case()
    trace_ripemd160_padding()
    
    print("\n" + "="*70)
    if success:
        print("All tests passed! The hash160 implementation is correct.")
    else:
        print("Tests failed! There is an issue with the hash160 implementation.")
    print("="*70)
