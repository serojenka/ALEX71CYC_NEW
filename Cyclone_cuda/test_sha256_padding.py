#!/usr/bin/env python3
"""Test SHA256 padding for 33-byte input."""

import hashlib

def trace_sha256_padding():
    """Trace the exact padding that should be used for SHA256 on 33-byte pubkey"""
    
    # The 33-byte compressed public key
    pubkey_hex = "031A864BAE3922F351F1B57CFDD827C25B7E093CB9C88A72C1CD893D9F90F44ECE"
    pubkey_bytes = bytes.fromhex(pubkey_hex)
    
    print("="*70)
    print("SHA256 Padding Details for 33-byte Public Key")
    print("="*70)
    print(f"\nInput (compressed pubkey): {pubkey_hex}")
    print(f"Input length: {len(pubkey_bytes)} bytes = {len(pubkey_bytes) * 8} bits")
    
    # Build the padded message
    padded = bytearray(pubkey_bytes)
    
    # Append 0x80
    padded.append(0x80)
    print(f"\nAfter appending 0x80: {len(padded)} bytes")
    
    # Pad with zeros to 56 bytes (448 bits)
    while len(padded) < 56:
        padded.append(0x00)
    print(f"After zero padding: {len(padded)} bytes")
    
    # Append length in bits as 64-bit big-endian
    bit_length = len(pubkey_bytes) * 8  # 264 bits
    # Big-endian: most significant byte first
    length_bytes = bit_length.to_bytes(8, 'big')
    padded.extend(length_bytes)
    
    print(f"After appending length: {len(padded)} bytes")
    print(f"Length value: {bit_length} bits = 0x{length_bytes.hex()} (big-endian)")
    
    # Display the padded block
    print("\nFinal 64-byte padded block:")
    print("-" * 70)
    for i in range(0, 64, 16):
        hex_str = ' '.join(f"{b:02x}" for b in padded[i:i+16])
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in padded[i:i+16])
        print(f"  {i:02d}: {hex_str}  |{ascii_str}|")
    
    # Verify with Python's hashlib
    result = hashlib.sha256(pubkey_bytes).digest()
    print(f"\nSHA256 result: {result.hex()}")
    print("\nExpected:      8a8904be5cb8e8d9907de7abd33781c42781e43408057dbe62bc5aface9a5875")

if __name__ == "__main__":
    trace_sha256_padding()
