#!/usr/bin/env python3
"""
Simulate the CUDA SHA256 implementation to verify it produces correct padding.
This directly mirrors the logic in cuda_hash.cuh sha256() function.
"""

def simulate_sha256_padding(data_len):
    """
    Simulate the padding logic from cuda_hash.cuh sha256() function.
    Returns the padded buffer layout.
    """
    print(f"\n{'='*70}")
    print(f"Simulating SHA256 padding for {data_len}-byte input")
    print(f"{'='*70}\n")
    
    # Simulate the state
    i = 0
    
    # Step 1: Process full 64-byte blocks
    print("Step 1: Process full 64-byte blocks")
    while i + 64 <= data_len:
        print(f"  Processing block at offset {i}")
        i += 64
    print(f"  After full blocks: i = {i}")
    
    # Step 2: Handle remaining bytes
    remaining = data_len - i
    print(f"\nStep 2: Handle remaining bytes")
    print(f"  remaining = {data_len} - {i} = {remaining}")
    print(f"  Copy {remaining} bytes to buffer[0..{remaining-1}]")
    
    # Build the buffer
    buffer = ['DATA'] * remaining
    
    # Step 3: Append 0x80
    print(f"\nStep 3: Append mandatory 0x80 byte")
    print(f"  buffer[{remaining}] = 0x80")
    buffer.append('0x80')
    remaining += 1
    print(f"  remaining = {remaining}")
    
    # Step 4: Check if length field fits
    print(f"\nStep 4: Check if length field (8 bytes) fits in current block")
    print(f"  if (remaining > 56): remaining={remaining}, condition={remaining > 56}")
    if remaining > 56:
        print(f"    YES - Pad current block and process it")
        while len(buffer) < 64:
            buffer.append('0x00')
        print(f"    Process this 64-byte block")
        print(f"    Start fresh block")
        buffer = []
        remaining = 0
    else:
        print(f"    NO - Length field fits in current block")
    
    # Step 5: Pad with zeros
    print(f"\nStep 5: Pad with zeros up to byte 56")
    print(f"  for (j = {remaining}; j < 56; j++) buffer[j] = 0")
    while len(buffer) < 56:
        buffer.append('0x00')
    print(f"  Buffer now has {len(buffer)} bytes")
    
    # Step 6: Append length
    bit_length = data_len * 8
    print(f"\nStep 6: Append message length in bits as 64-bit big-endian")
    print(f"  bitlen = {data_len} * 8 = {bit_length}")
    print(f"  Big-endian encoding: 0x{bit_length:016x}")
    
    # Simulate the big-endian encoding loop
    for j in range(8):
        shift = 56 - j * 8
        byte_val = (bit_length >> shift) & 0xff
        buffer.append(f'0x{byte_val:02x}')
    
    print(f"  Buffer now has {len(buffer)} bytes")
    
    # Display the buffer
    print(f"\n{'='*70}")
    print(f"Final buffer layout ({len(buffer)} bytes):")
    print(f"{'='*70}")
    for i in range(0, len(buffer), 16):
        line = buffer[i:i+16]
        print(f"  [{i:02d}..{min(i+15, len(buffer)-1):02d}]: {' '.join(str(x).ljust(4) for x in line)}")
    
    return buffer

if __name__ == "__main__":
    # Test with 33-byte input (compressed public key)
    buffer = simulate_sha256_padding(33)
    
    print(f"\n{'='*70}")
    print("Summary for 33-byte input:")
    print(f"{'='*70}")
    print(f"Bytes 0-32:   33-byte compressed public key")
    print(f"Byte 33:      0x80")
    print(f"Bytes 34-55:  22 zero bytes")
    print(f"Bytes 56-63:  0x0000000000000108 (264 bits in big-endian)")
    print(f"Total:        64 bytes (1 block)")
