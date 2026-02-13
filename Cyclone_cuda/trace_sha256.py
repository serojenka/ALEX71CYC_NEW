#!/usr/bin/env python3
"""Manually trace through SHA256 padding logic for 33-byte input to check the algorithm."""

def trace_logic():
    len_input = 33  # 33-byte public key
    
    print("="*70)
    print("Tracing SHA256 padding logic for 33-byte input")
    print("="*70)
    print(f"\nInput length: {len_input} bytes")
    
    # Step 1: Process full 64-byte blocks
    i = 0
    while i + 64 <= len_input:
        print(f"Processing full block starting at byte {i}")
        i += 64
    
    print(f"\nAfter processing full blocks, i = {i}")
    
    # Step 2: Handle remaining bytes
    remaining = len_input - i
    print(f"Remaining bytes to process: {remaining}")
    print(f"These {remaining} bytes go into buffer[0..{remaining-1}]")
    
    # Step 3: Append 0x80
    print(f"\nAppend 0x80 at buffer[{remaining}]")
    remaining += 1
    print(f"remaining is now {remaining}")
    
    # Step 4: Check if length field fits
    print(f"\nCheck if remaining ({remaining}) > 56:")
    if remaining > 56:
        print("  YES - Need an extra block")
        print(f"  Pad current block with zeros from buffer[{remaining}..63]")
        print("  Process this block")
        print("  Set remaining = 0")
        remaining = 0
    else:
        print("  NO - Length field fits in current block")
    
    # Step 5: Pad with zeros
    print(f"\nPad with zeros from buffer[{remaining}..55]")
    
    # Step 6: Append length
    print(f"Append 8-byte length field at buffer[56..63]")
    bitlen = len_input * 8
    print(f"Length in bits: {bitlen} = 0x{bitlen:016x}")
    print(f"Big-endian encoding: 0x{bitlen.to_bytes(8, 'big').hex()}")
    
    # Show final block layout
    print("\nFinal 64-byte block layout:")
    print(f"  [0..32]:   33-byte public key")
    print(f"  [33]:      0x80")
    print(f"  [34..55]:  22 zero bytes")
    print(f"  [56..63]:  0x{bitlen.to_bytes(8, 'big').hex()} (length)")

if __name__ == "__main__":
    trace_logic()
