# Cyclone_gpu — Optimized GPU Bitcoin Puzzle Solver

GPU-accelerated Bitcoin address search ported from `Cyclone_avx2`, using the
same optimized batch algorithm: GTable precomputation, batch point operations,
and Fermat modular inversion in parallel across CUDA threads.

## Algorithm

The GPU implementation mirrors the AVX2 approach:

1. **Host-side GTable precomputation** — 8,192 affine points
   `GTable[i×256 + j] = (j+1) × 256^i × G` (same as `SECP256K1::Init()`).
   Copied once to GPU global memory.

2. **Precomputed plus/minus points** — `plus[i] = i×G`, `minus[i] = −(i×G)`
   for `i = 0…255`. Copied to GPU global memory.

3. **GPU kernel — one CUDA block = one batch of ~510 keys**
   - Thread 0 computes `base = priv × G` via GTable lookup (~32 additions).
   - Each thread `tid` independently computes
     `deltaX[tid] = plus[tid].x − base.x` and inverts it via
     Fermat's little theorem (`a^(p−2) mod p`) — fully parallel across
     all 256 threads.
   - Each thread `tid != 0` computes two affine points:
     `base + plus[tid]`  (key `priv + tid`) and
     `base + minus[tid]` (key `priv − tid`).
     Thread 0 handles only the base point itself.
   - Each thread serializes its points to compressed 33-byte pubkeys,
     computes `Hash160 = RIPEMD160(SHA256(pubkey))`, and compares with
     the target.
   - Thread 0 advances `base += 510×G` for the next batch.
   - Loop repeats for `BATCHES_PER_BLOCK` batches.

4. **SHA-256 and RIPEMD-160** — standard single-block GPU implementations
   with round constants in `__constant__` memory.

### Why faster than naive double-and-add?

| Metric | Naive (Cyclone_cuda) | Optimized (Cyclone_gpu) |
|---|---|---|
| Modular inversions / key | ~512 | ~1 |
| Base-point computation | 256 doublings + adds | 32 GTable lookups |
| Parallelism | 1 key / thread | 512 keys / block |

## Build

Requirements:
- CUDA Toolkit 11.x or 12.x
- NVIDIA GPU with Compute Capability ≥ 7.0 (Volta or newer)
- C++14 compiler (GCC 9+, MSVC 2019+)

```bash
# Default (sm_70, Volta)
make

# RTX 3090 / 3080 (Ampere sm_86)
make CUDA_ARCH=sm_86

# RTX 4090 (Ada Lovelace sm_89)
make CUDA_ARCH=sm_89

# More batches per block (higher throughput, less frequent host updates)
make CUDA_ARCH=sm_86 BATCHES_PER_BLOCK=16
```

### Windows x64 Build

Requirements:
- CUDA Toolkit 11.x or 12.x
- Visual Studio 2019/2022 (Desktop development with C++)
- x64 Native Tools Command Prompt (or Developer Command Prompt)

From the `Cyclone_gpu` directory:

```batch
build_windows.bat
```

Optional parameters:

```batch
build_windows.bat sm_89 16
```

This builds `Cyclone_gpu.exe` in the same directory.

Architecture examples:
- `sm_75` for RTX 20xx / GTX 16xx
- `sm_86` for RTX 30xx
- `sm_89` for RTX 40xx

## Usage

```
./Cyclone_gpu -a <Base58_P2PKH> -r <START:END> [options]

Options:
  -a <addr>         Target P2PKH Bitcoin address
  -r <start:end>    Private-key range in hex  (e.g. 1:ffffffff)
  -p <N>            Partial Hash160 prefix match length (hex nibbles, 1-40)
  -j <N>            Jump size on partial match (requires -p)
  -s                Save partial-match candidates to candidates.txt
  -t <id>           GPU device ID (default 0)
  --public-deny <N> Skip keys with N leading zero hex chars in pubkey X
  -h / --help       Print this help
```

### Example — Puzzle #66

```bash
./Cyclone_gpu \
  -a 13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so \
  -r 20000000000000000:3ffffffffffffffff
```

## Output Files

| File | Contents |
|---|---|
| `found_keys.txt` | Full match: `privkey pubkey WIF address` |
| `candidates.txt` | Partial matches (when `-s` enabled) |
| `progress.txt`   | Periodic progress snapshots (every 300 s) |

## Status Display

```
================= WORK IN PROGRESS =================
Target Address: 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH
Hash160       : 751e76e8199196d454941c45d1b3a323f1433bd6
GPU Device    : NVIDIA GeForce RTX 3090
Mkeys/s       : 850.23
Total Checked : 123456789
Elapsed Time  : 00:15:30
Range         : 20000000000000000:3ffffffffffffffff
Progress      : 12.3456 %
Progress Save : 3
```

## Performance Notes

- The GTable precomputation takes a few seconds on the first run but is
  only performed once per process invocation.
- `BATCHES_PER_BLOCK` controls the trade-off between GPU utilisation and
  host-side status update frequency. Values 8–16 work well in practice.
- For best performance, set `CUDA_ARCH` to match your GPU's actual
  compute capability (e.g. `sm_86` for RTX 30xx).
