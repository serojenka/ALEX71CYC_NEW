# Cyclone CUDA - GPU Bitcoin Puzzle Solver

High-performance GPU-accelerated Bitcoin puzzle solver using CUDA 12 and SECP256k1 elliptic curve cryptography.

## Features

- ✅ **GPU Acceleration**: Utilizes NVIDIA CUDA 12 for massive parallelization
- ✅ **Fast Modular Arithmetic**: Optimized secp256k1-specific multiplication using p = 2^256 - 0x1000003D1
- ✅ **Montgomery Multiplication**: Optional Montgomery arithmetic for modular operations (compile-time selectable)
- ✅ **High Performance**: 10-100x faster than repeated addition approach
- ✅ **Multi-GPU Support**: Distribute workload across multiple GPUs
- ✅ **Independent Threads**: Each GPU thread operates on separate key ranges
- ✅ **Random Search**: Probabilistic key search capability
- ✅ **Partial Matching**: Jump forward after finding partial hash matches
- ✅ **CLI Compatible**: Maintains command-line interface compatibility with CPU version
- ✅ **Cross-Platform**: Builds on Linux and Windows (Win64)

## Performance Optimization

This implementation provides two highly optimized modular multiplication algorithms:

### 1. Fast secp256k1-specific Multiplication (Default)
Leverages the special form of the secp256k1 prime:

```
p = 2^256 - 0x1000003D1 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
```

This special form allows for extremely fast reduction:
1. Multiply two 256-bit numbers to get a 512-bit result
2. Reduce from 512 to 320 bits by multiplying the high part by 0x1000003D1
3. Reduce from 320 to 256 bits with one final multiplication
4. Final conditional subtraction if result >= p

This approach is 10-100x faster than naive methods and matches the optimization used in the AVX2/AVX512 CPU implementations.

### 2. Montgomery Multiplication (Optional)
Montgomery multiplication offers an alternative approach using Montgomery space (aR mod N):
- Eliminates expensive modular reduction operations
- Works in "Montgomery space" where R = 2^256
- Uses CIOS (Coarsely Integrated Operand Scanning) algorithm
- Can be enabled with `make USE_MONTGOMERY=1`

See [MONTGOMERY_IMPLEMENTATION.md](MONTGOMERY_IMPLEMENTATION.md) for details.

**Note:** The default fast secp256k1 method is recommended as it's typically faster for this specific prime. Montgomery is provided for comparison and may show benefits on certain GPU architectures.

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 7.0 or higher
  - Volta (V100, Titan V)
  - Turing (RTX 20xx, GTX 16xx)
  - Ampere (RTX 30xx, A100)
  - Ada Lovelace (RTX 40xx)
  - Hopper (H100)

### Software
- **CUDA Toolkit 12.x** (12.0 or later)
- **Linux**: GCC 9+ or LLVM/Clang 10+
- **Windows**: Visual Studio 2019+ or MinGW-w64
- **Optional**: OpenSSL (for cryptographically secure WIF generation)

## Installation

### Linux

```bash
# Clone the repository
git clone https://github.com/Dookoo2/Cyclone.git
cd Cyclone/Cyclone_cuda

# Build using Makefile
make

# Or with OpenSSL support (recommended for WIF)
make USE_OPENSSL=1

# Or compile directly
nvcc -O3 -arch=sm_70 -std=c++14 -o Cyclone_cuda Cyclone_cuda.cu -lcuda

# With OpenSSL
nvcc -O3 -arch=sm_70 -std=c++14 -DUSE_OPENSSL -o Cyclone_cuda Cyclone_cuda.cu -lcuda -lssl -lcrypto
```

### Windows

```bash
# Using Visual Studio Developer Command Prompt
nvcc -O3 -arch=sm_70 -std=c++14 -o Cyclone_cuda.exe Cyclone_cuda.cu

# Or using Makefile with MinGW
make
```

### Architecture Selection

The default build targets Compute Capability 7.0 (`sm_70`). To build for your specific GPU:

```bash
# RTX 30xx (Ampere)
make CUDA_ARCH=sm_80

# RTX 40xx (Ada Lovelace)
make CUDA_ARCH=sm_89

# Or directly with nvcc
nvcc -O3 -arch=sm_89 -std=c++14 -o Cyclone_cuda Cyclone_cuda.cu -lcuda
```

### Montgomery Multiplication Option

To test Montgomery multiplication as an alternative:

```bash
# Build with Montgomery multiplication
make USE_MONTGOMERY=1

# Or with specific architecture
make USE_MONTGOMERY=1 CUDA_ARCH=sm_89

# Build and run test program
make test
./test_montgomery_cuda
```

The default (USE_MONTGOMERY=0) uses the fast secp256k1-specific method which is typically faster. Montgomery is provided for comparison and may show benefits on certain architectures. See [MONTGOMERY_IMPLEMENTATION.md](MONTGOMERY_IMPLEMENTATION.md) for details.

## Usage

The command-line interface is compatible with the CPU version:

### Basic Usage

```bash
./Cyclone_cuda -a <target_address> -r <start:end>
```

### Options

- `-a <address>` - Target Bitcoin P2PKH address (required)
- `-r <start:end>` - Search range in hexadecimal (required)
- `-p <length>` - Partial match length in hex digits (optional)
- `-j <size>` - Jump size after partial match (optional, requires `-p`)
- `--random` - Use random search mode instead of sequential (optional)
- `--gpus <n>` - Number of GPUs to use (default: all available)
- `-h, --help` - Display help message

### Examples

#### Sequential Search
```bash
./Cyclone_cuda -a 128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k -r 875:6FAC3875
```

#### Random Search
```bash
./Cyclone_cuda -a 128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k -r 1:FFFFFFFF --random
```

#### Probabilistic Search with Partial Matching
```bash
./Cyclone_cuda -a 128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k -r FAC875:6FAC3875 -p 6 -j 10000000
```

#### Multi-GPU Search
```bash
./Cyclone_cuda -a 128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k -r 1:FFFFFFFFFFFFFFFF --gpus 4
```

## Performance

Performance varies based on GPU model:

| GPU Model | Compute Capability | Estimated Speed |
|-----------|-------------------|-----------------|
| RTX 4090 | 8.9 | ~5000 Mkeys/s |
| RTX 3090 | 8.6 | ~3000 Mkeys/s |
| RTX 3080 | 8.6 | ~2500 Mkeys/s |
| A100 | 8.0 | ~3500 Mkeys/s |
| V100 | 7.0 | ~2000 Mkeys/s |

*Note: Actual performance depends on key range, search mode, and system configuration.*

## Implementation Details

### CUDA Kernels

The implementation includes optimized CUDA kernels for:

1. **256-bit Integer Arithmetic** (`cuda_uint256.cuh`)
   - Addition, subtraction, comparison
   - Modular arithmetic operations

2. **SECP256k1 Elliptic Curve** (`cuda_secp256k1.cuh`)
   - Point addition in Jacobian coordinates
   - Point doubling
   - Scalar multiplication (double-and-add algorithm)
   - Affine coordinate conversion

3. **Cryptographic Hashing** (`cuda_hash.cuh`)
   - SHA256 implementation
   - RIPEMD160 implementation
   - Hash160 (SHA256 + RIPEMD160)

### Thread Independence

Each GPU thread operates independently:
- Separate key range assignment
- Independent jump calculations for partial matches
- No inter-thread synchronization required during search

### Multi-GPU Strategy

When using multiple GPUs:
- Work is distributed evenly across devices
- Each GPU processes a non-overlapping portion of the range
- Results are collected asynchronously

## Troubleshooting

### CUDA Not Found
```bash
# Linux: Add CUDA to PATH
export PATH=/usr/local/cuda-12/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH

# Check CUDA installation
nvcc --version
```

### Compilation Errors

**Error: `unsupported GNU version`**
- Use a compatible GCC version (9-11 for CUDA 12.x)
- Or specify compiler: `nvcc -ccbin gcc-10 ...`

**Error: `No kernel image is available`**
- Recompile with correct architecture flag for your GPU
- Check GPU compute capability: `nvidia-smi --query-gpu=compute_cap --format=csv`

### Runtime Errors

**Error: `no CUDA-capable device is detected`**
- Verify GPU is installed and recognized: `nvidia-smi`
- Update NVIDIA drivers to latest version

**Error: `out of memory`**
- Reduce `BLOCKS_PER_GRID` or `KEYS_PER_THREAD` in source code
- Use fewer GPUs or smaller batches

## Security Notice

This software is developed for solving cryptographic puzzles and educational purposes. Use of this software for any illegal activities is strictly prohibited. The author is not responsible for any misuse.

## License

GNU General Public License v3.0 - see LICENSE file

## Credits

Based on the CPU implementation by Dookoo2/Cyclone
SECP256k1 mathematics adapted from JeanLucPons/VanitySearch

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Support

BTC: bc1qtq4y9l9ajeyxq05ynq09z8p52xdmk4hqky9c8n
