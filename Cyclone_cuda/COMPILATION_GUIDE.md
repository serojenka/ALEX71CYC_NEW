# Cyclone CUDA - Compilation Guide

This guide provides detailed instructions for compiling Cyclone CUDA on various platforms.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Linux Compilation](#linux-compilation)
- [Windows Compilation](#windows-compilation)
- [Architecture Selection](#architecture-selection)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

---

## Prerequisites

### Required Software

1. **CUDA Toolkit 12.x** (12.0 or later)
   - Download: https://developer.nvidia.com/cuda-downloads
   - Verify: `nvcc --version`

2. **NVIDIA GPU** with Compute Capability 7.0 or higher
   - Check: `nvidia-smi --query-gpu=compute_cap --format=csv`
   - Supported: Volta (7.0), Turing (7.5), Ampere (8.0/8.6), Ada Lovelace (8.9), Hopper (9.0)

3. **C++ Compiler**
   - Linux: GCC 9-11 or Clang 10+
   - Windows: Visual Studio 2019/2022 or MinGW-w64

### System Requirements

- **GPU Memory**: 2GB+ recommended
- **System RAM**: 4GB+ minimum
- **Disk Space**: 500MB for CUDA Toolkit + build files
- **OS**: 
  - Linux: Ubuntu 18.04+, CentOS 7+, or compatible
  - Windows: Windows 10/11 64-bit

---

## Linux Compilation

### Method 1: Using Makefile (Recommended)

```bash
cd Cyclone_cuda
make
```

The Makefile automatically detects your platform and uses appropriate settings.

### Method 2: Using Build Script

```bash
cd Cyclone_cuda
chmod +x build_verify.sh
./build_verify.sh
```

This script checks dependencies and attempts compilation.

### Method 3: Manual Compilation

```bash
cd Cyclone_cuda

# Basic compilation (Volta/Turing/Ampere)
nvcc -O3 -arch=sm_70 -std=c++14 -o Cyclone_cuda Cyclone_cuda.cu -lcuda

# For RTX 30xx (Ampere)
nvcc -O3 -arch=sm_80 -std=c++14 -o Cyclone_cuda Cyclone_cuda.cu -lcuda

# For RTX 40xx (Ada Lovelace)
nvcc -O3 -arch=sm_89 -std=c++14 -o Cyclone_cuda Cyclone_cuda.cu -lcuda

# With specific GCC version (if needed)
nvcc -O3 -arch=sm_70 -std=c++14 -ccbin gcc-10 -o Cyclone_cuda Cyclone_cuda.cu -lcuda
```

### Adding CUDA to PATH (if needed)

```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH=/usr/local/cuda-12/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH

# Reload
source ~/.bashrc
```

---

## Windows Compilation

### Method 1: Using Visual Studio Developer Command Prompt (Recommended)

1. Open "x64 Native Tools Command Prompt for VS 2022" (or VS 2019)
2. Navigate to Cyclone_cuda directory:
   ```batch
   cd path\to\Cyclone_cuda
   ```
3. Run build script:
   ```batch
   build_windows.bat
   ```

### Method 2: Manual Compilation with NVCC

```batch
cd Cyclone_cuda
nvcc -O3 -arch=sm_70 -std=c++14 -o Cyclone_cuda.exe Cyclone_cuda.cu
```

### Method 3: Using Makefile with MinGW

```batch
cd Cyclone_cuda
mingw32-make
```

### Setting Up CUDA PATH (Windows)

If `nvcc` is not found:

1. Add CUDA to PATH:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
   ```
2. Add to System Variables:
   - Variable: `CUDA_PATH`
   - Value: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`

---

## Architecture Selection

Choose the correct architecture flag for your GPU:

| GPU Series | Architecture | Compute Capability | Flag |
|------------|--------------|-------------------|------|
| Tesla V100, Titan V | Volta | 7.0 | `-arch=sm_70` |
| RTX 20xx, GTX 16xx | Turing | 7.5 | `-arch=sm_75` |
| RTX 30xx, A100 | Ampere | 8.0 | `-arch=sm_80` |
| RTX 30xx Mobile | Ampere | 8.6 | `-arch=sm_86` |
| RTX 40xx | Ada Lovelace | 8.9 | `-arch=sm_89` |
| H100 | Hopper | 9.0 | `-arch=sm_90` |

**To check your GPU's compute capability:**

```bash
# Linux
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Or check online
# https://developer.nvidia.com/cuda-gpus
```

**Build for multiple architectures** (increases binary size):

```bash
nvcc -O3 -arch=sm_70 -arch=sm_75 -arch=sm_80 -arch=sm_86 -arch=sm_89 \
     -std=c++14 -o Cyclone_cuda Cyclone_cuda.cu -lcuda
```

---

## Troubleshooting

### Error: "nvcc: command not found"

**Solution:**
- Ensure CUDA Toolkit is installed
- Add CUDA bin directory to PATH
- Restart terminal/command prompt after installation

### Error: "unsupported GNU version"

**Cause:** GCC version incompatible with CUDA 12

**Solution:**
```bash
# Use compatible GCC version
nvcc -ccbin gcc-10 -O3 -arch=sm_70 -std=c++14 -o Cyclone_cuda Cyclone_cuda.cu -lcuda

# Or install compatible GCC
sudo apt install gcc-10 g++-10
```

### Error: "no kernel image is available for execution"

**Cause:** Binary compiled for wrong architecture

**Solution:**
- Check GPU compute capability
- Recompile with correct `-arch=sm_XX` flag
- Or compile for multiple architectures

### Error: "out of memory"

**Cause:** GPU has insufficient memory

**Solution:**
- Reduce `BLOCKS_PER_GRID` or `KEYS_PER_THREAD` in source
- Close other GPU applications
- Use a GPU with more memory

### Error: "cannot find -lcuda"

**Cause:** CUDA libraries not in linker path

**Solution (Linux):**
```bash
# Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH

# Or create symlink
sudo ldconfig /usr/local/cuda-12/lib64
```

**Solution (Windows):**
- Ensure CUDA is in PATH
- Use Visual Studio Developer Command Prompt

### Warning: "implicit declaration of function"

**Cause:** Missing includes or function declarations

**Solution:**
- Verify all `.cuh` and `.h` files are present
- Check file permissions
- Ensure files are in same directory

---

## Verification

### Test Compilation

After successful build, verify:

```bash
# Check executable exists
ls -lh Cyclone_cuda       # Linux
dir Cyclone_cuda.exe      # Windows

# Test help output
./Cyclone_cuda --help     # Linux
Cyclone_cuda.exe --help   # Windows

# Check GPU detection
nvidia-smi
```

### Quick Test Run

```bash
# Simple test (replace with valid address and range)
./Cyclone_cuda -a 1BitcoinAddress... -r 1:FFFF
```

### Performance Check

For optimal performance:
- GPU temperature should stay below 85°C
- GPU utilization should be near 100%
- Monitor with: `nvidia-smi -l 1`

---

## Advanced Options

### Optimization Flags

```bash
# Maximum optimization
nvcc -O3 -use_fast_math -arch=sm_80 -std=c++14 \
     -o Cyclone_cuda Cyclone_cuda.cu -lcuda

# Debug build (for development)
nvcc -g -G -arch=sm_80 -std=c++14 \
     -o Cyclone_cuda_debug Cyclone_cuda.cu -lcuda

# With warnings
nvcc -O3 -arch=sm_80 -std=c++14 -Xcompiler -Wall \
     -o Cyclone_cuda Cyclone_cuda.cu -lcuda
```

### Cross-Compilation (Linux to Windows)

```bash
# Using MinGW cross-compiler
x86_64-w64-mingw32-nvcc -O3 -arch=sm_70 -std=c++14 \
     -o Cyclone_cuda.exe Cyclone_cuda.cu
```

---

## Build System Integration

### CMake (Optional)

Create `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18)
project(CycloneCUDA CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)

add_executable(Cyclone_cuda Cyclone_cuda.cu)
target_link_libraries(Cyclone_cuda cuda)
```

Build:
```bash
mkdir build && cd build
cmake ..
make
```

---

## Additional Resources

- CUDA Documentation: https://docs.nvidia.com/cuda/
- CUDA Toolkit Download: https://developer.nvidia.com/cuda-downloads
- GPU Compute Capabilities: https://developer.nvidia.com/cuda-gpus
- NVIDIA Developer Forums: https://forums.developer.nvidia.com/

---

For issues or questions, please open an issue on GitHub.
