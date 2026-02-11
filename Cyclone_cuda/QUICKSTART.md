# Cyclone CUDA - Quick Start Guide

Get started with Cyclone CUDA in 5 minutes!

## Prerequisites Check

Before you begin, verify you have:

```bash
# Check CUDA installation
nvcc --version
# Should show: release 12.x

# Check GPU
nvidia-smi
# Should show your NVIDIA GPU

# Check compiler (Linux)
g++ --version
# Should show GCC 9 or later
```

If any command fails, see [COMPILATION_GUIDE.md](COMPILATION_GUIDE.md) for installation instructions.

## Quick Build (3 Steps)

### Linux

```bash
# 1. Navigate to directory
cd Cyclone_cuda

# 2. Build
make

# 3. Test
./Cyclone_cuda --help
```

### Windows

```batch
REM 1. Open VS Developer Command Prompt
REM 2. Navigate to directory
cd Cyclone_cuda

REM 3. Build
build_windows.bat

REM 4. Test
Cyclone_cuda.exe --help
```

## Your First Search

Try finding a known key (for testing):

```bash
# Test puzzle #10 (known key: 0x3FA)
./Cyclone_cuda -a 128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k -r 1:1000

# Expected output:
# ================== FOUND MATCH! ==================
# Private Key   : 00000000000000000000000000000000000000000000000000000000000003FA
# ...
```

## Common Use Cases

### 1. Sequential Search (Default)

Search a range sequentially:

```bash
./Cyclone_cuda -a <your_address> -r 1:FFFFFFFF
```

### 2. Random Search

For large ranges, random search is faster:

```bash
./Cyclone_cuda -a <your_address> -r 1:FFFFFFFFFFFFFFFF --random
```

### 3. Partial Match with Jumps

Jump after finding partial hash matches:

```bash
./Cyclone_cuda -a <your_address> -r 1:FFFFFFFF -p 6 -j 1000000
```

- `-p 6` = Match first 6 hex digits of hash160
- `-j 1000000` = Jump 1 million keys forward after match

### 4. Multi-GPU

Use all available GPUs:

```bash
./Cyclone_cuda -a <your_address> -r 1:FFFFFFFF --gpus 4
```

## Performance Tips

### 1. Check GPU Utilization

While Cyclone is running:

```bash
# Monitor GPU in real-time
watch -n 1 nvidia-smi
```

Look for:
- GPU utilization near 100%
- Memory usage stable
- Temperature < 85°C

### 2. Optimize for Your GPU

Build for your specific architecture:

```bash
# RTX 30xx
make CUDA_ARCH=sm_80

# RTX 40xx
make CUDA_ARCH=sm_89
```

### 3. Larger Ranges = Better

GPU performance improves with larger ranges due to:
- Better parallelization
- Amortized kernel launch overhead
- Higher GPU occupancy

## Troubleshooting

### Build Fails

```bash
# Check CUDA path
echo $PATH | grep cuda

# If not found, add it:
export PATH=/usr/local/cuda-12/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH
```

### "No CUDA device found"

```bash
# Check GPU is detected
nvidia-smi

# If not shown, update drivers:
# Ubuntu/Debian:
sudo apt update
sudo apt install nvidia-driver-535

# Or download from: https://www.nvidia.com/drivers
```

### Very Slow Performance

Common causes:
1. **Wrong architecture**: Rebuild with correct -arch flag
2. **Thermal throttling**: Check GPU temp with nvidia-smi
3. **Other GPU processes**: Close them and try again
4. **Too small range**: Use larger ranges for better performance

## Example Workflow

Complete example for puzzle solving:

```bash
# 1. Build with optimizations
make clean
make CUDA_ARCH=sm_89

# 2. Start search with multiple GPUs
./Cyclone_cuda -a 1BitcoinAddress... -r 800000000:900000000 --gpus 2

# 3. Monitor progress
# (Output updates every few seconds)

# 4. If found:
# Private key, public key, and WIF are displayed
# Results also saved to found_keys.txt
```

## Next Steps

Once you're comfortable with basics:

1. **Read full documentation**:
   - [README.md](README.md) - Detailed features
   - [COMPILATION_GUIDE.md](COMPILATION_GUIDE.md) - Advanced build options
   - [FEATURES.md](FEATURES.md) - Complete feature list

2. **Optimize for your hardware**:
   - Try different thread/block configurations
   - Experiment with random vs sequential
   - Test partial match with various jump sizes

3. **Scale up**:
   - Use multiple GPUs in parallel
   - Divide work across machines
   - Run 24/7 for large puzzles

## Safety Checklist

Before running on real puzzles:

- [ ] Tested on small ranges first
- [ ] Verified GPU doesn't overheat
- [ ] Have proper cooling/ventilation
- [ ] Monitoring GPU temperature
- [ ] Backed up any important work
- [ ] Understand electricity costs
- [ ] Legal to use in your jurisdiction

## Resources

- **NVIDIA CUDA**: https://developer.nvidia.com/cuda-toolkit
- **GPU List**: https://developer.nvidia.com/cuda-gpus
- **Bitcoin Puzzles**: Research online for current unsolved puzzles
- **Community**: GitHub Issues for questions

## Performance Expectations

Rough estimates for sequential search:

| Range Size | RTX 4090 | RTX 3080 | Time |
|------------|----------|----------|------|
| 2^20 (1M) | 5000 Mkeys/s | 2500 Mkeys/s | <1 sec |
| 2^30 (1B) | 5000 Mkeys/s | 2500 Mkeys/s | ~3-6 min |
| 2^40 (1T) | 5000 Mkeys/s | 2500 Mkeys/s | ~3-6 hours |
| 2^50 (1P) | 5000 Mkeys/s | 2500 Mkeys/s | ~5-11 days |

*Note: Random search changes these estimates significantly*

## Getting Help

If you encounter issues:

1. Check this guide
2. Read [COMPILATION_GUIDE.md](COMPILATION_GUIDE.md)
3. Review [FEATURES.md](FEATURES.md)
4. Search existing GitHub Issues
5. Open a new Issue with:
   - Your GPU model
   - CUDA version
   - Error messages
   - Build command used

---

**Happy Hunting!** 🎯
