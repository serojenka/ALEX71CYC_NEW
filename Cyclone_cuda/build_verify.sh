#!/bin/bash

# Build verification script for Cyclone CUDA
# This script checks dependencies and attempts to build the project

echo "============================================"
echo "Cyclone CUDA Build Verification"
echo "============================================"
echo ""

# Check for CUDA
echo "[1/5] Checking for CUDA Toolkit..."
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    echo "✓ CUDA found: nvcc version $NVCC_VERSION"
    
    # Check if CUDA 12 or later
    MAJOR_VERSION=$(echo $NVCC_VERSION | cut -d. -f1)
    if [ "$MAJOR_VERSION" -ge 12 ]; then
        echo "✓ CUDA version is 12 or later"
    else
        echo "⚠ Warning: CUDA $NVCC_VERSION detected. CUDA 12+ recommended."
    fi
else
    echo "✗ CUDA not found. Please install CUDA Toolkit 12.x"
    echo "  Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo ""

# Check for NVIDIA GPU
echo "[2/5] Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found"
    GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)
    echo "  GPU: $GPU_INFO"
    
    # Extract compute capability
    COMPUTE_CAP=$(echo $GPU_INFO | awk -F',' '{print $2}' | tr -d ' ')
    COMPUTE_MAJ=$(echo $COMPUTE_CAP | cut -d. -f1)
    
    if [ "$COMPUTE_MAJ" -ge 7 ]; then
        echo "✓ Compute capability $COMPUTE_CAP is supported (7.0+)"
    else
        echo "⚠ Warning: Compute capability $COMPUTE_CAP. 7.0+ recommended."
    fi
else
    echo "⚠ Warning: nvidia-smi not found. GPU may not be available."
fi

echo ""

# Check for GCC/Compiler
echo "[3/5] Checking for compiler..."
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -1)
    echo "✓ G++ found: $GCC_VERSION"
else
    echo "⚠ Warning: g++ not found. Install build tools."
fi

echo ""

# Check for make
echo "[4/5] Checking for make..."
if command -v make &> /dev/null; then
    echo "✓ make found"
else
    echo "⚠ Warning: make not found. Will try direct nvcc compilation."
fi

echo ""

# Attempt build
echo "[5/5] Attempting to build..."
if [ -f "Makefile" ]; then
    echo "Building with Makefile..."
    make clean 2>/dev/null
    if make; then
        echo "✓ Build successful!"
        echo ""
        echo "Executable created: ./Cyclone_cuda"
        echo ""
        echo "To test, run:"
        echo "  ./Cyclone_cuda --help"
    else
        echo "✗ Build failed with Makefile"
        echo "Trying direct nvcc compilation..."
        
        if nvcc -O3 -arch=sm_70 -std=c++14 -o Cyclone_cuda Cyclone_cuda.cu -lcuda 2>&1; then
            echo "✓ Direct compilation successful!"
        else
            echo "✗ Direct compilation failed"
            echo ""
            echo "Common issues:"
            echo "  1. Check CUDA Toolkit installation"
            echo "  2. Verify GPU drivers are up to date"
            echo "  3. Check compute capability compatibility"
            exit 1
        fi
    fi
else
    echo "Makefile not found. Trying direct nvcc compilation..."
    if nvcc -O3 -arch=sm_70 -std=c++14 -o Cyclone_cuda Cyclone_cuda.cu -lcuda; then
        echo "✓ Build successful!"
    else
        echo "✗ Build failed"
        exit 1
    fi
fi

echo ""
echo "============================================"
echo "Build verification complete!"
echo "============================================"
