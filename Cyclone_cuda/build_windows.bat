@echo off
REM Build script for Cyclone CUDA on Windows
REM Requires CUDA Toolkit 12.x and Visual Studio 2019+

echo ============================================
echo Cyclone CUDA Windows Build Script
echo ============================================
echo.

REM Check for nvcc
where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: nvcc not found!
    echo Please install CUDA Toolkit 12.x and add it to PATH
    echo Download from: https://developer.nvidia.com/cuda-downloads
    exit /b 1
)

echo [1/3] CUDA Toolkit found
nvcc --version | findstr "release"
echo.

REM Clean previous build
if exist Cyclone_cuda.exe (
    echo [2/3] Cleaning previous build...
    del Cyclone_cuda.exe
)

REM Build
echo [3/3] Building Cyclone CUDA...
nvcc -O3 -arch=sm_70 -std=c++14 -o Cyclone_cuda.exe Cyclone_cuda.cu -lcuda

if %errorlevel% equ 0 (
    echo.
    echo ============================================
    echo Build successful!
    echo ============================================
    echo.
    echo Executable: Cyclone_cuda.exe
    echo.
    echo To test, run:
    echo   Cyclone_cuda.exe --help
    echo.
) else (
    echo.
    echo ============================================
    echo Build failed!
    echo ============================================
    echo.
    echo Common issues:
    echo   1. Ensure Visual Studio 2019+ is installed
    echo   2. Run from Visual Studio Developer Command Prompt
    echo   3. Check CUDA Toolkit installation
    echo   4. Verify GPU drivers are up to date
    echo.
    exit /b 1
)
