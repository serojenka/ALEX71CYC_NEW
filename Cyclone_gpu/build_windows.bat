@echo off
setlocal ENABLEEXTENSIONS

REM Cyclone_gpu Windows x64 build script
REM Usage:
REM   build_windows.bat [sm_70|sm_75|sm_80|sm_86|sm_89] [BATCHES_PER_BLOCK]

set "CUDA_ARCH=%~1"
if "%CUDA_ARCH%"=="" set "CUDA_ARCH=sm_86"

set "BATCHES_PER_BLOCK=%~2"
if "%BATCHES_PER_BLOCK%"=="" set "BATCHES_PER_BLOCK=8"

echo ============================================
echo Cyclone_gpu Windows Build
echo ============================================
echo CUDA_ARCH=%CUDA_ARCH%
echo BATCHES_PER_BLOCK=%BATCHES_PER_BLOCK%
echo.

where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: nvcc not found in PATH.
    echo Install CUDA Toolkit and run from "x64 Native Tools Command Prompt".
    exit /b 1
)

if exist Cyclone_gpu.exe (
    del /Q Cyclone_gpu.exe
)

echo Building Cyclone_gpu.exe ...
nvcc -O3 -arch=%CUDA_ARCH% -std=c++14 -DBATCHES_PER_BLOCK=%BATCHES_PER_BLOCK% --expt-relaxed-constexpr -o Cyclone_gpu.exe main.cu p2pkh_decoder.cpp

if %errorlevel% neq 0 (
    echo.
    echo Build failed.
    echo Check: CUDA install, Visual Studio C++ tools, and GPU architecture flag.
    exit /b 1
)

echo.
echo Build successful: Cyclone_gpu.exe
echo Example run:
echo   Cyclone_gpu.exe -a 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH -r 1:1000
exit /b 0
