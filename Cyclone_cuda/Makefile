# Makefile for Cyclone CUDA Bitcoin Puzzle Solver
# Supports Linux and Windows (MinGW) with CUDA 12

# Compiler settings
NVCC = nvcc
CUDA_ARCH = sm_70
CUDA_FLAGS = -O3 -arch=$(CUDA_ARCH) -std=c++14
TARGET = Cyclone_cuda

# Optional OpenSSL support (for proper WIF generation)
# Uncomment to enable:
# USE_OPENSSL = 1

ifdef USE_OPENSSL
    CUDA_FLAGS += -DUSE_OPENSSL
    SSL_LIBS = -lssl -lcrypto
else
    SSL_LIBS =
endif

# Source files
SOURCES = Cyclone_cuda.cu
HEADERS = cuda_uint256.cuh cuda_secp256k1.cuh cuda_hash.cuh

# Platform detection
ifeq ($(OS),Windows_NT)
    # Windows build
    TARGET_EXT = .exe
    RM = del /Q
    CUDA_LIBS = -lcuda $(SSL_LIBS)
else
    # Linux build
    TARGET_EXT =
    RM = rm -f
    CUDA_LIBS = -lcuda $(SSL_LIBS)
endif

# Default target
all: $(TARGET)$(TARGET_EXT)

# Build target
$(TARGET)$(TARGET_EXT): $(SOURCES) $(HEADERS)
	@echo "Building Cyclone CUDA for CUDA 12..."
ifdef USE_OPENSSL
	@echo "Using OpenSSL for cryptographic functions"
endif
	$(NVCC) $(CUDA_FLAGS) -o $(TARGET)$(TARGET_EXT) $(SOURCES) $(CUDA_LIBS)
	@echo "Build complete: $(TARGET)$(TARGET_EXT)"

# Clean build artifacts
clean:
	$(RM) $(TARGET)$(TARGET_EXT) *.o

# Help target
help:
	@echo "Cyclone CUDA Build System"
	@echo "========================="
	@echo ""
	@echo "Targets:"
	@echo "  make         - Build Cyclone CUDA"
	@echo "  make clean   - Remove build artifacts"
	@echo "  make help    - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - CUDA 12.x Toolkit"
	@echo "  - NVIDIA GPU with Compute Capability 7.0+"
	@echo "  - Linux: GCC 9+ or LLVM/Clang 10+"
	@echo "  - Windows: Visual Studio 2019+ or MinGW-w64"
	@echo ""
	@echo "Manual compilation:"
	@echo "  Linux:"
	@echo "    nvcc -O3 -arch=sm_70 -std=c++14 -o Cyclone_cuda Cyclone_cuda.cu -lcuda"
	@echo ""
	@echo "  Windows (Visual Studio):"
	@echo "    nvcc -O3 -arch=sm_70 -std=c++14 -o Cyclone_cuda.exe Cyclone_cuda.cu"
	@echo ""
	@echo "Architecture notes:"
	@echo "  sm_70 = Volta (Tesla V100, Titan V)"
	@echo "  sm_75 = Turing (RTX 20xx, GTX 16xx)"
	@echo "  sm_80 = Ampere (RTX 30xx, A100)"
	@echo "  sm_86 = Ampere (RTX 30xx mobile)"
	@echo "  sm_89 = Ada Lovelace (RTX 40xx)"
	@echo ""
	@echo "To build for a different architecture, use:"
	@echo "  make CUDA_ARCH=sm_80"

.PHONY: all clean help
