# Makefile for Cyclone CUDA Bitcoin Puzzle Solver
# Supports Linux and Windows (MinGW) with CUDA 12

# Compiler settings
NVCC = nvcc
CUDA_ARCH = sm_70
CUDA_FLAGS = -O3 -arch=$(CUDA_ARCH) -std=c++14
TARGET = Cyclone_cuda
TEST_TARGET = test_montgomery_cuda

# Optional Montgomery multiplication (0 = fast secp256k1, 1 = Montgomery)
# Uncomment to enable Montgomery (default is fast secp256k1):
# USE_MONTGOMERY = 1

ifdef USE_MONTGOMERY
    CUDA_FLAGS += -DUSE_MONTGOMERY=$(USE_MONTGOMERY)
    $(info Building with Montgomery multiplication: $(USE_MONTGOMERY))
else
    $(info Building with fast secp256k1 multiplication (default))
endif

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

# Test target
test: $(TEST_TARGET)$(TARGET_EXT)
	@echo "Test binary built: $(TEST_TARGET)$(TARGET_EXT)"
	@echo "Run with: ./$(TEST_TARGET)$(TARGET_EXT)"

# Build main target
$(TARGET)$(TARGET_EXT): $(SOURCES) $(HEADERS)
	@echo "Building Cyclone CUDA for CUDA 12..."
ifdef USE_OPENSSL
	@echo "Using OpenSSL for cryptographic functions"
endif
	$(NVCC) $(CUDA_FLAGS) -o $(TARGET)$(TARGET_EXT) $(SOURCES) $(CUDA_LIBS)
	@echo "Build complete: $(TARGET)$(TARGET_EXT)"

# Build test target
$(TEST_TARGET)$(TARGET_EXT): test_montgomery_cuda.cu $(HEADERS)
	@echo "Building Montgomery multiplication test..."
	$(NVCC) $(CUDA_FLAGS) -o $(TEST_TARGET)$(TARGET_EXT) test_montgomery_cuda.cu $(CUDA_LIBS)
	@echo "Test build complete: $(TEST_TARGET)$(TARGET_EXT)"

# Clean build artifacts
clean:
	$(RM) $(TARGET)$(TARGET_EXT) $(TEST_TARGET)$(TARGET_EXT) *.o

# Help target
help:
	@echo "Cyclone CUDA Build System"
	@echo "========================="
	@echo ""
	@echo "Targets:"
	@echo "  make         - Build Cyclone CUDA"
	@echo "  make test    - Build Montgomery test program"
	@echo "  make clean   - Remove build artifacts"
	@echo "  make help    - Show this help message"
	@echo ""
	@echo "Options:"
	@echo "  USE_MONTGOMERY=1  - Enable Montgomery multiplication (default: 0)"
	@echo "  USE_OPENSSL=1     - Enable OpenSSL for WIF (default: 0)"
	@echo "  CUDA_ARCH=sm_XX   - Set target GPU architecture (default: sm_70)"
	@echo ""
	@echo "Examples:"
	@echo "  make                           - Build with fast secp256k1"
	@echo "  make USE_MONTGOMERY=1          - Build with Montgomery"
	@echo "  make CUDA_ARCH=sm_89           - Build for RTX 40xx"
	@echo "  make test                      - Build test program"
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

.PHONY: all test clean help
