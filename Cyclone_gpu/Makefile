# Makefile for Cyclone GPU Bitcoin Puzzle Solver
# Optimized batch algorithm ported from Cyclone_avx2 to CUDA

NVCC        = nvcc
CUDA_ARCH   = sm_70
STD         = -std=c++14
OPT         = -O3
TARGET      = Cyclone_gpu

# Compute capability options:
#   sm_70 = Volta (V100, Titan V)
#   sm_75 = Turing (RTX 20xx, GTX 16xx)
#   sm_80 = Ampere (A100, RTX 30xx)
#   sm_86 = Ampere (RTX 3090 / mobile 30xx)
#   sm_89 = Ada Lovelace (RTX 40xx)

# Number of key-batches each block processes per kernel launch (default 8)
BATCHES_PER_BLOCK ?= 8

CUDA_FLAGS  = $(OPT) -arch=$(CUDA_ARCH) $(STD) \
              -DBATCHES_PER_BLOCK=$(BATCHES_PER_BLOCK) \
              --expt-relaxed-constexpr

SOURCES     = main.cu p2pkh_decoder.cpp
HEADERS     = gpu_uint256.cuh gpu_secp256k1.cuh gpu_hash.cuh p2pkh_decoder.h

ifeq ($(OS),Windows_NT)
  EXT  = .exe
  RM   = del /Q
  LIBS =
else
  EXT  =
  RM   = rm -f
  LIBS = -lcuda
endif

# Default target
all: $(TARGET)$(EXT)

$(TARGET)$(EXT): $(SOURCES) $(HEADERS)
	@echo "Building Cyclone GPU (arch=$(CUDA_ARCH), batches=$(BATCHES_PER_BLOCK)) ..."
	$(NVCC) $(CUDA_FLAGS) -o $@ $(SOURCES) $(LIBS)
	@echo "Build complete: $@"

# Convenience: build for multiple architectures with PTX fallback
multi-arch: $(SOURCES) $(HEADERS)
	$(NVCC) $(OPT) -arch=sm_70 -gencode=arch=compute_70,code=sm_70 \
	        -gencode=arch=compute_75,code=sm_75 \
	        -gencode=arch=compute_80,code=sm_80 \
	        -gencode=arch=compute_86,code=sm_86 \
	        -gencode=arch=compute_89,code=sm_89 \
	        $(STD) $(CUDA_FLAGS) \
	        -o $(TARGET)$(EXT) $(SOURCES) $(LIBS)

clean:
	$(RM) $(TARGET)$(EXT) *.o

help:
	@echo "Cyclone GPU Build System"
	@echo "========================"
	@echo ""
	@echo "Targets:"
	@echo "  make              - Build with default settings (sm_70)"
	@echo "  make multi-arch   - Build for sm_70/75/80/86/89"
	@echo "  make clean        - Remove build artifacts"
	@echo ""
	@echo "Options:"
	@echo "  CUDA_ARCH=sm_XX          - GPU architecture (default: sm_70)"
	@echo "  BATCHES_PER_BLOCK=N      - Batches per block per launch (default: 8)"
	@echo ""
	@echo "Examples:"
	@echo "  make CUDA_ARCH=sm_86                  - RTX 3090"
	@echo "  make CUDA_ARCH=sm_89                  - RTX 4090"
	@echo "  make CUDA_ARCH=sm_86 BATCHES_PER_BLOCK=16"

.PHONY: all multi-arch clean help
