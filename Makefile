CXX := g++
NVCC := nvcc
CUDA_ARCH ?= sm_75

CXXFLAGS := -std=c++20 -O3
NVCCFLAGS := -std=c++17 -arch=$(CUDA_ARCH) -O3
LDFLAGS := -lcudart

TARGET := cyclone_gpu

all: $(TARGET)

$(TARGET): main.o p2pkh_decoder.o
	$(CXX) main.o p2pkh_decoder.o -o $(TARGET) $(LDFLAGS)

p2pkh_decoder.o: p2pkh_decoder.cpp p2pkh_decoder.h
	$(CXX) $(CXXFLAGS) -c p2pkh_decoder.cpp

main.o: main.cu p2pkh_decoder.h
	$(NVCC) $(NVCCFLAGS) -c main.cu

clean:
	rm -f *.o $(TARGET)
