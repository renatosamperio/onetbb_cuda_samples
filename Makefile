CXX = g++ -std=gnu++20
NVCC = /usr/local/cuda/bin/nvcc

# used compile oneAPI library
ONEAPI_HOME=/opt/intel/oneapi/tbb/latest
ONEAPI_LIB=-L$(ONEAPI_HOME)/lib -ltbb -pthread 
ONEAPI_INC=-I$(ONEAPI_HOME)/include
ONEAPI_FLAGS=$(ONEAPI_LIB) $(ONEAPI_INC)

# used compile cuda library
CUDA_HOME=/usr/local/cuda
CUDA_LIB=-L$(CUDA_HOME)/lib64 -lcudart
CUDA_INC=-I$(CUDA_HOME)/include
CUDA_ARCH=-arch=compute_50 #all #all-major #compute_50
CUDA_FLAGS=$(CUDA_LIB) $(CUDA_INC)

# general compilation composed flags
CXXFLAGS=$(CUDA_FLAGS) $(ONEAPI_FLAGS)
LINKING_FLAGS=-Wl,-R$(ONEAPI_HOME)/lib

## Library source and object files
cpp_sources = $(wildcard *.cpp)
cpp_objs := $(patsubst %.cpp, %.o, $(cpp_sources))
cu_sources = $(wildcard *.cu)
cu_objs := $(patsubst %.cu, %.o, $(cu_sources))
all_objs = $(cu_objs) $(cpp_objs)

TARGET=main
## PHONY targets
.PHONY: all $(TARGET)

## Overwrite R targets prepared
all: main

%.o: %.cpp $(cpp_sources)
	$(CXX) $< -c $(ONEAPI_FLAGS)

%.o: %.cu $(cu_sources)
	$(NVCC) $< -c $(CUDA_FLAGS) $(CUDA_ARCH)

$(TARGET): $(all_objs)
	$(CXX) $(CXXFLAGS) $(LINKING_FLAGS) -o $(TARGET) $(all_objs)

clean:
	rm -f *.o main

# Performance command
perf: $(TARGET)
	time ./$(TARGET)