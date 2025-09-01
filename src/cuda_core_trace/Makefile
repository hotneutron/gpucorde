# Makefile for CUDA Core NVBit Instrumentation Tool

# Directories
NVBIT_PATH ?= ../../3rdparty/nvbit_release/core
CUDA_PATH ?= /usr/local/cuda-12.5
BUILD_DIR := build

# Compiler
NVCC := nvcc -ccbin=$(CXX) -D_FORCE_INLINES

# Architecture
ARCH ?= all

# Includes and libs
INCLUDES := -I$(NVBIT_PATH) -I$(CUDA_PATH)/include
LIBS := -L$(NVBIT_PATH) -lnvbit -lelf
NVCC_PATH := -L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

# Build target
all: $(BUILD_DIR)/simple_counter_v2.so

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/simple_counter_v2.so: $(BUILD_DIR)/simple_counter_v2.o $(BUILD_DIR)/inject_v2.o | $(BUILD_DIR)
	@echo "Linking simple_counter_v2.so..."
	@$(NVCC) -arch=$(ARCH) -O3 $^ $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@
	@echo "Build complete: $@"

$(BUILD_DIR)/simple_counter_v2.o: simple_counter_v2.cu | $(BUILD_DIR)
	@echo "Compiling simple_counter_v2.cu..."
	@$(NVCC) -dc -c -std=c++14 $(INCLUDES) -Xptxas -cloning=no \
		-Xcompiler -Wall -arch=$(ARCH) -O3 -Xcompiler -fPIC $< -o $@

$(BUILD_DIR)/inject_v2.o: inject_v2.cu | $(BUILD_DIR)
	@echo "Compiling inject_v2.cu..."
	@$(NVCC) $(INCLUDES) -Xptxas -astoolspatch --keep-device-functions \
		-arch=$(ARCH) -Xcompiler -Wall -Xcompiler -fPIC -c -O3 $< -o $@

clean:
	@rm -rf $(BUILD_DIR)

.PHONY: all clean