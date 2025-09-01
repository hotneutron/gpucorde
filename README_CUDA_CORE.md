# CUDA Core Analytical Model

This directory contains the CUDA Core analytical model implementation following the Concorde-CPU compositional approach for GPU performance modeling.

## Quick Start

```bash
# 1. Build the NVBit instrumentation tool
cd src/cuda_core_trace
make clean && make

# 2. Compile example kernel  
nvcc -arch=sm_75 -O2 examples/cuda_kernels/simple_gemm.cu -o simple_gemm

# 3. Collect real GPU metrics
export LD_PRELOAD=$PWD/src/cuda_core_trace/build/simple_counter_v2.so
./simple_gemm

# 4. Analyze metrics with analytical model
python analyze_real_metrics.py
```

## Project Structure

```
├── models/cuda_core/           # Model specifications
│   └── analytical_model_card.md
├── src/
│   ├── analytical_models/      # Python analytical model
│   │   └── cuda_core_model.py
│   └── cuda_core_trace/        # NVBit instrumentation tool
│       ├── simple_counter_v2.cu
│       ├── inject_v2.cu
│       └── Makefile
├── examples/cuda_kernels/      # Test CUDA kernels
│   └── simple_gemm.cu
├── doc/                        # Documentation
│   ├── cuda_core_model_documentation.md
│   ├── model_alignment_verification.md
│   └── cuda_core_quickstart.md
└── analyze_real_metrics.py     # Analysis script
```

## Key Formula

The CUDA Core bound in FLOPs/cycle:

```
Φ = θ × C_fp32 × (1 + m_FMA) × f_fp32
```

Where:
- **θ**: Active thread fraction (warp occupancy)
- **C_fp32**: Hardware FP32 lanes per SM
- **m_FMA**: FMA instruction ratio
- **f_fp32**: FP32 instruction density

## Documentation

- [Full Documentation](doc/cuda_core_model_documentation.md) - Complete guide
- [Quick Start Guide](doc/cuda_core_quickstart.md) - Get started quickly
- [Model Alignment](doc/model_alignment_verification.md) - Theory vs implementation
- [Model Card](models/cuda_core/analytical_model_card.md) - Theoretical specification

## Key Features

✅ **Real GPU metrics** via NVBit binary instrumentation  
✅ **Concorde-CPU approach** adapted for GPU CUDA cores  
✅ **Hardware-aware** modeling (configurable for different GPUs)  
✅ **Optimization insights** from performance bounds  
✅ **Compositional** - integrates with other GPU component models

✅ Model Alignment Verified:

The implementation correctly follows the analytical model card:
- Formula: Φ = θ × C_fp32 × (1 + m_FMA) × f_fp32 ✅
- Uses real GPU data from NVBit (not simulated) ✅
- Properly handles thread divergence, FMA counting, and hardware parameters ✅

📋 How to Use:

# 1. Build
cd src/cuda_core_trace && make

# 2. Run
LD_PRELOAD=$PWD/build/simple_counter_v2.so ../../examples/cuda_kernels/simple_gemm

# 3. Analyze
cd ../.. && python analyze_real_metrics.py

The project now has a clean structure with only essential files for the CUDA core analytical model, following the Concorde-CPU
compositional approach with real GPU metrics.