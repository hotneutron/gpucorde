# GPUCorde - CUDA Core Analytical Model

A Concorde-CPU inspired analytical performance model for GPU CUDA cores, using real GPU metrics collected via NVBit instrumentation.

## Quick Start

### 1. Build NVBit Tool
```bash
cd src/cuda_core_trace
make -f Makefile.v2
```

### 2. Run on CUDA Kernel
```bash
# Compile test kernel
nvcc -arch=sm_75 -O2 examples/cuda_kernels/simple_gemm.cu -o simple_gemm

# Collect metrics
export LD_PRELOAD=$PWD/src/cuda_core_trace/build/simple_counter_v2.so
./simple_gemm
```

### 3. Analyze Results
```bash
python analyze_real_metrics.py
```

## Key Formula

**CUDA Core Bound**: `Φ = θ × C_fp32 × (1 + m_FMA) × f_fp32`

- **θ**: Warp occupancy (0-1)
- **C_fp32**: FP32 lanes per SM (hardware)
- **m_FMA**: FMA ratio (0-1)
- **f_fp32**: FP32 density (0-1)

## Real Example Output

```
📊 Real GPU Metrics:
  Total instructions: 4,089,856
  FP32 instructions:  1,048,576
  θ = 1.00, m_FMA = 0.00, f_fp32 = 0.26

🎯 Performance Bound:
  32.82 FLOPs/cycle (12.8% of peak)

💡 Optimization: Enable FMA for 2x speedup
```

## Files

- `src/cuda_core_trace/`: NVBit instrumentation tool
- `src/analytical_models/`: Python analytical model
- `analyze_real_metrics.py`: Analysis script
- `models/cuda_core/analytical_model_card.md`: Theory

See [CUDA_CORE_MODEL_DOCUMENTATION.md](CUDA_CORE_MODEL_DOCUMENTATION.md) for complete documentation.