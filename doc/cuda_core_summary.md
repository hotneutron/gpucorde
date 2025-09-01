# CUDA Core Analytical Model - Summary

## Overview

This implementation provides a **Concorde-CPU inspired analytical model for GPU CUDA cores**, using real GPU metrics collected via NVBit instrumentation.

## How It Works

### 1. **Data Collection** (NVBit Tool)
   - `src/cuda_core_trace/simple_counter_v2.cu` - Instruments CUDA kernels at runtime
   - `src/cuda_core_trace/inject_v2.cu` - GPU-side counting functions
   - Collects: instruction counts, FP32 operations, FMA usage, thread activity

### 2. **Analytical Model** (Python)
   - `src/analytical_models/cuda_core_model.py` - Implements theoretical bounds
   - `models/cuda_core/analytical_model_card.md` - Mathematical specification
   - Formula: `Φ = θ × C_fp32 × (1 + m_FMA) × f_fp32`

### 3. **Analysis** (Results Processing)
   - `analyze_real_metrics.py` - Processes metrics through model
   - Calculates performance bounds in FLOPs/cycle
   - Provides optimization recommendations

## Running the Model

```bash
# Step 1: Build NVBit tool
cd src/cuda_core_trace && make

# Step 2: Run with instrumentation
LD_PRELOAD=$PWD/build/simple_counter_v2.so ../../examples/cuda_kernels/simple_gemm

# Step 3: Analyze results
cd ../.. && python analyze_real_metrics.py
```

## Key Metrics Explained

### θ (Theta) - Warp Occupancy
- **What**: Fraction of threads active in warps (0-1)
- **Collected**: Via `__ballot_sync()` and `__popc()` in GPU code
- **Impact**: Directly scales performance (θ=0.5 means half performance)

### m_FMA - FMA Ratio
- **What**: Fraction of FP32 ops that are FMA instructions
- **Collected**: By detecting FFMA opcodes during instrumentation
- **Impact**: FMA = 2 FLOPs vs 1 FLOP for FADD/FMUL

### f_fp32 - FP32 Density  
- **What**: Fraction of all instructions that are FP32
- **Collected**: Ratio of FP32 to total instruction count
- **Impact**: Shows if kernel is compute-bound vs memory/control-bound

### C_fp32 - Hardware Constant
- **What**: FP32 CUDA core lanes per SM
- **Values**: A100=64, V100=64, RTX 3090=128
- **Usage**: Configured in `analyze_real_metrics.py`

## Example Results Interpretation

```
Input Metrics (from NVBit):
- Total instructions: 4,089,856
- FP32 instructions: 1,048,576  
- θ = 1.00 (perfect occupancy)
- m_FMA = 0.00 (no FMA usage)
- f_fp32 = 0.26 (26% FP32 density)

Analytical Model Output:
- Bound: 32.82 FLOPs/cycle
- Peak: 256 FLOPs/cycle
- Efficiency: 12.8%

Bottlenecks Identified:
1. No FMA usage (could 2x performance)
2. Low FP32 density (kernel is memory-bound)
```

## Model Validation

The implementation is **fully aligned** with the theoretical specification:
- ✅ Uses real GPU metrics (not simulated)
- ✅ Correctly implements formula from model card
- ✅ Properly counts FMA as 2 FLOPs
- ✅ Captures thread divergence effects
- ✅ Hardware-aware (configurable C_fp32)

See [model_alignment_verification.md](model_alignment_verification.md) for detailed proof.

## Optimization Workflow

1. **Run NVBit tool** → Collect metrics
2. **Analyze with model** → Identify bottlenecks
3. **Apply optimizations**:
   - Low θ → Reduce thread divergence
   - Low m_FMA → Enable FMA (`-ffp-contract=fast`)
   - Low f_fp32 → Increase compute intensity
4. **Re-measure** → Verify improvements

## Integration with Full GPU Model

This CUDA core model is one component in a compositional GPU model:

```
GPU Performance = min(
    CUDA_Core_Bound,
    Tensor_Core_Bound,
    Memory_Bound,
    SFU_Bound,
    ...
)
```

Each component can be modeled independently and combined to find the overall bottleneck.

## Files Summary

| File | Purpose |
|------|---------|
| `simple_counter_v2.cu` | NVBit tool for metric collection |
| `inject_v2.cu` | GPU-side instruction counting |
| `cuda_core_model.py` | Python analytical model |
| `analyze_real_metrics.py` | Combines metrics + model |
| `analytical_model_card.md` | Mathematical specification |

## Next Steps

- Add sliding window analysis for time-varying behavior
- Integrate with other GPU component models
- Automate optimization suggestions
- Add support for different GPU architectures