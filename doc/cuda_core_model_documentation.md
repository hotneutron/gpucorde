# CUDA Core Analytical Model - Complete Documentation

## Overview

This project implements a **Concorde-CPU inspired analytical model for GPU CUDA cores**, providing performance bounds and optimization insights based on real GPU metrics collected via NVIDIA's NVBit binary instrumentation framework.

The model follows the compositional approach from Concorde-CPU, treating CUDA cores as one component in a larger GPU performance model. It calculates theoretical throughput bounds in FLOPs/cycle based on:
- **θ (theta)**: Active thread fraction (warp occupancy)
- **m_FMA**: FMA instruction ratio
- **f_fp32**: FP32 instruction density
- **C_fp32**: Hardware FP32 lanes per SM

## Project Structure

```
gpucorde/
├── models/
│   └── cuda_core/
│       └── analytical_model_card.md    # Theoretical model specification
├── src/
│   ├── analytical_models/
│   │   └── cuda_core_model.py         # Python implementation of analytical model
│   └── cuda_core_trace/
│       ├── simple_counter_v2.cu       # NVBit tool main file
│       ├── inject_v2.cu               # GPU-side injection functions
│       ├── Makefile.v2                # Build configuration
│       └── build/
│           └── simple_counter_v2.so   # Compiled NVBit tool
├── examples/
│   └── cuda_kernels/
│       └── simple_gemm.cu              # Example CUDA kernel for testing
├── analyze_real_metrics.py            # Analysis script for real GPU metrics
└── fp32_metrics.csv                   # Output metrics from NVBit tool
```

## Key Components

### 1. NVBit Instrumentation Tool (`src/cuda_core_trace/`)

The NVBit tool instruments CUDA kernels at runtime to collect:
- Total instructions executed
- FP32 instructions (FADD, FMUL, FFMA)
- FMA instructions specifically
- Active thread counts per instruction

**Key features:**
- Uses managed memory for GPU-host communication
- Warp-level instruction counting
- Predicate-aware (handles thread divergence)
- CSV output for analysis

### 2. Analytical Model (`src/analytical_models/cuda_core_model.py`)

Implements the theoretical model from the model card:

**Core Formula:**
```
Φ_SM,CUDA = θ × C_fp32 × (1 + m_FMA)
```

Where:
- **Φ**: CUDA core bound (FLOPs/cycle)
- **θ**: Average active thread fraction (0-1)
- **C_fp32**: FP32 lanes per SM (hardware constant)
- **m_FMA**: Fraction of FP32 ops that are FMAs

**Extended Formula:**
```
Φ_extended = u_issue × f_fp32 × θ × C_fp32 × (1 + m_FMA)
```

Additional factors:
- **u_issue**: Effective issue utilization
- **f_fp32**: FP32 instruction density in workload

### 3. Analysis Script (`analyze_real_metrics.py`)

Processes real GPU metrics and provides:
- Performance bound calculations
- Efficiency analysis vs theoretical peak
- Optimization recommendations
- Performance bottleneck identification

## How to Build and Run

### Prerequisites

1. **NVIDIA GPU** with CUDA support
2. **CUDA Toolkit** (12.5 or compatible)
3. **NVBit 1.7.5** (included as submodule)
4. **Python 3** with pandas, numpy, matplotlib

### Step 1: Build the NVBit Tool

```bash
cd src/cuda_core_trace
make -f Makefile.v2 clean
make -f Makefile.v2
```

This creates `build/simple_counter_v2.so`

### Step 2: Compile Test Kernel

```bash
nvcc -arch=sm_75 -O2 examples/cuda_kernels/simple_gemm.cu -o simple_gemm
```

Adjust `-arch` for your GPU (sm_75 for Turing, sm_80 for Ampere, etc.)

### Step 3: Collect GPU Metrics

```bash
export LD_PRELOAD=/absolute/path/to/gpucorde/src/cuda_core_trace/build/simple_counter_v2.so
./simple_gemm
```

This generates `fp32_metrics.csv` with real GPU metrics.

### Step 4: Analyze Results

```bash
python analyze_real_metrics.py
```

## Understanding the Results

### Example Output

```
📊 Raw GPU Metrics (from NVBit):
  Total instructions executed:  4,089,856
  FP32 instructions executed:   1,048,576
  FMA instructions executed:    0
  Average active threads:       32.00

📈 Calculated Metrics:
  θ (theta - warp occupancy):   1.0000
  m_FMA (FMA ratio):            0.0000
  f_fp32 (FP32 density):        0.2564

🎯 Analytical Model Results:
  CUDA Core Bound (Φ_basic):    128.00 FLOPs/cycle
  CUDA Core Bound (Φ_extended): 32.82 FLOPs/cycle
  Peak theoretical:              256 FLOPs/cycle
  Efficiency vs peak:            12.8%
```

### Metric Interpretations

1. **θ (Theta) - Warp Occupancy**
   - 1.0 = Perfect (all 32 threads active)
   - <0.8 = Thread divergence issues
   - Impacts: Directly scales performance

2. **m_FMA - FMA Ratio**
   - 0.0 = No FMA usage (separate FMUL/FADD)
   - 1.0 = All FP32 ops are FMAs
   - Impacts: FMA doubles throughput (2 FLOPs vs 1)

3. **f_fp32 - FP32 Density**
   - Fraction of instructions that are FP32
   - <0.3 = Likely memory or control bound
   - >0.7 = Compute intensive

4. **Efficiency**
   - Percentage of theoretical peak achieved
   - Considers all factors: occupancy, FMA usage, density

### Optimization Insights

The model provides actionable recommendations:

1. **Low θ**: Improve thread divergence
   - Reorganize branching logic
   - Use warp-level primitives

2. **Low m_FMA**: Enable FMA operations
   - Compiler flag: `-ffp-contract=fast`
   - Use intrinsics: `__fmaf_rn()`

3. **Low f_fp32**: Increase compute intensity
   - Optimize memory access patterns
   - Reduce control flow overhead
   - Consider kernel fusion

## Model Alignment with Theory

The implementation correctly follows the analytical model card:

✅ **Core Formula**: `Φ = θ × C_fp32 × (1 + m_FMA)`
✅ **Metrics Collection**: Real GPU data via NVBit
✅ **Window Analysis**: Can process sliding windows
✅ **FMA Counting**: 2 FLOPs per FMA as specified
✅ **Divergence Handling**: θ captures predication effects
✅ **Hardware Awareness**: C_fp32 configurable per GPU

## Example Use Cases

### 1. GEMM Optimization
```python
# Current: No FMA, 25% FP32 density
# Optimization: Enable FMA, increase compute intensity
# Potential speedup: 2x from FMA, 1.5x from density = 3x total
```

### 2. Performance Roofline
```python
# CUDA Core Bound: 32.82 FLOPs/cycle
# Memory Bound: [calculate from bandwidth]
# Actual Performance: min(cuda_bound, memory_bound, other_bounds)
```

### 3. Multi-Kernel Analysis
```python
# Process multiple kernels
# Identify common bottlenecks
# Generate optimization priorities
```

## Advanced Features

### Custom GPU Configuration

Edit `analyze_real_metrics.py`:
```python
# For A100 (64 FP32 lanes per SM)
model = CUDACoreAnalyticalModel(C_fp32=64)

# For RTX 3090 (128 FP32 lanes per SM)
model = CUDACoreAnalyticalModel(C_fp32=128)
```

### Sliding Window Analysis

The model supports processing instruction windows:
```python
# Process 400-instruction windows
windows = split_kernel_into_windows(kernel_trace, window_size=400)
distributions = model.process_window_metrics(windows)
```

### Integration with ML Models

Generate features for performance prediction:
```python
features = model.generate_performance_features(distributions)
# Use features as input to ML model
```

## Limitations and Assumptions

1. **FP32 Only**: Model focuses on FP32 CUDA cores
   - Does not model tensor cores
   - Does not model INT operations
   - Does not model SFU (special functions)

2. **Single SM**: Per-SM modeling
   - Scale to full GPU by multiplying by SM count
   - Assumes uniform work distribution

3. **Static Analysis**: Upper bounds only
   - Does not model dynamic effects
   - Does not capture memory latency hiding
   - Does not model scheduler policies

4. **Instruction-Level**: Not cycle-accurate
   - Abstracts microarchitectural details
   - Does not model pipeline stages

## Troubleshooting

### NVBit Tool Reports 0 Counts
- Ensure managed memory is enabled: `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1`
- Check CUDA architecture compatibility
- Verify injection functions are linked correctly

### Compilation Errors
- Check NVBit version (requires 1.7.5)
- Verify CUDA toolkit path
- Ensure C++ standard is c++14 or later

### Low Performance Metrics
- Kernel may be memory bound (check f_fp32)
- Thread divergence issues (check θ)
- No FMA usage (check m_FMA)

## Future Enhancements

1. **Additional Components**
   - Tensor core modeling
   - Memory subsystem modeling
   - SFU pipeline modeling

2. **Dynamic Analysis**
   - Cycle-accurate simulation
   - Cache behavior modeling
   - Scheduler policy effects

3. **Automation**
   - Automatic optimization suggestions
   - Code transformation tools
   - Performance regression detection

## References

1. **Concorde-CPU**: The original compositional modeling approach
2. **NVBit**: NVIDIA Binary Instrumentation Tool
3. **CUDA Programming Guide**: For FP32/FMA semantics
4. **GPU Architecture Manuals**: For C_fp32 values

## Contact and Contribution

This implementation demonstrates the Concorde-CPU methodology applied to GPU CUDA cores. The modular design allows easy extension to other GPU components following the same compositional approach.

---

**Note**: This model provides theoretical upper bounds. Actual performance depends on many factors including memory bandwidth, cache behavior, and dynamic scheduling effects. Use in conjunction with profiling tools for comprehensive performance analysis.