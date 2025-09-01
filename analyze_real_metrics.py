#!/usr/bin/env python3
"""
Analyze real GPU metrics collected by NVBit using the CUDA Core analytical model
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append('src/analytical_models')

from cuda_core_model import CUDACoreAnalyticalModel, CUDACoreMetrics

def main():
    # Load real GPU metrics from NVBit
    metrics_file = "fp32_metrics.csv"
    
    print("=" * 70)
    print("CUDA CORE ANALYTICAL MODEL - REAL GPU METRICS ANALYSIS")
    print("=" * 70)
    
    if not os.path.exists(metrics_file):
        print(f"Error: Metrics file '{metrics_file}' not found.")
        print("Please run the NVBit tool first to collect real GPU metrics.")
        return
    
    # Read the CSV file
    df = pd.read_csv(metrics_file)
    print(f"\nLoaded {len(df)} kernel measurements from NVBit")
    print("-" * 70)
    
    # Initialize the analytical model (RTX 3090 configuration with 128 FP32 lanes)
    # Adjust this based on your GPU architecture
    model = CUDACoreAnalyticalModel(C_fp32=128)
    
    for idx, row in df.iterrows():
        print(f"\n{'='*70}")
        print(f"KERNEL {row['kernel_id']} ANALYSIS")
        print(f"{'='*70}")
        
        # Display raw metrics
        print("\n📊 Raw GPU Metrics (from NVBit):")
        print(f"  Total instructions executed:  {row['total_instrs']:,}")
        print(f"  FP32 instructions executed:   {row['fp32_instrs']:,}")
        print(f"  FMA instructions executed:    {row['fma_instrs']:,}")
        print(f"  Average active threads:       {row['avg_active_threads']:.2f}")
        
        # Display calculated metrics
        print("\n📈 Calculated Metrics:")
        print(f"  θ (theta - warp occupancy):   {row['theta']:.4f}")
        print(f"  m_FMA (FMA ratio):            {row['m_fma']:.4f}")
        print(f"  f_fp32 (FP32 density):        {row['f_fp32']:.4f}")
        
        # Create metrics object (we need to adapt to the model's expected format)
        # The model expects window-based metrics, but we have kernel-level metrics
        # We'll treat the entire kernel as one window
        metrics = CUDACoreMetrics(
            window_id=0,
            sm_id=0,
            total_instructions=int(row['total_instrs']),
            fp32_instructions=int(row['fp32_instrs']),
            fma_instructions=int(row['fma_instrs']),
            warp_instructions=int(row['total_instrs']),  # Approximation
            active_threads_sum=int(row['avg_active_threads'] * row['total_instrs'])
        )
        
        # Note: We're using the values from the CSV directly since they're already calculated
        # But we could also use the metrics object's properties
        
        # Calculate analytical bound using the CSV values directly
        # Φ = θ * C_fp32 * (1 + m_FMA) * f_fp32
        theta = row['theta']
        m_fma = row['m_fma']
        f_fp32 = row['f_fp32']
        
        # Basic bound
        phi_basic = theta * model.C_fp32 * (1 + m_fma)
        
        # Extended bound with FP32 density
        phi_extended = f_fp32 * phi_basic
        
        # Peak theoretical performance
        peak_flops = model.C_fp32 * 2  # Assuming all FMAs (2 FLOPs per FMA)
        
        print("\n🎯 Analytical Model Results:")
        print(f"  CUDA Core Bound (Φ_basic):    {phi_basic:.2f} FLOPs/cycle")
        print(f"  CUDA Core Bound (Φ_extended): {phi_extended:.2f} FLOPs/cycle")
        print(f"  Peak theoretical:              {peak_flops} FLOPs/cycle")
        print(f"  Efficiency vs peak:            {phi_extended/peak_flops*100:.1f}%")
        
        # Performance insights
        print("\n💡 Performance Insights:")
        
        # Warp occupancy analysis
        if theta < 0.5:
            print(f"  ⚠️  CRITICAL: Very low warp occupancy ({theta:.2%})")
            print("      → Severe thread divergence or low parallelism")
        elif theta < 0.8:
            print(f"  ⚠️  Low warp occupancy ({theta:.2%})")
            print("      → Consider reducing thread divergence")
        else:
            print(f"  ✅ Good warp occupancy ({theta:.2%})")
        
        # FMA usage analysis
        if m_fma == 0:
            print(f"  ⚠️  No FMA usage detected")
            print("      → Consider using FMA instructions to double throughput")
            print("      → Compiler flag: -ffp-contract=fast or use __fmaf_rn()")
        elif m_fma < 0.3:
            print(f"  ⚠️  Low FMA usage ({m_fma:.2%})")
            print("      → More opportunities for FMA operations")
        else:
            print(f"  ✅ Good FMA usage ({m_fma:.2%})")
        
        # FP32 density analysis
        if f_fp32 < 0.3:
            print(f"  ⚠️  Low FP32 density ({f_fp32:.2%})")
            print("      → Kernel is likely memory-bound or control-heavy")
            print("      → Consider optimizing memory access patterns")
        elif f_fp32 < 0.5:
            print(f"  ⚠️  Moderate FP32 density ({f_fp32:.2%})")
            print("      → Mix of compute and other operations")
        else:
            print(f"  ✅ Good FP32 density ({f_fp32:.2%})")
            print("      → Kernel is compute-intensive")
        
        # Optimization recommendations
        print("\n🔧 Optimization Recommendations:")
        
        potential_speedup = 1.0
        
        if theta < 1.0:
            speedup_from_occupancy = 1.0 / theta
            potential_speedup *= min(speedup_from_occupancy, 1.2)
            print(f"  • Improving warp occupancy could yield up to {(speedup_from_occupancy-1)*100:.0f}% speedup")
        
        if m_fma < 0.5 and f_fp32 > 0.3:
            speedup_from_fma = 1.0 + (0.5 - m_fma)
            potential_speedup *= speedup_from_fma
            print(f"  • Using more FMA instructions could yield up to {(speedup_from_fma-1)*100:.0f}% speedup")
        
        if f_fp32 < 0.5:
            print(f"  • Increasing compute intensity could improve GPU utilization")
        
        if potential_speedup > 1.1:
            print(f"\n  📈 Total potential speedup: up to {(potential_speedup-1)*100:.0f}%")
        
        # Actual FLOPS calculation
        print("\n📐 Actual Performance:")
        # Each FMUL/FADD is 1 FLOP, each FMA is 2 FLOPs
        non_fma_fp32 = row['fp32_instrs'] - row['fma_instrs']
        total_flops = non_fma_fp32 + (2 * row['fma_instrs'])
        print(f"  Total FLOPs executed: {total_flops:,}")
        
        # If we had timing information, we could calculate actual FLOPS/sec
        print(f"  Note: Add kernel execution time to calculate GFLOPS")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nThis analysis uses the Concorde-CPU inspired analytical model")
    print("to understand CUDA core performance characteristics.")
    print("The model provides upper bounds and optimization insights.")

if __name__ == "__main__":
    main()