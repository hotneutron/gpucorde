/*
 * Simplified FP32 counter - all in one file
 */

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <unordered_set>
#include <vector>
#include <fstream>

/* NVBit headers */
#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"

/* Global counters */
__managed__ uint64_t total_instrs = 0;
__managed__ uint64_t fp32_instrs = 0;
__managed__ uint64_t fma_instrs = 0;
__managed__ uint64_t active_threads_total = 0;

/* Kernel counter */
uint32_t kernel_id = 0;

/* Mutex for thread safety */
pthread_mutex_t mutex;

/* Output file */
std::ofstream output_file;

/* Set to track instrumented functions */
std::unordered_set<CUfunction> instrumented_funcs;

/* Forward declaration of GPU-side function */
extern "C" __device__ __noinline__ void count_instr(int pred,
                                                     uint32_t is_fp32, 
                                                     uint32_t is_fma,
                                                     uint64_t pcounter_total,
                                                     uint64_t pcounter_fp32,
                                                     uint64_t pcounter_fma,
                                                     uint64_t pcounter_active);

/* Check if instruction is FP32 */
bool is_fp32_instruction(Instr* instr) {
    const char* opcode = instr->getOpcode();
    std::string op(opcode);
    
    // Check for FP32 operations
    return (op.find("FADD.F32") != std::string::npos ||
            op.find("FMUL.F32") != std::string::npos ||
            op.find("FFMA.F32") != std::string::npos ||
            op.find("FSETP.") != std::string::npos && op.find(".F32") != std::string::npos ||
            // Also check for simple FADD/FMUL without suffix (usually F32)
            (op == "FADD" || op == "FMUL" || op == "FFMA"));
}

/* Check if instruction is FMA */
bool is_fma_instruction(Instr* instr) {
    const char* opcode = instr->getOpcode();
    std::string op(opcode);
    
    return (op.find("FFMA") != std::string::npos || 
            op.find("FMA") != std::string::npos) &&
           (op.find(".F32") != std::string::npos || 
            op.find(".F16") == std::string::npos); // Default to F32 if no F16
}

/* Instrument function */
void instrument_func(CUcontext ctx, CUfunction func) {
    /* Avoid re-instrumenting */
    if (instrumented_funcs.find(func) != instrumented_funcs.end()) return;
    instrumented_funcs.insert(func);
    
    /* Get related functions */
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);
    
    /* Instrument each function */
    for (auto f : related) {
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        
        printf("Instrumenting %s: %lu instructions\n", 
               nvbit_get_func_name(ctx, f), instrs.size());
        
        int fp32_count = 0, fma_count = 0;
        for (auto instr : instrs) {
            /* Check instruction type */
            uint32_t is_fp32 = is_fp32_instruction(instr) ? 1 : 0;
            uint32_t is_fma = is_fma_instruction(instr) ? 1 : 0;
            
            if (is_fp32) {
                fp32_count++;
                if (is_fma) fma_count++;
            }
            
            /* Insert instrumentation for all instructions */
            nvbit_insert_call(instr, "count_instr", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_const_val32(instr, is_fp32);
            nvbit_add_call_arg_const_val32(instr, is_fma);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&total_instrs);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&fp32_instrs);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&fma_instrs);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&active_threads_total);
        }
        printf("  Found %d FP32 instructions (%d FMA)\n", fp32_count, fma_count);
    }
}

/* NVBit initialization */
void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    pthread_mutex_init(&mutex, NULL);
    
    output_file.open("fp32_metrics.csv");
    output_file << "kernel_id,total_instrs,fp32_instrs,fma_instrs,avg_active_threads,theta,m_fma,f_fp32\n";
    
    printf("====================================\n");
    printf("Simple FP32 Counter V2 Initialized\n");
    printf("====================================\n");
}

/* CUDA event handler */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    
    if (cbid == API_CUDA_cuLaunchKernel || 
        cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchCooperativeKernel) {
        
        pthread_mutex_lock(&mutex);
        
        if (!is_exit) {
            /* Kernel starting - instrument it */
            printf("\n--- Kernel %u starting ---\n", kernel_id);
            
            cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
            
            /* Instrument the function */
            instrument_func(ctx, p->f);
            
            /* Enable instrumentation */
            nvbit_enable_instrumented(ctx, p->f, true);
            
        } else {
            /* Kernel finished - collect metrics */
            cudaDeviceSynchronize();
            
            /* Calculate metrics */
            double avg_active = total_instrs > 0 ? 
                               (double)active_threads_total / total_instrs : 0;
            double theta = avg_active / 32.0;  // Normalized by warp size
            double m_fma = fp32_instrs > 0 ? 
                          (double)fma_instrs / fp32_instrs : 0;
            double f_fp32 = total_instrs > 0 ? 
                           (double)fp32_instrs / total_instrs : 0;
            
            /* Output results */
            printf("Kernel %u completed:\n", kernel_id);
            printf("  Total instructions executed: %lu\n", total_instrs);
            printf("  FP32 instructions executed: %lu\n", fp32_instrs);
            printf("  FMA instructions executed: %lu\n", fma_instrs);
            printf("  Average active threads: %.2f\n", avg_active);
            printf("  Theta (warp occupancy): %.4f\n", theta);
            printf("  m_FMA (FMA ratio): %.4f\n", m_fma);
            printf("  f_fp32 (FP32 density): %.4f\n", f_fp32);
            
            /* Write to file */
            output_file << kernel_id << ","
                       << total_instrs << ","
                       << fp32_instrs << ","
                       << fma_instrs << ","
                       << avg_active << ","
                       << theta << ","
                       << m_fma << ","
                       << f_fp32 << "\n";
            output_file.flush();
            
            /* Reset counters */
            total_instrs = 0;
            fp32_instrs = 0;
            fma_instrs = 0;
            active_threads_total = 0;
            
            kernel_id++;
        }
        
        pthread_mutex_unlock(&mutex);
    }
}

/* Cleanup */
void nvbit_at_term() {
    if (output_file.is_open()) {
        output_file.close();
    }
    
    printf("====================================\n");
    printf("Simple FP32 Counter V2 Terminated\n");
    printf("Metrics saved to fp32_metrics.csv\n");
    printf("====================================\n");
}