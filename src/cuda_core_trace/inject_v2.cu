/*
 * Injection functions for V2 counter
 */

#include <stdint.h>
#include "utils/utils.h"

/* GPU-side instruction counter */
extern "C" __device__ __noinline__ void count_instr(int pred,
                                                     uint32_t is_fp32, 
                                                     uint32_t is_fma,
                                                     uint64_t pcounter_total,
                                                     uint64_t pcounter_fp32,
                                                     uint64_t pcounter_fma,
                                                     uint64_t pcounter_active) {
    /* Get active mask */
    const int active_mask = __ballot_sync(__activemask(), 1);
    
    /* Get predicate mask */
    const int predicate_mask = __ballot_sync(__activemask(), pred);
    
    /* Get lane id */
    const int laneid = get_laneid();
    
    /* Get first active lane */
    const int first_laneid = __ffs(active_mask) - 1;
    
    /* Count active threads */
    const int num_active = __popc(predicate_mask);
    
    /* Only first active thread performs atomics */
    if (first_laneid == laneid) {
        /* Always count total instructions (warp level) */
        if (num_active > 0) {
            atomicAdd((unsigned long long*)pcounter_total, 1);
            atomicAdd((unsigned long long*)pcounter_active, num_active);
            
            if (is_fp32) {
                atomicAdd((unsigned long long*)pcounter_fp32, 1);
                if (is_fma) {
                    atomicAdd((unsigned long long*)pcounter_fma, 1);
                }
            }
        }
    }
}