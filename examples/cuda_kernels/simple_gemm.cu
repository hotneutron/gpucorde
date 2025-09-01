/*
 * Simple GEMM CUDA kernel for testing NVBit instrumentation
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Simple GEMM kernel (C = A * B) - explicitly using FP32
__global__ void gemm_kernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        // Unroll loop a bit to generate more FP32 operations
        for (int k = 0; k < N; k += 4) {
            if (k < N) sum = __fadd_rn(sum, __fmul_rn(A[row * N + k], B[k * N + col]));
            if (k+1 < N) sum = __fadd_rn(sum, __fmul_rn(A[row * N + k+1], B[(k+1) * N + col]));
            if (k+2 < N) sum = __fadd_rn(sum, __fmul_rn(A[row * N + k+2], B[(k+2) * N + col]));
            if (k+3 < N) sum = __fadd_rn(sum, __fmul_rn(A[row * N + k+3], B[(k+3) * N + col]));
        }
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 256;  // Small matrix for quick testing
    const int size = N * N * sizeof(float);
    
    // Allocate memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Initialize matrices (simple pattern)
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);
    
    printf("Launching GEMM kernel with N=%d\n", N);
    gemm_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    printf("GEMM kernel completed\n");
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    
    return 0;
}