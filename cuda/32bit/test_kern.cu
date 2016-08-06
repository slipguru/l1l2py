/* Includes, system */
//#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

#include "utils.h"

using namespace std;

// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C) {
    int j = threadIdx.x;
    
    int imax = 1000000000;
    //           2147483647
    //int imax =   50000000;
    
    for (int i = 0; i < imax; i++) {
        //C[j] = C[j]+1.0;
        atomicAdd(C+j, 1.0);
        C[2] = i;
    }
    
    
    
    
    //C[i] = A[i] + B[i];
    //C[0] = A[0] + B[0];
    
}
    
/**
 *
 * d_matrixT : the matrix, transposed, on the device; the shape of the original matrix is n X p, so the shape of the transposed matrix is p X n
 * n
 *
 * ###TODO stub
 */
int main() {
    
    int n = 4;
    
    cublasStatus_t status;
    cublasHandle_t handle;
    
    int dev = findCudaDevice(1, (const char **) NULL);
    if (dev == -1)
    {
        return EXIT_FAILURE;
    }

    status = cublasCreate(&handle);
    
    //### Allocate memory for host A
    float * h_A = (float *)malloc(n*sizeof(float));
    float * h_B = (float *)malloc(n*sizeof(float));
    float * h_C = (float *)malloc(n*sizeof(float));
    
    //### Initialize h_A and h_B
    for (int i = 0; i < n; i++) {
        h_A[i] = i;
        h_B[i] = 2*i;
        h_C[i] = 666;
    }
    
    printf("C before: %f\n", h_C[0]);
    
    float * d_A = 0;
    //### Allocate memory for matrix A on device
    CUDA_MALLOC(d_A, n, "d_A");
    
    float * d_B = 0;
    //### Allocate memory for matrix A on device
    CUDA_MALLOC(d_B, n, "d_B");
    
    float * d_C = 0;
    CUDA_MALLOC(d_C, n, "d_C");
    
    cudaMemcpy(d_C, h_C, n * sizeof(*d_C), cudaMemcpyHostToDevice);
    
    // Kernel invocation
    //dim3 threadsPerBlock(2);
    dim3 threadsPerBlock(1);
    dim3 numBlocks(1);
    
    VecAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
    
    //printf("immediately after!");
    
    cudaDeviceSynchronize();
    
    printf("Last error message: %s\n", cudaGetErrorString(cudaGetLastError()));
    
    cudaMemcpy(h_C, d_C, n * sizeof(*d_C), cudaMemcpyDeviceToHost);
    
    printf("C[0] after: %f\n", h_C[0]);
    printf("C[1] after: %f\n", h_C[1]);
    printf("C[2] after: %f\n", h_C[2]);
    
    printf("Last error message: %s\n", cudaGetErrorString(cudaGetLastError()));
    
    //### Copy on device matrix A
    status = cublasSetVector(n, sizeof(h_A[0]), h_A, 1, d_A, 1);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write X)\n");
        return EXIT_FAILURE;
    }
    
   
    
    
    
    //cudaMemcpy(h_S, d_S, n_tmp * sizeof(*d_S), cudaMemcpyDeviceToHost);
    
    CUDA_FREE(d_A, "d_A");
    CUDA_FREE(d_B, "d_B");
    CUDA_FREE(d_C, "d_C");
    
    free(h_A);
    free(h_B);
    free(h_C);
    
    cublasDestroy(handle);
    
    return EXIT_SUCCESS;
}