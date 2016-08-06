/* Includes, system */
//#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

/* local utils */
#include "utils.h"

//### EXAMPLES:
//CUDA_MALLOC(d_A, n, "d_A");
//CUDA_FREE(d_A, "d_A");

//#define THREADS_PER_BLOCK 500
//#define THREADS_PER_BLOCK 3

using namespace std;

//### TEST BRIDGE FOR USING WITH PYTHON VIA CTYPES

extern "C" {
/**
 * Allocate memory on device and copy from host to device
 */
int call1(float * h_X, float ** d_X, int n) {
 
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    
    int n2 = n*n;
 
    printf("CALL1:\n");
    printf("pointer address of d_X: %p\n", *d_X); 
 
    CUDA_MALLOC(*d_X, n2, "d_X");
    
    printf("pointer address of d_X: %p\n", *d_X); 
    
    CUDA_MEMCPY(*d_X, h_X, n2, cudaMemcpyHostToDevice, "Error copying d_X from host to device")
    
    //cublasDestroy(handle);
    
    return EXIT_SUCCESS;
    
}

/*
 * Copy from device  back to host and deallocate memory on device
 */
int call2(float * h_X, float ** d_X, int n) {
 
    cublasStatus_t status;
    cublasHandle_t handle;
    //status = cublasCreate(&handle);
    
    int n2 = n*n;
    
    printf("CALL2:\n");
    printf("pointer address of d_X: %p\n", *d_X); 
    
    CUDA_MEMCPY(*d_X, h_X, n2, cudaMemcpyDeviceToHost, "Error copying d_X from device to host")

    CUDA_FREE(*d_X, "d_X");
    
    //cublasDestroy(handle);
    
    return EXIT_SUCCESS;
    
}

}