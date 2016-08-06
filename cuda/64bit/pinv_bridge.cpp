/*
 * Just a bridge module intended to be called from python
 *
 *
 */

#include "pinv.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>


/**
 *
 */
int pinv_bridge(float * h_X, float * h_Xpinv, int n, int p) {
    
    cublasStatus_t status;
    cublasHandle_t handle;
    
    cusolverDnHandle_t cs_handle;
    
    status = cublasCreate(&handle);
    cusolverDnCreate(&cs_handle);
    
    float * d_X = 0;
    float * d_Xpinv = 0;
    
    //### The data matrix X (transposed)
    CUDA_MALLOC(d_X, n*p, "d_X");
    CUDA_MALLOC(d_Xpinv, n*p, "d_Xpinv");
    
    /* Initialize the device matrices with the host matrices */
    status = cublasSetMatrix(p, n, sizeof(h_X[0]), h_X, p, d_X , p);
    CUDA_CHECK_STATUS(status, "device access error (write X)");
    
    //### DO STUFF
    
    pinv(d_X, n, p, d_Xpinv, handle, cs_handle);
    
    CUDA_FREE(d_X, "d_X");
    CUDA_FREE(d_Xpinv, "d_Xpinv");
    
    
    
    return 0;
    
}