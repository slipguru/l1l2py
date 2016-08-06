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
//#include <cusolverSp.h> // Cusolver for SVD and stuff
#include <cusolverDn.h> // Cusolver for SVD and stuff
#include <cusolver_common.h> // Cusolver for SVD and stuff

/* local utils */
#include "utils.h"

//### EXAMPLES:
//CUDA_MALLOC(d_A, n, "d_A");
//CUDA_FREE(d_A, "d_A");

#define THREADS_PER_BLOCK 500
//#define THREADS_PER_BLOCK 3

using namespace std;

/*
 * Creates the "inverted" sigma matrix starting from the vector of singular values
 *
 */
__global__ void invert_sigma(float * d_S, float * d_Sinv, int n) {
    
    float myeps = 0.001;
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    //# Soft-Thresholding
    
    if (i < n) {
        //### TODO must be done outside
        //### Fill the line with zeros
        for (int j = 0; j < n; j++) {
            d_Sinv[i*n + j] = 0;
        }
        
        if (d_S[i] > d_S[0]*myeps) {
            d_Sinv[i*n + i] = 1/d_S[i];
        } else {
            d_Sinv[i*n + i] = 0;
        }
    }
    
}


/**
 *
 * d_X : the matrix whose pseudoinverse must be computed
 * n : the number of rows of the matrix
 * p : the number of columns of the matrix
 * d_Xpinv : the pseudoinverse of d_X
 */
int pinv(float * d_X_orig, int n, int p, float * d_Xpinv, cublasHandle_t handle, cusolverDnHandle_t cs_handle) {
 
    //cout << "CUSOLVER_STATUS_SUCCESS: " << CUSOLVER_STATUS_SUCCESS << endl;
    
    if (n < p) {
        cout << "n must be greater or equal than p; aborting." << endl;
        return -1;
    }
 
    int n2 = n*n;
 
    cublasStatus_t status;
    cusolverStatus_t cs_status;
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    int numBlocks;
    int threadsPerBlock = THREADS_PER_BLOCK;
    
    float * d_X = 0;
    CUDA_MALLOC(d_X, n2, "d_X");
    
    CUDA_MEMCPY(d_X, d_X_orig, n2, cudaMemcpyDeviceToDevice, "Error copying d_X from device to device")
    
    //### used to control the level of debug output
    int debug = 0;
    
    int Lwork; //### size for the work buffer for the SVD computation
    
    //### Compute the size of the buffer for the SVD computation
    cs_status = cusolverDnSgesvd_bufferSize(cs_handle, n, p, &Lwork);
    
    if (cs_status != 0) {
        printf("Error in cusolverDnSgesvd_bufferSize! \n");
        return -1;
    }
    
    if (debug) {
        printf("Lwork size = %d\n", Lwork);
    }
 
    //### Allocate matrices/vectors for qsolvers
    
    float * Work = 0;
    //### The temp matrix Work
    CUDA_MALLOC(Work, Lwork, "Work");
    
    int * devInfo = 0;
    //### stuff
    CUDA_MALLOC(devInfo, 1, "devInfo");
    
    float * d_S = 0;
    //### The (device) vector for storing singular values
    //### More space is allocated in order to store the actual matrix
    CUDA_MALLOC(d_S, n, "d_S");
    
    float * d_U = 0;
    //### The (device) vector for storing singular values
    CUDA_MALLOC(d_U, n2, "d_U");
    
    float * d_VH = 0;
    //### The (device) vector for storing singular values
    CUDA_MALLOC(d_VH, n2, "d_VH");
    
    float * d_Sinv = 0;
    //### The (device) vector for storing singular values
    CUDA_MALLOC(d_Sinv, n2, "d_Sinv");
 
    //### Actually compute the SVD
    cs_status = cusolverDnSgesvd(cs_handle, 'A', 'A', n, p, d_X, n, d_S, d_U, n, d_VH, p, Work, Lwork, NULL, devInfo);
    
    if (cs_status != 0) {
        printf("Error in cusolverDnSgesvd: %d! \n", cs_status);
        return -1;
    }
    
    //### Create the inverted matrix with singular values
    
    //### Compute the number of necessary blocks and threads per block
    
    if (n <= THREADS_PER_BLOCK) {
        numBlocks = 1;
        threadsPerBlock = n;
    } else {
        numBlocks = n/THREADS_PER_BLOCK + 1;
        threadsPerBlock = n/numBlocks + 1; //###XXX to check
    }
    
    invert_sigma<<<numBlocks, threadsPerBlock>>>(d_S, d_Sinv, n);
    
    if (debug) {
        cout << "cs_status: " << cs_status << endl;
    }
    
    //### DEBUG just check the content of the "inverted" matrix
    //float * h_Sinv;
    //h_Sinv = (float *)malloc(n2 * sizeof(float));
    //
    ////### Copy matrix to device
    //if (cudaMemcpy(h_Sinv, d_Sinv, n2 * sizeof(*d_Sinv), cudaMemcpyDeviceToHost) != cudaSuccess) {
    //    fprintf(stderr, "!!!! Error copying d_Sinv from device\n");
    //    return EXIT_FAILURE;
    //}
    //
    //for (int i1 = 0; i1 < n; i1++) {
    //    for (int i2 = 0; i2 < n; i2++) {
    //        cout << h_Sinv[i1*n + i2] << " ";
    //    }
    //    cout << endl;
    //}
    //
    //free(h_Sinv);
    
    //### DEBUG
    cout << "DEBUG: printing stuff" << endl;
    float * h_tmp;
    h_tmp = (float *)malloc(n2 * sizeof(float));
    
    //CUDA_MEMCPY(h_tmp, d_S, n, cudaMemcpyDeviceToHost, "Error copying X from host to device")
    CUDA_MEMCPY(h_tmp, d_Sinv, n2, cudaMemcpyDeviceToHost, "Error copying X from host to device")
    
    //print_array(n, n, h_tmp);
    
    free(h_tmp);
    
    //### A+ = V E+ U*
    
    //status = cublasSgemm(handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
    
    //### Check matrix dimensions!!!
    // E+ = V x E+
    status = cublasSgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, n, n, n, &alpha, d_VH, n, d_Sinv, n, &beta, d_Sinv, n);
    //cout << "status: " << status << endl;
    
    //### d_Xpinv = d_Sinv x d_U^H
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_C, n, n, n, &alpha, d_Sinv, n, d_U, n, &beta, d_Xpinv, n);
    
    CUDA_FREE(Work, "Work");
    CUDA_FREE(devInfo, "devInfo");
    
    CUDA_FREE(d_S, "d_S");
    CUDA_FREE(d_U, "d_U");
    CUDA_FREE(d_VH, "d_VH");
    CUDA_FREE(d_Sinv, "d_Sinv");
    CUDA_FREE(d_X, "d_X");
    
    return EXIT_SUCCESS;
    
}

//### TEST BRIDGE FOR USING WITH PYTHON VIA CTYPES

extern "C" {
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
    
    //cout << "printing h_X:" << endl;
    //for (int i = 0; i < n*p; i++){
        //cout << h_X[i] << " ";
    //}
    
    cout << endl;
    
    //### The data matrix X (transposed)
    CUDA_MALLOC(d_X, n*p, "d_X");
    CUDA_MALLOC(d_Xpinv, n*p, "d_Xpinv");
    
    
    
    /* Initialize the device matrices with the host matrices */
    status = cublasSetMatrix(p, n, sizeof(h_X[0]), h_X, p, d_X , p);
    CUDA_CHECK_STATUS(status, "device access error (write X)");
    
    //######### DEBUG START
    //CUDA_MEMCPY(h_Xpinv, d_X, n*p, cudaMemcpyDeviceToHost, "Error copying d_Xpinv from device to host")
    
    //cout << "printing h_Xpinv (middle):" << endl;
    //for (int i = 0; i < n*p; i++){
        //cout << h_Xpinv[i] << " ";
    //}
    //######### DEBUG END
    
    //### DO STUFF
    
    pinv(d_X, n, p, d_Xpinv, handle, cs_handle);
    
    //cudaError_t cerr;
    //cerr = cudaGetLastError();
    //cout << "cerr: " << cudaGetErrorString(cerr) << endl;;
    
    //### copy d_Xpinv to host h_Xpinv
    CUDA_MEMCPY(h_Xpinv, d_Xpinv, n*p, cudaMemcpyDeviceToHost, "Error copying d_Xpinv from device to host")
    
    //cout << "printing h_Xpinv (after):" << endl;
    //for (int i = 0; i < n*p; i++){
        //cout << h_Xpinv[i] << " ";
    //}
    
    CUDA_FREE(d_X, "d_X");
    CUDA_FREE(d_Xpinv, "d_Xpinv");
    
    return 0;
    
  }
}