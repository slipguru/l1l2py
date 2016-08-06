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

#include "pinv.h"

//### EXAMPLES:
//CUDA_MALLOC(d_A, n, "d_A");
//CUDA_FREE(d_A, "d_A");

/* Matrix size */
//#define N  (2000)

//#define THREADS_PER_BLOCK 200
#define THREADS_PER_BLOCK 500

using namespace std;

int ridge_regression(float * h_XT, float * h_Y, int n, int p, float lambda, float * beta_out) {
 
    cublasStatus_t status;
    cublasHandle_t handle;
    
    cusolverStatus_t cs_status;
    cusolverDnHandle_t cs_handle;
    
    struct timeval t0, t1, t2;
    
    gettimeofday(&t0,NULL);
    
    float alpha = 1.0f;
    float beta = 1.0f;
    int n_tmp, n2;
    
    int debug = 1;
    
    float * h_tmp;
    
    status = cublasCreate(&handle);
    cusolverDnCreate(&cs_handle);
    
    //### determine the size of the squared matrix
    if (p > n) {
        n_tmp = n;
    } else {
        n_tmp = p;
    }
    
    n2 = n_tmp*n_tmp;
    
    //### TODO Y should be probably already on device (or not?)
    //### copy XT and Y on device
    float * d_XT = 0;
    CUDA_MALLOC(d_XT, n*p, "d_XT");
    if (cudaMemcpy(d_XT, h_XT, n * p * sizeof(*d_XT), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "!!!! Error copying h_XT to d_XT on device\n");
        return EXIT_FAILURE;
    }
    
    float * d_Y = 0;
    CUDA_MALLOC(d_Y, n, "d_Y");
    if (cudaMemcpy(d_Y, h_Y, n * sizeof(*d_Y), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "!!!! Error copying h_Y to d_Y on device\n");
        return EXIT_FAILURE;
    }
    
    //### allocate memory on host
    h_tmp = (float *)calloc(n2, sizeof(float));
    
    float val = lambda*n;
    
    //### create the diagonal matrix where entries are all mu * n
    for (int i = 0; i < n_tmp; i++) {
        h_tmp[i*n_tmp + i] = val;
    }
    
    float * d_tmp = 0;
    //### The matrix on the device for the main square matrix
    CUDA_MALLOC(d_tmp, n2, "d_tmp");
 
    //### copy h_tmp on device
    if (cudaMemcpy(d_tmp, h_tmp, n2 * sizeof(*d_tmp), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "!!!! Error copying h_tmp to d_tmp on device\n");
        return EXIT_FAILURE;
    }
    
    gettimeofday(&t1,NULL);
 
    if (p > n) {
        // d_tmp = <X,XT> + mu*n*eye(n)
        status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, p, &alpha, d_XT, p, d_XT, p, &beta, d_tmp, n);
        //cusolverDnSgesvd_bufferSize(cs_handle, n, n, &Lwork);
    } else {
        // d_tmp = <XT,X> + mu*n*eye(p)
        status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, p, p, n, &alpha, d_XT, p, d_XT, p, &beta, d_tmp, p);
        //cusolverDnSgesvd_bufferSize(cs_handle, p, p, &Lwork);
    }
    
    //### d_tmp = pinv(d_tmp)
    pinv(d_tmp, n_tmp, n_tmp, d_tmp, handle, cs_handle);
    
    //### Set beta to 0 for the rest of the computations
    beta = 0.0;
    
    //### at this point, d_tmp is the pseudoinverse (at least I hope so!)
    //### d_XT is overwritten in the process. problem?
    if (n < p) {
        // np.dot(np.dot(data.T, tmp), labels.reshape(-1, 1))
        
        //### save the result on d_XT
        status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, n, n, &alpha, d_XT, p, d_tmp, n, &beta, d_XT, p);
        CUDA_CHECK_STATUS(status, "Error performing operation np.dot(X.T, tmp)");
        
        status = cublasSgemv(handle, CUBLAS_OP_N, p, n, &alpha, d_XT, p, d_Y, 1, &beta, d_tmp, 1);
        CUDA_CHECK_STATUS(status, "Error performing operation np.dot(X.T, Y)");
        
    } else {
        // np.dot(tmp, np.dot(data.T, labels.reshape(-1, 1)))
        
        status = cublasSgemv(handle, CUBLAS_OP_N, p, n, &alpha, d_XT, p, d_Y, 1, &beta, d_XT, 1);
        CUDA_CHECK_STATUS(status, "Error performing operation np.dot(X.T, Y)");
        
        status = cublasSgemv(handle, CUBLAS_OP_N, p, p, &alpha, d_tmp, p, d_XT, 1, &beta, d_tmp, 1);
        CUDA_CHECK_STATUS(status, "Error performing operation np.dot(tmp, X.T)");
    }
    
    if (cudaMemcpy(beta_out, d_tmp, p * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "!!!! Error copying d_tmp to beta_out on device\n");
        return EXIT_FAILURE;
    }
    
    gettimeofday(&t2,NULL);
    
    double init_time, comp_time, total_time;
    
    init_time = (t1.tv_sec-t0.tv_sec)*1000000;
    init_time += (t1.tv_usec-t0.tv_usec);
    
    comp_time = (t2.tv_sec-t1.tv_sec)*1000000;
    comp_time += (t2.tv_usec-t1.tv_usec);
    
    total_time = (t2.tv_sec-t0.tv_sec)*1000000;
    total_time += (t2.tv_usec-t0.tv_usec);
    
    if (debug) {
        printf("Initialization time: %lf\n", init_time/1000000.0);
        printf("Computation time: %lf\n", comp_time/1000000.0);
        printf("Total time: %lf\n", total_time/1000000.0);
    }
    
    //### cleanup
    CUDA_FREE(d_tmp, "d_tmp");
    
    CUDA_FREE(d_XT, "d_XT");
    
    CUDA_FREE(d_Y, "d_Y");
    
    free(h_tmp);
    
    cusolverDnDestroy(cs_handle);
    cublasDestroy(handle);
    
    return EXIT_SUCCESS;
    
}
extern "C" {

int ridge_regression_bridge(float * h_XT, float * h_Y, int n, int p, float lambda, float * beta_out) {
    return ridge_regression(h_XT, h_Y, n, p, lambda, beta_out);
}


}
//n, p = data.shape
//
//    if n < p:
//        tmp = np.dot(data, data.T)
//        if mu:
//            tmp += mu*n*np.eye(n)
//        tmp = la.pinv(tmp)
//
//        return np.dot(np.dot(data.T, tmp), labels.reshape(-1, 1))
//    else:
//        tmp = np.dot(data.T, data)
//        if mu:
//            tmp += mu*n*np.eye(p)
//        tmp = la.pinv(tmp)
//
//        return np.dot(tmp, np.dot(data.T, labels.reshape(-1, 1)))