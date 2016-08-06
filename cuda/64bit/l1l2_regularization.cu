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

/* Matrix size */
//#define N  (2000)

//#define THREADS_PER_BLOCK 200
//#define THREADS_PER_BLOCK 500
#define THREADS_PER_BLOCK 1024

using namespace std;

/*
 * Kernel for soft thresholding
 *
 */
__global__ void SoftThreshFISTA(double * d_precalc, double * d_nsigma, double * d_mu_s, double * d_tau_s, double * d_beta, double * d_aux_beta, double * d_beta_next, double * d_beta_diff, double * d_value, double * d_t, double * d_t_next) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    //# Soft-Thresholding

    //value = (precalc / nsigma) + ((1.0 - mu_s) * aux_beta)
    d_value[i] = (d_precalc[i] / (d_nsigma[0])) + ((1.0 - (d_mu_s[0])) * d_aux_beta[i]);

    //beta_next = np.sign(value) * np.clip(np.abs(value) - tau_s, 0, np.inf)
    //float newval = fabsf(d_value[i]) - (d_tau_s[0]);
    //if (newval > 0) {
    //    d_beta_next[i] = copysignf(newval, d_value[i]);
    //    //d_beta_next[i] = i;
    //    //d_beta_next[i] = newval;
    //} else {
    //    d_beta_next[i] = 0.0f;
    //}

    //### Soft thresholding without IF statements
    double newval = fabsf(d_value[i]) - (d_tau_s[0]);
    // d_beta_next[i] = ((1+copysignf(1, newval))/2)*newval;
    d_beta_next[i] = (!signbit(d_value[i])*2 - 1) * (!signbit(newval))*newval;


    //######## FISTA ####################################################
    //beta_diff = (beta_next - beta)
    //d_beta_diff[i] = (d_beta_next[i] - d_beta[i]);
    d_beta_diff[i] = (d_beta_next[i] - d_beta[i]);

    //float t_next = 0.5 * (1.0 + sqrt(1.0 + 4.0 * (d_t[0]*d_t[0])));
    //d_aux_beta[i] = d_beta_next[i] + ((d_t[0] - 1.0)/t_next)*d_beta_diff[i];
    
    d_t_next[0] = 0.5 * (1.0 + sqrt(1.0 + 4.0 * (d_t[0]*d_t[0])));
    d_aux_beta[i] = d_beta_next[i] + ((d_t[0] - 1.0)/ d_t_next[0])*d_beta_diff[i];

}

/**
 *
 * d_matrixT : the matrix, transposed, on the device; the shape of the original matrix is n X p, so the shape of the transposed matrix is p X n
 * n
 *
 */
float _sigma(double * d_matrixT, int n, int p, double mu, cublasHandle_t handle) {

    double * d_tmp = 0;
    int n2;
    cublasStatus_t status;
    double alpha = 1.0;
    double beta = 0.0;
    int n_tmp;

    int Lwork; //### size for the work buffer for the SVD computation

    cusolverStatus_t cs_status;
    cusolverDnHandle_t cs_handle;

    //### XXX is it needed here as well?
    //cublasSetDevice(0);
    
    cusolverDnCreate(&cs_handle);

    if (p > n) {
        n2 = n*n;
        n_tmp = n;
    } else {
        n2 = p*p;
        n_tmp = p;
    }

    //### The temp matrix d_tmp
    CUDA_MALLOC(d_tmp, n2, "d_tmp");

    if (p > n) {
        status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, p, &alpha, d_matrixT, p, d_matrixT, p, &beta, d_tmp, n);
        cusolverDnDgesvd_bufferSize(cs_handle, n, n, &Lwork);
    } else {
        status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, p, p, n, &alpha, d_matrixT, p, d_matrixT, p, &beta, d_tmp, p);
        cusolverDnDgesvd_bufferSize(cs_handle, p, p, &Lwork);
    }

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! Error performing operation XTY \n");
        return EXIT_FAILURE;
    }

    //printf("Lwork: %d", Lwork);

    double * Work = 0;
    //### The temp matrix Work
    CUDA_MALLOC(Work, Lwork, "Work");

    double * d_S = 0;
    //### The vector for storing singular values
    CUDA_MALLOC(d_S, n_tmp, "d_S");

    double * d_U = 0;
    //### The vector for storing singular values
    CUDA_MALLOC(d_U, n_tmp * n_tmp, "d_U");

    double * d_VH = 0;
    //### The vector for storing singular values
    CUDA_MALLOC(d_VH, n_tmp * n_tmp, "d_VH");

    int * devInfo = 0;
    //### stuff
    CUDA_MALLOC(devInfo, 1, "devInfo");

    //### Actually compute the SVD
    cs_status = cusolverDnDgesvd(cs_handle, 'A', 'A', n_tmp, n_tmp, d_tmp, n_tmp, d_S, d_U, n_tmp, d_VH, n_tmp, Work, Lwork, NULL, devInfo);

    if (cs_status != 0) {
        printf("cs_status: %d\n", cs_status);
        return EXIT_FAILURE;
    }

    //float * h_S = (float *)malloc(n_tmp * sizeof(*h_S));
    //### Only copy the first (largest) singular value
    double aux;

    cudaMemcpy(&aux, d_S, 1 * sizeof(*d_S), cudaMemcpyDeviceToHost);

    CUDA_FREE(d_tmp, "d_tmp");

    CUDA_FREE(Work, "Work");

    CUDA_FREE(d_S, "d_S");

    CUDA_FREE(d_U, "d_U");

    CUDA_FREE(d_VH, "d_VH");

    CUDA_FREE(devInfo, "devInfo");

    cusolverDnDestroy(cs_handle);

    //return (la.norm(tmp, 2)/n) + mu
    return aux/(double)n + mu;
}

/**
 * No adaptive
 *
 * In this implementation the data matrix and labels vector are passed already on the device
 *
 */
int l1l2_regularization_optimized(double * d_XT, double * d_Y, double * d_XTY, int n, int p, double mu, double tau, double * d_beta_in, double * d_beta_out, int * k_final, int kmax, double tolerance, int adaptive,  cublasHandle_t handle) {

    int debug = 0;

    struct timeval t0, t1, t2;

    gettimeofday(&t0,NULL);

    cublasStatus_t status;

    //### Compute CUDA geometry once and for all
    int n_threads_per_block;
    int n_blocks;

    if (p <= THREADS_PER_BLOCK) {
        n_threads_per_block = p;
        n_blocks = 1;
    } else {
        n_threads_per_block = THREADS_PER_BLOCK;
        n_blocks = (p-1)/n_threads_per_block + 1;
    }

    dim3 threadsPerBlock(n_threads_per_block);
    dim3 numBlocks(n_blocks);
    
    //### size of vectors must be equal to the total number of threads, despite how many elements are actually useful
    int work_size = n_threads_per_block * n_blocks;
    
    //### Variables for device quantities

    double a = 1.0;
    double b = 0.0;

    //### Allocate memory on device for the main matrices used; also copy them

    gettimeofday(&t1,NULL);
    
    /************************************/
    /********ALGORITHM BEGINS HERE*******/
    /************************************/

    //### Compute sigma
    double sigma;

    //### First iteration with standard sigma
    //sigma = _sigma(data, mu)
    sigma = _sigma(d_XT, n, p, mu, handle);

    //printf("Sigma = %f\n", sigma);

    //### skipping this
    //if sigma < np.finfo(float).eps: # is zero...
    //    return np.zeros(d), 0

    double h_mu_s = mu/sigma;
    double h_tau_s = tau/(2.0 * sigma);
    double h_nsigma = n * sigma;

    //printf("h_mu_s: %f\n", h_mu_s);
    //printf("h_tau_s: %f\n", h_tau_s);
    //printf("h_nsigma: %f\n", h_nsigma);

    //### Copy on device values h_mu_s, h_tau_s, h_nsigma
    double * d_mu_s = 0;
    CUDA_MALLOC(d_mu_s, 1, "d_mu_s")
    status = cublasSetVector(1, sizeof(h_mu_s), &h_mu_s, 1, d_mu_s, 1);
    CUDA_CHECK_STATUS(status, "device access error (write d_mu_s)")

    double * d_tau_s = 0;
    CUDA_MALLOC(d_tau_s, 1, "d_tau_s")
    status = cublasSetVector(1, sizeof(h_tau_s), &h_tau_s, 1, d_tau_s, 1);
    CUDA_CHECK_STATUS(status, "device access error (write d_tau_s)")

    double * d_nsigma = 0;
    CUDA_MALLOC(d_nsigma, 1, "d_nsigma")
    status = cublasSetVector(1, sizeof(h_nsigma), &h_nsigma, 1, d_nsigma, 1);
    CUDA_CHECK_STATUS(status, "device access error (write d_nsigma)")

    //### Copy on device the beta vector
    double * d_beta = 0;
    //CUDA_MALLOC(d_beta, p, "d_beta")
    CUDA_MALLOC(d_beta, work_size, "d_beta")
    cudaMemcpy(d_beta, d_beta_in, p * sizeof(*d_beta_in), cudaMemcpyDeviceToDevice);
    CUDA_CHECK_STATUS(status, "device access error (write d_beta)")

    //# Starting conditions
    //aux_beta = beta
    //### Copy on device the aux_beta vector, intialized to beta
    double * d_aux_beta = 0;
    CUDA_MALLOC(d_aux_beta, work_size, "d_aux_beta")
    
    //status = cublasSetVector(p, sizeof(h_beta[0]), h_beta, 1, d_aux_beta, 1);
    cudaMemcpy(d_aux_beta, d_beta_in, p * sizeof(*d_beta_in), cudaMemcpyDeviceToDevice);
    //CUDA_CHECK_STATUS(status, "device access error (write d_aux_beta)")

    //### auxiliary vectors
    double * d_beta_next = 0;
    //CUDA_MALLOC(d_beta_next, p, "d_beta_next")
    CUDA_MALLOC(d_beta_next, work_size, "d_beta_next")
    
    double * d_beta_diff = 0;
    //CUDA_MALLOC(d_beta_next, p, "d_beta_next")
    CUDA_MALLOC(d_beta_diff, work_size, "d_beta_diff")

    double * d_value = 0;
    //CUDA_MALLOC(d_value, p, "d_value")
    CUDA_MALLOC(d_value, work_size, "d_value")

    double h_t = 1.0;

    double * d_t = 0;
    CUDA_MALLOC(d_t, 1, "d_t")
    status = cublasSetVector(1, sizeof(h_t), &h_t, 1, d_t, 1);
    CUDA_CHECK_STATUS(status, "device access error (write d_t)")

    double * d_t_next = 0;
    CUDA_MALLOC(d_t_next, 1, "d_t_next")

    double * d_precalc = 0;
    //CUDA_MALLOC(d_precalc, p, "d_precalc")
    CUDA_MALLOC(d_precalc, work_size, "d_precalc")

    //### Allocate more space so that it can be used for internal computations as well
    //### TODO it could be set to max(n,p)
    double * d_aux_np = 0;
    CUDA_MALLOC(d_aux_np, n+p, "d_aux_np")

    int k;

    //# Convergence values
    int maxid;
    double max_diff;
    double max_coef;
    
    //### Main loop
    for (k = 0; k < kmax; k++) {

        //printf("Iteration %d\n", k);

        if (n > p) {
            //precalc = XTY - np.dot(X.T, np.dot(X, aux_beta))

            //d_aux_np = np.dot(X, aux_beta)
            a = 1.0f;
            b = 0.0f;
            status = cublasDgemv(handle, CUBLAS_OP_T, p, n, &a, d_XT, p, d_aux_beta, 1, &b, d_aux_np, 1);
            CUDA_CHECK_STATUS(status, "Error performing operation X*aux_beta");

            //precalc = XTY
            if (cudaMemcpy(d_precalc, d_XTY, p * sizeof(*d_XTY), cudaMemcpyDeviceToDevice) != cudaSuccess) {
                fprintf(stderr, "!!!! Error copying d_XTY to d_precalc on device\n");
                return EXIT_FAILURE;
            }

            // precalc = -np.dot(X.T, d_aux_np) + precalc
            a = -1.0f;
            b = 1.0f;
            status = cublasDgemv(handle, CUBLAS_OP_N, p, n, &a, d_XT, p, d_aux_np, 1, &b, d_precalc, 1);
            CUDA_CHECK_STATUS(status, "Error computing d_precalc");

        } else {
            //precalc = np.dot(X.T, Y - np.dot(X, aux_beta))

            //d_aux_np = Y
            if (cudaMemcpy(d_aux_np, d_Y, n * sizeof(*d_Y), cudaMemcpyDeviceToDevice) != cudaSuccess) {
                fprintf(stderr, "!!!! Error copying d_Y to d_aux_np on device\n");
                return EXIT_FAILURE;
            }

            //d_aux_np = -np.dot(X, aux_beta) + d_aux_np
            a = -1.0f;
            b = 1.0f;
            status = cublasDgemv(handle, CUBLAS_OP_T, p, n, &a, d_XT, p, d_aux_beta, 1, &b, d_aux_np, 1);
            CUDA_CHECK_STATUS(status, "Error performing operation X*aux_beta");

            //d_precalc = np.dot(X.T, d_aux_np)
            a = 1.0f;
            b = 0.0f;
            status = cublasDgemv(handle, CUBLAS_OP_N, p, n, &a, d_XT, p, d_aux_np, 1, &b, d_precalc, 1);
            //cudaError_t err = cudaPeekAtLastError();
            //cout << cudaGetErrorString(err) << endl;
            CUDA_CHECK_STATUS(status, "Error performing operation X.T*d_aux_np");

        }
       
        // Invoke kernel
        SoftThreshFISTA<<<numBlocks, threadsPerBlock>>>(d_precalc, d_nsigma, d_mu_s, d_tau_s, d_beta, d_aux_beta, d_beta_next, d_beta_diff, d_value, d_t, d_t_next);
        
        //max_diff = np.abs(beta_diff).max()
        //cublasIsamax(handle, p, d_beta_diff, 1, &maxid);
        cublasIdamax(handle, p, d_beta_diff, 1, &maxid);
        CUDA_MEMCPY(&max_diff, d_beta_diff + maxid - 1, 1, cudaMemcpyDeviceToHost, "Error copying max_diff from device to host")
        max_diff = abs(max_diff);
        
        //max_coef = np.abs(beta_next).max()
        cublasIdamax(handle, p, d_beta_next, 1, &maxid);
        CUDA_MEMCPY(&max_coef, d_beta_next + maxid - 1, 1, cudaMemcpyDeviceToHost, "Error copying max_coef from device to host")
        
        max_coef = abs(max_coef);
        
        CUDA_MEMCPY(d_t, d_t_next, 1, cudaMemcpyDeviceToDevice, "update d_t")
        
        //if (k < 20) {
        //    float h_t;
        //    cudaMemcpy(&h_t, d_t, sizeof(*d_t), cudaMemcpyDeviceToHost);
        //    printf("k = %d, t = %f\n", k, h_t);
        //}

        //beta = beta_next
        cudaMemcpy(d_beta, d_beta_next, p * sizeof(*d_beta), cudaMemcpyDeviceToDevice);

        //printf("%f\n", (max_diff/max_coef));
        
        //# Stopping rule (exit even if beta_next contains only zeros)
        //if max_coef == 0.0 or (max_diff / max_coef) <= tolerance: break
        
        double cond_value = max_diff / max_coef;
        
        if (max_coef == 0 || (cond_value) <= tolerance) {
            break;
        }
        
        //if (max_coef == 0 || (max_diff / max_coef) <= tolerance) {
        //    break;
        //}

    }
    
    //printf("Final max_diff / max_coef = %f\n", (max_diff/ max_coef));
    
    //*k_final = -666;
    *k_final = k + 1;
    
    cudaMemcpy(d_beta_out, d_beta, p * sizeof(*d_beta), cudaMemcpyDeviceToDevice);

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

    /************************************/
    /*********ALGORITHM ENDS HERE********/
    /************************************/

    //### Free memory on device

    //### One of these should probably not be deallocated in the end...
    CUDA_FREE(d_aux_beta, "d_aux_beta");
    CUDA_FREE(d_beta, "d_beta");
    
    CUDA_FREE(d_beta_next, "d_beta_next");
    CUDA_FREE(d_beta_diff, "d_beta_diff");
    CUDA_FREE(d_precalc, "d_precalc");
    CUDA_FREE(d_aux_np, "d_aux_np");
    CUDA_FREE(d_mu_s, "d_mu_s");
    CUDA_FREE(d_tau_s, "d_tau_s");
    CUDA_FREE(d_nsigma, "d_nsigma");
    CUDA_FREE(d_t, "d_t");
    CUDA_FREE(d_t_next, "d_t_next");
    CUDA_FREE(d_value, "d_value");

    //printf("Exiting l1l2_regularization_optimized\n");
    
    return EXIT_SUCCESS;

}

///**
// * No adaptive
// *
// */
//int l1l2_regularization(float * XT, float * Y, int n, int p, float mu, float tau, float * beta_out, int * k_final, int kmax, float tolerance) {
//
//    //float tolerance;
//
//    int debug = 1;
//
//    struct timeval t0, t1, t2;
//
//    gettimeofday(&t0,NULL);
//
//    //### Default values for parameters
//
//    //tolerance = 0.00001;
//    //kmax=10000;
//
//    cublasStatus_t status;
//    cublasHandle_t handle;
//
//    //int debug = 1;
//
//    //int dev = findCudaDevice(1, (const char **) NULL);
//
//    //if (dev == -1)
//    //{
//        //return EXIT_FAILURE;
//    //}
//
//    //for (int i = 0; i < 20; i++) {
//        //printf("%f\n", XT[i]);
//    //}
//
//    status = cublasCreate(&handle);
//
//    //### Compute CUDA geometry once and for all
//    int n_threads_per_block;
//    int n_blocks;
//
//    if (p <= THREADS_PER_BLOCK) {
//        n_threads_per_block = p;
//        n_blocks = 1;
//    } else {
//        n_threads_per_block = THREADS_PER_BLOCK;
//        n_blocks = (p-1)/n_threads_per_block + 1;
//    }
//
//    dim3 threadsPerBlock(n_threads_per_block);
//    dim3 numBlocks(n_blocks);
//
//    //### size of vectors must be equal to the total number of threads, despite how many elements are actually useful
//    int work_size = n_threads_per_block * n_blocks;
//
//    //printf("threads_per_blocks: %d\n", n_threads_per_block);
//    //printf("n blocks: %d\n", n_blocks);
//
//    //### Variables for device quantities
//    float * d_XT = 0; //### pointer to the device copy of matrix X
//    float * d_Y = 0; //### pointer to the device copy of matrix Y
//    float * d_XTY = 0; //### pointer to the device copy of matrix XTY
//
//    float * h_XTY = 0;
//
//    float a = 1.0f;
//    float b = 0.0f;
//
//    h_XTY = (float *)malloc(p*sizeof(float));
//
//    //### Initialize beta
//    //### TODO allow beta to be passed as a parameter for warm start
//    float * h_beta;
//    h_beta = (float *)calloc(p, sizeof(float));
//
//    //### Allocate memory on device for the main matrices used; also copy them
//
//    //### The data matrix X (transposed)
//    CUDA_MALLOC(d_XT, n*p, "d_XT");
//
//    /* Initialize the device matrices with the host matrices */
//    status = cublasSetMatrix(p, n, sizeof(XT[0]), XT, p, d_XT , p);
//    CUDA_CHECK_STATUS(status, "device access error (write XT)");
//
//    //### The labels vector Y
//    CUDA_MALLOC(d_Y, n, "d_Y");
//
//    /* Initialize the device matrices with the host matrices */
//    status = cublasSetVector(n, sizeof(Y[0]), Y, 1, d_Y, 1);
//    CUDA_CHECK_STATUS(status, "device access error (write Y)");
//
//    //### The product XTY
//    CUDA_MALLOC(d_XTY, p, "d_XTY");
//
//    gettimeofday(&t1,NULL);
//
//    /************************************/
//    /********ALGORITHM BEGINS HERE*******/
//    /************************************/
//
//    ///### Compute X.T.dot(Y)
//    if (n > p) {
//        status = cublasSgemv(handle, CUBLAS_OP_N, p, n, &a, d_XT, p, d_Y, 1, &b, d_XTY, 1);
//        CUDA_CHECK_STATUS(status, "Error performing operation XTY");
//    }
//
//    //### Compute sigma
//    float sigma;
//
//    //### First iteration with standard sigma
//    //sigma = _sigma(data, mu)
//    sigma = _sigma(d_XT, n, p, mu, handle);
//
//    //printf("Sigma = %f\n", sigma);
//
//    //### skipping this
//    //if sigma < np.finfo(float).eps: # is zero...
//    //    return np.zeros(d), 0
//
//    float h_mu_s = mu/sigma;
//    float h_tau_s = tau/(2.0 * sigma);
//    float h_nsigma = n * sigma;
//
//    //printf("h_mu_s: %f\n", h_mu_s);
//    //printf("h_tau_s: %f\n", h_tau_s);
//    //printf("h_nsigma: %f\n", h_nsigma);
//
//    //### Copy on device values h_mu_s, h_tau_s, h_nsigma
//    float * d_mu_s = 0;
//    CUDA_MALLOC(d_mu_s, 1, "d_mu_s")
//    status = cublasSetVector(1, sizeof(h_mu_s), &h_mu_s, 1, d_mu_s, 1);
//    CUDA_CHECK_STATUS(status, "device access error (write d_mu_s)")
//
//    float * d_tau_s = 0;
//    CUDA_MALLOC(d_tau_s, 1, "d_tau_s")
//    status = cublasSetVector(1, sizeof(h_tau_s), &h_tau_s, 1, d_tau_s, 1);
//    CUDA_CHECK_STATUS(status, "device access error (write d_tau_s)")
//
//    float * d_nsigma = 0;
//    CUDA_MALLOC(d_nsigma, 1, "d_nsigma")
//    status = cublasSetVector(1, sizeof(h_nsigma), &h_nsigma, 1, d_nsigma, 1);
//    CUDA_CHECK_STATUS(status, "device access error (write d_nsigma)")
//
//    //### Copy on device the beta vector
//    float * d_beta = 0;
//    //CUDA_MALLOC(d_beta, p, "d_beta")
//    CUDA_MALLOC(d_beta, work_size, "d_beta")
//
//    status = cublasSetVector(p, sizeof(h_beta[0]), h_beta, 1, d_beta, 1);
//    CUDA_CHECK_STATUS(status, "device access error (write d_beta)")
//
//    //# Starting conditions
//    //aux_beta = beta
//    //### Copy on device the aux_beta vector, intialized to beta
//    float * d_aux_beta = 0;
//    //CUDA_MALLOC(d_aux_beta, p, "d_aux_beta")
//    CUDA_MALLOC(d_aux_beta, work_size, "d_aux_beta")
//    status = cublasSetVector(p, sizeof(h_beta[0]), h_beta, 1, d_aux_beta, 1);
//    CUDA_CHECK_STATUS(status, "device access error (write d_aux_beta)")
//
//    //### auxiliary vectors
//    float * d_beta_next = 0;
//    CUDA_MALLOC(d_beta_next, work_size, "d_beta_next")
//    
//    float * d_beta_diff = 0;
//    CUDA_MALLOC(d_beta_diff, work_size, "d_beta_diff")
//
//    float * d_value = 0;
//    //CUDA_MALLOC(d_value, p, "d_value")
//    CUDA_MALLOC(d_value, work_size, "d_value")
//
//    float h_t = 1.0f;
//
//    float * d_t = 0;
//    CUDA_MALLOC(d_t, 1, "d_t")
//    status = cublasSetVector(1, sizeof(h_t), &h_t, 1, d_t, 1);
//    CUDA_CHECK_STATUS(status, "device access error (write d_t)")
//
//    float * d_t_next = 0;
//    CUDA_MALLOC(d_t_next, 1, "d_t_next")
//
//    float * d_precalc = 0;
//    //CUDA_MALLOC(d_precalc, p, "d_precalc")
//    CUDA_MALLOC(d_precalc, work_size, "d_precalc")
//
//    //### Allocate more space so that it can be used for internal computations as well
//    //### TODO it could be set to max(n,p)
//    float * d_aux_np = 0;
//    CUDA_MALLOC(d_aux_np, n+p, "d_aux_np")
//
//    int k;
//    
//    //# Convergence values
//    int maxid;
//    float max_diff;
//    float max_coef;
//
//    //### Main loop
//    for (k = 0; k < kmax; k++) {
//
//        //printf("Iteration %d\n", k);
//
//        if (n > p) {
//            //precalc = XTY - np.dot(X.T, np.dot(X, aux_beta))
//
//            //d_aux_np = np.dot(X, aux_beta)
//            a = 1.0f;
//            b = 0.0f;
//            status = cublasSgemv(handle, CUBLAS_OP_T, p, n, &a, d_XT, p, d_aux_beta, 1, &b, d_aux_np, 1);
//            CUDA_CHECK_STATUS(status, "Error performing operation X*aux_beta");
//
//            //precalc = XTY
//            if (cudaMemcpy(d_precalc, d_XTY, p * sizeof(*d_XTY), cudaMemcpyDeviceToDevice) != cudaSuccess) {
//                fprintf(stderr, "!!!! Error copying d_XTY to d_precalc on device\n");
//                return EXIT_FAILURE;
//            }
//
//            // precalc = -np.dot(X.T, d_aux_np) + precalc
//            a = -1.0f;
//            b = 1.0f;
//            status = cublasSgemv(handle, CUBLAS_OP_N, p, n, &a, d_XT, p, d_aux_np, 1, &b, d_precalc, 1);
//            CUDA_CHECK_STATUS(status, "Error computing d_precalc");
//
//        } else {
//            //precalc = np.dot(X.T, Y - np.dot(X, aux_beta))
//
//            //d_aux_np = Y
//            if (cudaMemcpy(d_aux_np, d_Y, n * sizeof(*d_Y), cudaMemcpyDeviceToDevice) != cudaSuccess) {
//                fprintf(stderr, "!!!! Error copying d_Y to d_aux_np on device\n");
//                return EXIT_FAILURE;
//            }
//
//            //d_aux_np = -np.dot(X, aux_beta) + d_aux_np
//            a = -1.0f;
//            b = 1.0f;
//            status = cublasSgemv(handle, CUBLAS_OP_T, p, n, &a, d_XT, p, d_aux_beta, 1, &b, d_aux_np, 1);
//            CUDA_CHECK_STATUS(status, "Error performing operation X*aux_beta");
//
//            //d_precalc = np.dot(X.T, d_aux_np)
//            a = 1.0f;
//            b = 0.0f;
//            status = cublasSgemv(handle, CUBLAS_OP_N, p, n, &a, d_XT, p, d_aux_np, 1, &b, d_precalc, 1);
//            //cudaError_t err = cudaPeekAtLastError();
//            //cout << cudaGetErrorString(err) << endl;
//            CUDA_CHECK_STATUS(status, "Error performing operation X.T*d_aux_np");
//
//        }
//
//        //### TODO to remove
//        //cudaMemcpy(beta_out, d_precalc, p * sizeof(*d_precalc), cudaMemcpyDeviceToHost);
//
//        //printf("Printing the first 20 values of beta\n");
//        //for (int ii = 0; ii < 20; ii++) {
//            //printf("beta[%d] = %f\n", ii, beta_out[ii]);
//        //}
//
//        // Invoke kernel
//        SoftThreshFISTA<<<numBlocks, threadsPerBlock>>>(d_precalc, d_nsigma, d_mu_s, d_tau_s, d_beta, d_aux_beta, d_beta_next, d_beta_diff, d_value, d_t, d_t_next);
//
//        //### Synchronization maybe?
//        //if (cudaDeviceSynchronize() != cudaSuccess){
//            //printf("Error in synch!\n");
//        //}
//
//        //# Convergence values
//        //max_diff = np.abs(beta_diff).max()
//        //max_coef = np.abs(beta_next).max()
//
//        //# Values update
//        //t = t_next
//        //float t_next = 0.5 * (1.0 + sqrt(1.0 + 4.0 * (t*t)));
//
//        //h_t = 0.5 * (1.0 + sqrt(1.0 + 4.0 * (h_t*h_t)));
//        ////cudaMemcpy(d_t, d_t_next, sizeof(*d_t), cudaMemcpyDeviceToDevice);
//        //cudaMemcpy(d_t, &h_t, sizeof(*d_t), cudaMemcpyHostToDevice);
//
//        //max_diff = np.abs(beta_diff).max()
//        cublasIsamax(handle, p, d_beta_diff, 1, &maxid);
//        CUDA_MEMCPY(&max_diff, d_beta_diff + maxid - 1, 1, cudaMemcpyDeviceToHost, "Error copying max_diff from device to host")
//        max_diff = abs(max_diff);
//        
//        //max_coef = np.abs(beta_next).max()
//        cublasIsamax(handle, p, d_beta_next, 1, &maxid);
//        CUDA_MEMCPY(&max_coef, d_beta_next + maxid - 1, 1, cudaMemcpyDeviceToHost, "Error copying max_coef from device to host")
//        
//        max_coef = abs(max_coef);
//        
//        CUDA_MEMCPY(d_t, d_t_next, 1, cudaMemcpyDeviceToDevice, "update d_t")
//
//        //beta = beta_next
//        cudaMemcpy(d_beta, d_beta_next, p * sizeof(*d_beta), cudaMemcpyDeviceToDevice);
//
//        //# Stopping rule (exit even if beta_next contains only zeros)
//        //if max_coef == 0.0 or (max_diff / max_coef) <= tolerance: break
//        
//        double cond_value = max_diff / max_coef;
//        
//        if (max_coef == 0 || (cond_value) <= tolerance) {
//            break;
//        }
//
//    }
//    
//    *k_final = k + 1;
//
//    //printf("Copying results back to host memory...\n");
//    cudaMemcpy(beta_out, d_beta, p * sizeof(*d_beta), cudaMemcpyDeviceToHost);
//
//    //for (int i = 0; i < 20; i++) {
//        //printf("%f\n", beta_out[i]);
//    //}
//
//    gettimeofday(&t2,NULL);
//
//    //cudaMemcpy(beta_out, d_beta_next, p * sizeof(*d_beta_next), cudaMemcpyDeviceToHost);
//
//    //### TODO to remove
//    //printf("Printing the first 20 values of beta\n");
//    //for (int ii = 0; ii < 20; ii++) {
//        //printf("beta[%d] = %f\n", ii, beta_out[ii]);
//    //}
//
//    if (debug) {
//        double init_time, comp_time, total_time;
//
//        init_time = (t1.tv_sec-t0.tv_sec)*1000000;
//        init_time += (t1.tv_usec-t0.tv_usec);
//    
//        comp_time = (t2.tv_sec-t1.tv_sec)*1000000;
//        comp_time += (t2.tv_usec-t1.tv_usec);
//    
//        total_time = (t2.tv_sec-t0.tv_sec)*1000000;
//        total_time += (t2.tv_usec-t0.tv_usec);
//        printf("Initialization time: %lf\n", init_time/1000000.0);
//        printf("Computation time: %lf\n", comp_time/1000000.0);
//        printf("Total time: %lf\n", total_time/1000000.0);
//    }
//
//    //printf("%d %lf %lf\n", p, init_time/1000000.0, comp_time/1000000.0);
//
//    /************************************/
//    /*********ALGORITHM ENDS HERE********/
//    /************************************/
//
//    //### Free memory on device
//
//    CUDA_FREE(d_XT, "d_XT");
//    CUDA_FREE(d_Y, "d_Y");
//    CUDA_FREE(d_XTY, "d_XTY");
//    CUDA_FREE(d_beta, "d_beta");
//    CUDA_FREE(d_aux_beta, "d_aux_beta");
//    
//    CUDA_FREE(d_beta_diff, "d_beta_diff");
//    
//    CUDA_FREE(d_beta_next, "d_beta_next");
//    CUDA_FREE(d_precalc, "d_precalc");
//    CUDA_FREE(d_aux_np, "d_aux_np");
//    CUDA_FREE(d_mu_s, "d_mu_s");
//    CUDA_FREE(d_tau_s, "d_tau_s");
//    CUDA_FREE(d_nsigma, "d_nsigma");
//    CUDA_FREE(d_t, "d_t");
//    CUDA_FREE(d_t_next, "d_t_next");
//    CUDA_FREE(d_value, "d_value");
//
//    cublasDestroy(handle);
//
//    free(h_beta);
//    free(h_XTY);
//
//    //return h_beta;
//
//    return EXIT_SUCCESS;
//
//}
//
//extern "C" {
//int l1l2_regularization_bridge(float * XT, float * Y, int n, int p, float mu, float tau, float * beta_out, int * k_final, int kmax, float tolerance) {
//    return l1l2_regularization(XT, Y, n, p, mu, tau, beta_out, k_final, kmax, tolerance);
//}
//}
//
