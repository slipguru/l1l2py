/* Includes, system */
//#include <stdio.h>
#include <iostream>

#include <fstream>

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
#include "csv.h"
#include "utils.h"
#include "l1l2_regularization.h"

//### EXAMPLES:
//CUDA_MALLOC(d_A, n, "d_A");
//CUDA_FREE(d_A, "d_A");

/* Matrix size */
//#define N  (2000)

//#define THREADS_PER_BLOCK 200
#define THREADS_PER_BLOCK 500

using namespace std;

// int l1l2_path(float * h_XT, float * h_Y, int n, int p, float mu, float * h_tau_range, int n_tau, float * h_beta, float * h_out, int * n_betas_out, int * k_final, int kmax, float tolerance, int adaptive) {
int test_l1l2regularization_opt(char * buffer_X, char * buffer_Y){
    int debug = 1;
    //int debug = 0;

    cout << "\nX:" << buffer_X << endl;
    cout << "Y:" << buffer_Y << endl;

    // ex-l1l2path input arguments
    int n = 1000;
    int p = 4000;

    float * h_XT;
    float * h_Y;
    float * h_beta;

    int kmax = 10000;
    float tolerance = 1e-5f;
    int adaptive = 0;
    // float * h_tau_range;
    // int n_tau;

    float * h_out;
    int * n_betas_out;
    // int * k_final;
    int k_final;

    // Get input data
    // char buffer_X[50];
    // char buffer_Y[50];

    // sprintf(buffer_X, "data_c/X_%d_%d.csv", n,p);
    // sprintf(buffer_Y, "data_c/Y_%d_%d.csv", n,p);

    h_XT = (float *)malloc(n*p*sizeof(float));
    h_Y = (float *)malloc(n*sizeof(float));
    h_beta = (float *)malloc(p*sizeof(float));
    h_out = (float *)malloc(p*sizeof(float));

    read_csv(buffer_X, n, p, h_XT);
    read_csv(buffer_Y, n, 1, h_Y);


    struct timeval t0, t1, t2;

    gettimeofday(&t0,NULL);

    //printf("Tolerance: %f\n", tolerance);

    cublasHandle_t handle;
    cublasStatus_t status;

    status = cublasCreate(&handle);

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

    float a = 1.0f;
    float b = 0.0f;

    dim3 threadsPerBlock(n_threads_per_block);
    dim3 numBlocks(n_blocks);

    //### size of vectors must be equal to the total number of threads, despite how many elements are actually useful
    int work_size = n_threads_per_block * n_blocks;

    //### Variables for device quantities
    float * d_XT = 0; //### pointer to the device copy of matrix X
    float * d_Y = 0; //### pointer to the device copy of matrix Y
    float * d_XTY = 0; //### pointer to the device copy of matrix XTY
    float * d_beta_in = 0;
    float * d_beta_out = 0;

    //### The data matrix X (transposed)
    CUDA_MALLOC(d_XT, n*p, "d_XT");

    /* Initialize the device matrices with the host matrices */
    status = cublasSetMatrix(p, n, sizeof(h_XT[0]), h_XT, p, d_XT , p);
    CUDA_CHECK_STATUS(status, "device access error (write XT)");

    //### The labels vector Y
    CUDA_MALLOC(d_Y, n, "d_Y");

    /* Initialize the device matrices with the host matrices */
    status = cublasSetVector(n, sizeof(h_Y[0]), h_Y, 1, d_Y, 1);
    CUDA_CHECK_STATUS(status, "device access error (write Y)");

    //### The product XTY
    CUDA_MALLOC(d_XTY, p, "d_XTY");

    ///### Compute X.T.dot(Y)
    if (n > p) {
        status = cublasSgemv(handle, CUBLAS_OP_N, p, n, &a, d_XT, p, d_Y, 1, &b, d_XTY, 1);
        CUDA_CHECK_STATUS(status, "Error performing operation XTY");
    }

    //### The in beta and out beta
    CUDA_MALLOC(d_beta_in, work_size, "d_beta_in");
    CUDA_MALLOC(d_beta_out, work_size, "d_beta_out");

    //### Assume the beta is not passed if it is a null pointer
    if (h_beta == 0) {
        // printf("beta is not initialized, creating a dummy one.\n");
        //### If no beta is passed, just allocate zeros and copy them on memory
        float * h_beta_tmp = (float *)calloc(work_size, sizeof(float));
        status = cublasSetVector(p, sizeof(h_beta_tmp[0]), h_beta_tmp, 1, d_beta_in, 1);
        CUDA_CHECK_STATUS(status, "device access error (copy beta_in)");
        free(h_beta_tmp);
    } else {
        status = cublasSetVector(p, sizeof(h_beta[0]), h_beta, 1, d_beta_in, 1);
        CUDA_CHECK_STATUS(status, "device access error (copy beta_in)");
    }

    gettimeofday(&t1,NULL);

    //### This part is not implemented for now, assume mu is always != 0
    //if mu == 0.0:
    //    beta_ls = ridge_regression(data, labels)
    //

    // //### iterator for taus
    // int z;
    // float tau;
    // int n_betas = 0;

    // int bb; //### used to check for an all zeros solution

    //cout << "tau = " << tau << endl;

    //### Again, we assume that mu is always != 0
    //if mu == 0.0 and nonzero >= n: # lasso saturation
    //beta_next = beta_ls
    //else:
    float mu = 1e-26f;
    float tau = 1e-1f;

    l1l2_regularization_optimized(d_XT, d_Y, d_XTY, n, p, mu, tau, d_beta_in, d_beta_out, &k_final, kmax, tolerance, adaptive, handle);

    //printf("K final (CUDA) = %d\n", *k_final);

    // //### After each iteration, if the solution is not ony zeros, copy the results from device back to host
    // //CUDA_MEMCPY(h_out + (n_tau - z - 1) * p, d_beta_out, p, cudaMemcpyDeviceToHost, "Error copying d_beta_out from device to host")
    // CUDA_MEMCPY(h_out + z * p, d_beta_out, p, cudaMemcpyDeviceToHost, "Error copying d_beta_out from device to host")
    CUDA_MEMCPY(h_out, d_beta_out, p, cudaMemcpyDeviceToHost, "Error copying d_beta_out from device to host")

    //for (int i=0; i < 20; i++){
    //    cout << h_out[i] << endl;
    //}

    //
    // //### Check if the solution only has zeros
    // for (bb = z * p; bb < z * p + p; bb++) {
    //     if (h_out[bb] != 0) {
    //         break;
    //     }
    // }
    //
    // if (bb < (z * p + p)) {
    //     n_betas++;
    // }
    //
    // CUDA_MEMCPY(d_beta_in, d_beta_out, p, cudaMemcpyDeviceToDevice, "Error copying d_beta_in from device to device")
    //
    //
    // *n_betas_out = n_betas;


    gettimeofday(&t2,NULL);

    if (debug) {
        double init_time, comp_time, total_time;

        init_time = (t1.tv_sec-t0.tv_sec)*1000000;
        init_time += (t1.tv_usec-t0.tv_usec);

        comp_time = (t2.tv_sec-t1.tv_sec)*1000000;
        comp_time += (t2.tv_usec-t1.tv_usec);

        total_time = (t2.tv_sec-t0.tv_sec)*1000000;
        total_time += (t2.tv_usec-t0.tv_usec);
        printf("Initialization time: %lf\n", init_time/1000000.0);
        printf("Computation time: %lf\n", comp_time/1000000.0);
        printf("Total time: %lf\n", total_time/1000000.0);
    }

    CUDA_FREE(d_XT, "d_XT");
    CUDA_FREE(d_Y, "d_Y");
    CUDA_FREE(d_XTY, "d_XTY");
    CUDA_FREE(d_beta_in, "d_beta_in");
    CUDA_FREE(d_beta_out, "d_beta_out");

    cublasDestroy(handle);

    return EXIT_SUCCESS;
}

// extern "C" {
//
// int l1l2_path_bridge(float * h_XT, float * h_Y, int n, int p, float mu, float * h_tau_range, int n_tau, float * h_beta, float * h_out, int * n_betas_out, int * k_final, int kmax, float tolerance, int adaptive) {
//
//     return l1l2_path(h_XT, h_Y, n, p, mu, h_tau_range, n_tau, h_beta, h_out, n_betas_out, k_final, kmax, tolerance, adaptive);
// }
// }

int main(int argc, char ** argv) {

    //### L1L2Regularization
    // int p = atoi(argv[1]);
    char * fileX = argv[1];
    char * fileY = argv[2];

    test_l1l2regularization_opt(fileX, fileY);

    cout << "end" << endl;
}
