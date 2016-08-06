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


//### Python CODE
    //def l1l2_path(data, labels, mu, tau_range, beta=None, kmax=100000,
    //          tolerance=1e-5, adaptive=False):
    //r"""Efficient solution of different `l1l2` regularization problems on
    //increasing values of the `l1-norm` parameter.
    //
    //Finds the `l1l2` regularization path for each value in ``tau_range`` and
    //fixed value of ``mu``.
    //
    //The values in ``tau_range`` are used during the computation in reverse
    //order, while the output path has the same ordering of the `tau` values.
    //
    //.. note ::
    //
    //    For efficency purposes, if ``mu = 0.0`` and the number of non-zero
    //    values is higher than `N` for a given value of tau (that means algorithm
    //    has reached the limit of allowed iterations), the following solutions
    //    (for smaller values of ``tau``) are simply the least squares solutions.
    //
    //.. warning ::
    //
    //    The number of solutions can differ from ``len(tau_range)``.
    //    The function returns only the solutions with at least one non-zero
    //    element.
    //    For values higher than *tau_max* a solution have all zero values.
    //
    //Parameters
    //----------
    //data : (N, P) ndarray
    //    Data matrix.
    //labels : (N,) or (N, 1) ndarray
    //    Labels vector.
    //mu : float
    //    `l2-norm` penalty.
    //tau_range : array_like of float
    //    `l1-norm` penalties in increasing order.
    //beta : (P,) or (P, 1) ndarray, optional (default is `None`)
    //    Starting value of the iterations.
    //    If `None`, then iterations starts from the empty model.
    //kmax : int, optional (default is `1e5`)
    //    Maximum number of iterations.
    //tolerance : float, optional (default is `1e-5`)
    //    Convergence tolerance.
    //adaptive : bool, optional (default is `False`)
    //    If `True`, minimization is performed calculating an adaptive step size
    //    for each iteration.
    //
    //Returns
    //-------
    //beta_path : list of (P,) or (P, 1) ndarrays
    //    `l1l2` solutions with at least one non-zero element.
    //
    //"""
    //from collections import deque
    //n, p = data.shape
    //
    //if mu == 0.0:
    //    beta_ls = ridge_regression(data, labels)
    //
    //if beta is None:
    //    beta = np.zeros((p, 1))
    //
    //out = deque()
    //nonzero = 0
    //for tau in reversed(tau_range):
    //    if mu == 0.0 and nonzero >= n: # lasso saturation
    //        beta_next = beta_ls
    //    else:
    //        beta_next = l1l2_regularization(data, labels, mu, tau, beta,
    //                                        kmax, tolerance, adaptive=adaptive)
    //
    //    nonzero = len(beta_next.nonzero()[0])
    //    if nonzero > 0:
    //        out.appendleft(beta_next)
    //
    //    beta = beta_next
    //
    //return out

/**
 *
 *
 * Parameters
 * ----------
 *
 * float * h_out: a $n_tau \times p$ array where each row represents one of the betas computed
 *
 */
int l1l2_path(float * h_XT, float * h_Y, int n, int p, float mu, float * h_tau_range, int n_tau, float * h_beta, float * h_out, int * n_betas_out, int * k_final, int kmax, float tolerance, int adaptive) {

    //int debug = 1;
    int debug = 0;

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
        printf("beta is not initialized, creating a dummy one.\n");
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

    //### iterator for taus
    int z;
    float tau;
    int n_betas = 0;

    int bb; //### used to check for an all zeros solution

    //### Use taus in decreasing order, therefore going from more sparse to less sparse models
    for (z = n_tau -1; z >= 0; z--) {

        tau = h_tau_range[z];

        //cout << "tau = " << tau << endl;

        //### Again, we assume that mu is always != 0
        //if mu == 0.0 and nonzero >= n: # lasso saturation
        //beta_next = beta_ls
        //else:

        l1l2_regularization_optimized(d_XT, d_Y, d_XTY, n, p, mu, tau, d_beta_in, d_beta_out, k_final, kmax, tolerance, adaptive, handle);

        //printf("K final (CUDA) = %d\n", *k_final);

        //### After each iteration, if the solution is not ony zeros, copy the results from device back to host
        //CUDA_MEMCPY(h_out + (n_tau - z - 1) * p, d_beta_out, p, cudaMemcpyDeviceToHost, "Error copying d_beta_out from device to host")
        CUDA_MEMCPY(h_out + z * p, d_beta_out, p, cudaMemcpyDeviceToHost, "Error copying d_beta_out from device to host")

        //### Check if the solution only has zeros
        for (bb = z * p; bb < z * p + p; bb++) {
            if (h_out[bb] != 0) {
                break;
            }
        }

        if (bb < (z * p + p)) {
            n_betas++;
        }

        CUDA_MEMCPY(d_beta_in, d_beta_out, p, cudaMemcpyDeviceToDevice, "Error copying d_beta_in from device to device")

    }

    *n_betas_out = n_betas;

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

    //nonzero = 0
    //for tau in reversed(tau_range):
    //    if mu == 0.0 and nonzero >= n: # lasso saturation
    //        beta_next = beta_ls
    //    else:
    //        beta_next = l1l2_regularization(data, labels, mu, tau, beta,
    //                                        kmax, tolerance, adaptive=adaptive)
    //
    //    nonzero = len(beta_next.nonzero()[0])
    //    if nonzero > 0:
    //        out.appendleft(beta_next)
    //
    //    beta = beta_next
    //
    //return out
}

extern "C" {

int l1l2_path_bridge(float * h_XT, float * h_Y, int n, int p, float mu, float * h_tau_range, int n_tau, float * h_beta, float * h_out, int * n_betas_out, int * k_final, int kmax, float tolerance, int adaptive) {

    return l1l2_path(h_XT, h_Y, n, p, mu, h_tau_range, n_tau, h_beta, h_out, n_betas_out, k_final, kmax, tolerance, adaptive);
}
}
