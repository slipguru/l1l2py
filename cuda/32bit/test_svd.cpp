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

#include <cusolverDn.h> // Cusolver for SVD and stuff
#include <cusolver_common.h> // Cusolver for SVD and stuff

using namespace std;

/**
 *
 * d_matrixT : the matrix, transposed, on the device; the shape of the original matrix is n X p, so the shape of the transposed matrix is p X n
 * n
 *
 * ###TODO stub
 */
int main() {
    
    //float * d_tmp = 0;
    int n2 = 4;

    float alpha = 1.0f;
    float beta = 0.0f;

    int n_tmp = 2;
    
    int Lwork; //### size for the work buffer for the SVD computation
    
    cublasStatus_t status;
    cublasHandle_t handle;
    
    cusolverStatus_t cs_status;
    cusolverDnHandle_t cs_handle;
    
    int dev = findCudaDevice(1, (const char **) NULL);
    if (dev == -1)
    {
        return EXIT_FAILURE;
    }

    status = cublasCreate(&handle);
    cs_status = cusolverDnCreate(&cs_handle);
    
    //### Allocate memory for host A
    float * h_A = (float *)malloc(n2*sizeof(float));
    
    //### Fill host matrix with values, column major order
    // [[ 1 -2]
    // [ 3  5]]
    h_A[0] = 1;
    h_A[1] = 3;
    h_A[2] = -2;
    h_A[3] = 5;
    
    float * d_A = 0;
    //### Allocate memory for matrix A on device
    if (cudaMalloc((void **)&d_A, n2 * sizeof(d_A[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate S)\n");
        return EXIT_FAILURE;
    }
    
    //### Copy on device matrix A
    status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write X)\n");
        return EXIT_FAILURE;
    }
    
    //### Compute buffer size for SVD
    cusolverDnSgesvd_bufferSize(cs_handle, 2, 2, &Lwork);
    
    printf("Lwork: %d\n", Lwork);
    
    float * Work = 0;
    //### The temp matrix Work
    if (cudaMalloc((void **)&Work, Lwork * sizeof(Work[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate Work)\n");
        return EXIT_FAILURE;
    }
    
    float * d_S = 0;
    //### The vector for storing singular values
    if (cudaMalloc((void **)&d_S, n_tmp * sizeof(d_S[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate d_S)\n");
        return EXIT_FAILURE;
    }
    
    float * d_U = 0;
    //### The vector for storing singular values
    if (cudaMalloc((void **)&d_U, n_tmp * n_tmp * sizeof(d_U[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate d_U)\n");
        return EXIT_FAILURE;
    }
    
    float * d_VH = 0;
    //### The vector for storing singular values
    if (cudaMalloc((void **)&d_VH, n_tmp * n_tmp * sizeof(d_VH[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate d_VH)\n");
        return EXIT_FAILURE;
    }
    
    int * devInfo = 0;
    //### stuff
    if (cudaMalloc((void **)&devInfo, sizeof(*devInfo)) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate devInfo)\n");
        return EXIT_FAILURE;
    }
    
    printf("cs_status: %d\n", cs_status);
    
    float * rwork = (float *)malloc(5*n_tmp * sizeof(float));
    
    cs_status = cusolverDnSgesvd(cs_handle, 'A', 'A', n_tmp, n_tmp, d_A, n_tmp, d_S, d_U, n_tmp, d_VH, n_tmp, Work, Lwork, NULL, devInfo);
    //cs_status = cusolverDnSgesvd(cs_handle, 'A', 'A', n_tmp, n_tmp, d_A, n_tmp, d_S, d_U, n_tmp, d_VH, n_tmp, Work, Lwork, rwork, devInfo);
    
    //cudaError_t err = cudaPeekAtLastError();
    //cout << cudaGetErrorString(err);
    
    free(rwork);
    
    //printf("devinfo: %d\n", devInfo);
    printf("cs_status: %d\n", cs_status);
    
    float * h_S = (float *)malloc(n_tmp * sizeof(*h_S));
    
    cudaMemcpy(h_S, d_S, n_tmp * sizeof(*d_S), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n_tmp; i++) {
        printf("%f \n", h_S[i]);
    }
    
    
    if (cudaFree(Work) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (Work)\n");
        return EXIT_FAILURE;
    }
    
    if (cudaFree(d_S) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (d_S)\n");
        return EXIT_FAILURE;
    }
    
    if (cudaFree(d_U) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (d_U)\n");
        return EXIT_FAILURE;
    }
    
    if (cudaFree(d_VH) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (d_VH)\n");
        return EXIT_FAILURE;
    }
    
    if (cudaFree(d_A) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (d_A)\n");
        return EXIT_FAILURE;
    }
    
    if (cudaFree(devInfo) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (devInfo)\n");
        return EXIT_FAILURE;
    }
    
    
    free(h_A);
    free(h_S);
    
    cublasDestroy(handle);
    cusolverDnDestroy(cs_handle);
    
    return EXIT_SUCCESS;
}


//#include <cstdio>
//#include <cstdlib>
//#include <iostream>
//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <cusolverDn.h>
//
//using namespace std;
//
//int main()
//{
//
//    int M = 1000;
//    int N = 1000;
//
//    float * A = (float *)malloc( M * N * sizeof(*A) );
//    for( int i = 0; i < M; i++ )
//    {
//        for( int j = 0; j < N; j++ )
//        {
//            A[ j * M + i ] = ( i + j ) * ( i + j );
//        }
//    }
//
//    float * devA;         
//    cudaMalloc( &devA ,  M * N * sizeof(*devA) );
//
//    float * S = (float *)malloc( M *     sizeof(*S) );
//    float * U = (float *)malloc( M * M * sizeof(*U) );
//    float * V = (float *)malloc( N * N * sizeof(*V) );
//
//    int WorkSize = M * M;
//
//    int * devInfo;
//    cudaMalloc( &devInfo, sizeof(*devInfo) );
//    float * devS;
//    cudaMalloc( &devS, M * sizeof(*devS) );
//    float * devU;
//    cudaMalloc( &devU,M * M * sizeof(*devU) );
//    float * devV;
//    cudaMalloc( &devV, N * N * sizeof(*devV) );
//    
//
//    cusolverStatus_t cuSolverStatus;
//    cusolverDnHandle_t cuSolverHandle;
//    cusolverDnCreate( &cuSolverHandle );
//
//    cuSolverStatus = cusolverDnSgesvd_bufferSize( cuSolverHandle, M, N, &WorkSize );
//
//    float * Work;   
//    cudaMalloc( &Work, WorkSize * sizeof(*Work) );
//    float * rwork;   
//    cudaMalloc( &rwork, M * M * sizeof(*rwork) );
//
//    cudaMemcpy( devA, A, M * N * sizeof(*A), cudaMemcpyHostToDevice );
//        
//    cuSolverStatus = cusolverDnSgesvd( cuSolverHandle, 'A', 'A', M, N, devA, M, devS, devU, M, devV, N, Work, WorkSize, NULL, devInfo );
//    cudaPeekAtLastError();
//    cudaDeviceSynchronize();
//	    
//    cudaMemcpy( S, devS, M * sizeof(*devS), cudaMemcpyDeviceToHost );
//
//    for( int i = 0; i < N; i++ )
//        cout << "S[ " << i << " ] = " << S[ i ] << endl;
//
//    cusolverDnDestroy( cuSolverHandle );
//    cudaDeviceReset();
//    
//    return 0;
//
//}