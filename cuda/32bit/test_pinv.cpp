#include <iostream>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

#include <cusolverDn.h> // Cusolver for SVD and stuff
#include <cusolver_common.h> // Cusolver for SVD and stuff

#include "utils.h"
#include "pinv.h"

using namespace std;

int main() {
    
    int n = 3;
    int n2 = n*n;
    
    cublasHandle_t handle;
    cublasStatus_t status;
    
    status = cublasCreate(&handle);
    
    cusolverDnHandle_t cs_handle;
    cusolverDnCreate(&cs_handle);
    
    int dev = findCudaDevice(1, (const char **) NULL);

    if (dev == -1)
    {
        return EXIT_FAILURE;
    }
    
    float * h_X;
    
    h_X = (float *)malloc(n2 * sizeof(float));
    
    //for (int i = 0; i < n2; i++) {
        ////h_X[i] = i;
        //h_X[i] = rand();
    //}
    
    
    // X = np.array([[1,2,9],[3,4,8], [-1, 6, 2]], np.float32)
    
    //### fixed values
    h_X[0] = 1;
    h_X[1] = 3;
    h_X[2] = -1;
    h_X[3] = 2;
    h_X[4] = 4;
    h_X[5] = 6;
    h_X[6] = 9;
    h_X[7] = 8;
    h_X[8] = 2;
    
    float * d_X = 0;
    //### The X input matrix
    CUDA_MALLOC(d_X, n2, "d_X");
    
    //cout << "CUBLAS_STATUS_SUCCESS: " << CUBLAS_STATUS_SUCCESS << endl;
    
    float * d_Xpinv = 0;
    //### The pseudo-inverse of the X matrix
    CUDA_MALLOC(d_Xpinv, n2, "d_Xpinv");
    
    CUDA_MEMCPY(d_X, h_X, n2, cudaMemcpyHostToDevice, "Error copying X from host to device")
    
    //### Compute the pseudo inverse
    //pinv(d_X, n, n, d_Xpinv);
    pinv(d_X, n, n, d_Xpinv, handle, cs_handle);
    
    CUDA_MEMCPY(h_X, d_Xpinv, n2, cudaMemcpyDeviceToHost, "Error copying X from device to host")
    
    for (int i = 0; i < n2; i++) {
        cout << h_X[i] << " ";
    }
    
    cout << endl;
    
    //### FREE UP MEMORY
    CUDA_FREE(d_X, "d_X");
    CUDA_FREE(d_Xpinv, "d_Xpinv");
    
    free(h_X);
    
    cublasDestroy(handle);
    cusolverDnDestroy(cs_handle);
    
}