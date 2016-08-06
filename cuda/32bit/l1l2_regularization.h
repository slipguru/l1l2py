#include <stdio.h>

int l1l2_regularization(float *, float *, int, int, float, float, float *, int *, int , float);

//int l1l2_regularization_optimized(float * d_XT, float * d_Y, float * d_XTY, int n, int p, float mu, float tau, float * d_beta, int kmax, int adaptive, cublasHandle_t handle);

int l1l2_regularization_optimized(float * d_XT, float * d_Y, float * d_XTY, int n, int p, float mu, float tau, float * d_beta_in, float * d_beta, int * k_final, int kmax, float tolerance, int adaptive,  cublasHandle_t handle);