#include <stdio.h>

int l1l2_regularization(float *, float *, int, int, float, float, float *, int *, int , float);

//int l1l2_regularization_optimized(float * d_XT, float * d_Y, float * d_XTY, int n, int p, float mu, float tau, float * d_beta, int kmax, int adaptive, cublasHandle_t handle);

int l1l2_regularization_optimized(double * d_XT, double * d_Y, double * d_XTY, int n, int p, double mu, double tau, double * d_beta_in, double * d_beta, int * k_final, int kmax, double tolerance, int adaptive,  cublasHandle_t handle);