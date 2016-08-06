#include <iostream>
#include <stdlib.h>

#include "csv.h"
#include "l1l2_regularization.h"
#include "ridge_regression.h"
#include "utils.h"

#include "test_svd.h"

#include <sys/time.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

using namespace std;

/* Matrix size */
#define N  (2000)

/**
 * Test for matrix slicing
 */
void test_slicing() {
    
    printf("wweqewqe\n");
    
    int n = 4;
    int p = 5;
    int p_sel = 3;
    
    float * X = (float *)malloc(n * p * sizeof(float));
    float * beta = (float *)malloc(p * sizeof(float));
    
    for (int i = 0; i < n*p; i++) {
        
        cout << i << endl;
        
        X[i] = i;
    }
    
    
    
    beta[0] = 1;
    beta[1] = 0;
    beta[2] = 1;
    beta[3] = 0;
    beta[4] = 1;
    
    float * X_out;
    
    slice_array(X, n, p, beta, &X_out);
    
    print_array(n, p_sel, X_out);
    
    free(X);
    free(beta);
    free(X_out);
    
}

void test_l1l2regularization(int p)
{
    int n = 200;
    
    //int p = 200;
    
    
    char buffer_X[50];
    char buffer_Y[50];
    
    //sprintf(buffer_X, "data/X_%d.csv", p);
    //sprintf(buffer_Y, "data/Y_%d.csv", p);
    
    sprintf(buffer_X, "data_b/X_%d.csv", p);
    sprintf(buffer_Y, "data_b/Y_%d.csv", p);
    
    //string X_fname, Y_fname;
    
    float * XT;
    float * Y;
    float * beta;
    
    float mu;
    float tau;
    
    struct timeval t0, t1, t2;
    
    mu = 1e-2f;
    tau = 1e-2f;
    
    XT = (float *)malloc(n*p*sizeof(float));
    Y = (float *)malloc(n*sizeof(float));
    beta = (float *)malloc(p*sizeof(float));

    //cout << "Reading data file..."     << endl;
    
    //read_csv("data/X.csv", n, p, XT);
    //read_csv("data/Y.csv", n, 1, Y);
    
    read_csv(buffer_X, n, p, XT);
    read_csv(buffer_Y, n, 1, Y);
    
    gettimeofday(&t0,NULL);
    
    l1l2_regularization(XT, Y, n, p, mu, tau, beta);
    
    gettimeofday(&t1,NULL);
    
    double total_time;
    
    for (int i = 0; i < 20; i++) {
        printf("%f\n", beta[i]);
    }
    
    //printf("\n");
    
    total_time = (t1.tv_sec-t0.tv_sec)*1000000;
    total_time += (t1.tv_usec-t0.tv_usec);
    total_time /= 1000000;
    
    printf("P= %d, total time = %f\n", p, total_time);
    
    free(XT);
    free(Y);
    free(beta);
    
}

void test_RLS() {
    
    int n = 200;
    
    int p = 200;
    
    char buffer_X[50];
    char buffer_Y[50];
    
    sprintf(buffer_X, "data_b/X_%d.csv", p);
    sprintf(buffer_Y, "data_b/Y_%d.csv", p);
    
    float * XT;
    float * Y;
    float * beta;
    
    float lambda = 1e-2f;
    
    XT = (float *)malloc(n*p*sizeof(float));
    Y = (float *)malloc(n*sizeof(float));
    beta = (float *)malloc(p*sizeof(float));

    read_csv(buffer_X, n, p, XT);
    read_csv(buffer_Y, n, 1, Y);
    
    ridge_regression(XT, Y, n, p, lambda, beta);
    
    free(XT);
    free(Y);
    free(beta);
    
}

int main(int argc, char ** argv) {
    
    //### L1L2Regularization
    int p = atoi(argv[1]);
    test_l1l2regularization(p);
    
    //### RLS
    //test_RLS();
    
    cout << "end" << endl;
    
    
}