#include <iostream>
#include <stdlib.h>

#include "csv.h"
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


void test_ridge_regression(int p)
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
    float * beta_out;
    
    float _lambda ;
    
    struct timeval t0, t1, t2;
    
    _lambda = 1e-2f;
    
    XT = (float *)malloc(n*p*sizeof(float));
    Y = (float *)malloc(n*sizeof(float));
    beta_out = (float *)malloc(p*sizeof(float));

    //cout << "Reading data file..."     << endl;
    
    read_csv(buffer_X, n, p, XT);
    read_csv(buffer_Y, n, 1, Y);
    
    gettimeofday(&t0,NULL);

    ridge_regression(XT, Y, n, p, _lambda, beta_out);
    
    gettimeofday(&t1,NULL);
    
    double total_time;
    
    for (int i = 0; i < 20; i++) {
        printf("%f\n", beta_out[i]);
    }
    
    //printf("\n");
    
    total_time = (t1.tv_sec-t0.tv_sec)*1000000;
    total_time += (t1.tv_usec-t0.tv_usec);
    total_time /= 1000000;
    
    printf("P= %d, total time = %f\n", p, total_time);
    
    free(XT);
    free(Y);
    free(beta_out);
    
}

int main(int argc, char ** argv) {
    
    //### L1L2Regularization
    int p = atoi(argv[1]);
    test_ridge_regression(p);
    
    //### RLS
    //test_RLS();
    
    cout << "end" << endl;
    
    
}