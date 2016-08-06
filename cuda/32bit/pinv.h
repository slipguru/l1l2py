//int pinv(float * d_X, int n, int p, float * d_Xinv);
int pinv(float *, int, int, float *, cublasHandle_t, cusolverDnHandle_t);