#ifndef _H_UTILS
#define _H_UTILS

#define CUDA_MALLOC(P, n, msg)  { \
    if (cudaMalloc((void **)&P, (n) * sizeof(*P)) != cudaSuccess) \
    { \
        fprintf(stderr, "!!!! device memory allocation error (allocate %s)\n", msg); \
        return EXIT_FAILURE; \
    } \
}

#define CUDA_FREE(P, msg) { \
    if (cudaFree(P) != cudaSuccess) \
    { \
        fprintf(stderr, "!!!! memory free error (%s) \n", msg); \
        return EXIT_FAILURE; \
    } \
}

#define CUDA_CHECK_STATUS(STATUS, msg) { \
if (STATUS != CUBLAS_STATUS_SUCCESS) \
    { \
        fprintf(stderr, "!!!! %s\n", msg); \
        return EXIT_FAILURE; \
    } \
}

#define CUDA_MEMCPY(_dst, _src, n, direction, msg) { \
    if (cudaMemcpy(_dst, _src, (n) * sizeof(*_dst), direction) != cudaSuccess) { \
        fprintf(stderr, "!!!! CUDA_MEMCPY ERROR: %s\n", msg); \
        fprintf(stderr, "!!!! ERROR STRING: %s\n", cudaGetErrorString(cudaGetLastError())); \
        return EXIT_FAILURE; \
    } \
}

//### EXAMPLES:
//CUDA_MALLOC(d_A, n, "d_A");
//CUDA_FREE(d_A, "d_A");

void print_array(int, int, float *);

void slice_array(float *, int, int, float *, float **);

void matrix_slice_rows(float *, int, int, int *, int, float *);
//                     input X,   n,   p, idx_row, n_row, X_sliced

void matrix_slice_cols(float *, int, int, int *, int, float *);
//                     input X,   n,   p, idx_col, n_col, X_sliced

void matrix_slice(float *, int, int, char, int *, int, float *);
//               input X,   n,   p,   dir, idx_sel, n_sel, X_sliced

#endif
