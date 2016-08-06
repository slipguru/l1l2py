#include <iostream>
#include <stdlib.h>

void print_array(int n, int p, float * X) {
    for (int i = 0; i < n; i++){

        for (int j = 0; j < p; j++){
            std::cout << X[i*p + j] << " ";
        }

        std::cout << std::endl;
    }
}

/**
 * Takes a transposed data matrix (row major) and selects
 * exclusively the columns corresponding to those features
 * which have non zero coefficients in the input beta vector.
 *
 * Also allocates the X_out matrix, since it is not known in advance
 * how many features will be selectged
 */
void slice_array(float * X, int n, int p, float * beta, float ** X_out) {

    int p_selected = 0;

    //### count how many features have been selected by the FS algorithm
    for (int j = 0; j < p; j++) {
        if (beta[j] != 0) {
            p_selected++;
        }
    }

    std::cout << "wewewe" << std::endl;

    //### allocate space for the X_out matrix
    *X_out = (float *)malloc(n * p_selected * sizeof(float));

    int j_sel;

    for (int i = 0; i < n; i++) {
        j_sel = 0;
        for (int j = 0; j < p; j++) {
            if (beta[j] != 0) {
                (*X_out)[i*p_selected + j_sel] = X[i*p + j];
                j_sel++;
            }
        }
    }


}


/**

  Matrix slicing utilities:
    - matrix_slice_rows: takes an input matrix X and return a new matrix presenting only the rows at specified indexes
    - matrix_slice_cols: takes an input matrix X and return a new matrix presenting only the cols at specified indexes
    - matrix_slice: wrapper of the first two functions

**/

void matrix_slice_rows(float * X, int n, int p, int * idx_rows, int n_rows, float * X_sliced) {
  // select rows
  int idx;
  for(int i=0; i < n_rows; i++){
    idx = idx_rows[i];

    for(int j=0; j < p; j++){
        X_sliced[i*p + j] = X[idx*p + j];
        //std::cout << X[idx*p+j] << " ";
    }
  } // end for i
} // end function


void matrix_slice_cols(float * X, int n, int p, int * idx_cols, int n_cols, float * X_sliced) {
  // select cols
  int idx;
  for(int j=0; j < n_cols; j++){
    idx = idx_cols[j];

    for(int i=0; i < n; i++){
        X_sliced[i*n_cols + j] = X[i*p + idx];
    }
  } // end for j
} // end function

void matrix_slice(float * X, int n, int p, char dir, int * idx_sel, int n_sel, float * X_sliced){
  // choose direction
  switch(dir){
    case 'R': matrix_slice_rows(X, n, p, idx_sel, n_sel, X_sliced);
      break;
    case 'C': matrix_slice_cols(X, n, p, idx_sel, n_sel, X_sliced);
      break;
  }

}
