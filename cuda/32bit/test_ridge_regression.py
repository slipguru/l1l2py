import numpy as np
import numpy.linalg as la

import time

import sys

from algorithms_lite import l1l2_regularization

import ctypes

# testlib = ctypes.CDLL("./pinv.so")

testlib = ctypes.CDLL("./ridge_regression.so", mode=ctypes.RTLD_GLOBAL)

def test_ridge_regression_py(X, Y, _lambda):

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    
    t1 = time.time()
    
    n, p = X.shape

    if n < p:
        tmp = np.dot(X, X.T)
        if _lambda:
            tmp += _lambda*n*np.eye(n)
        tmp = la.pinv(tmp)

        beta_out = np.dot(np.dot(X.T, tmp), Y.reshape(-1, 1))
    else:
        tmp = np.dot(X.T, X)
        if _lambda:
            tmp += _lambda*n*np.eye(p)
        tmp = la.pinv(tmp)

        beta_out = np.dot(tmp, np.dot(X.T, Y.reshape(-1, 1)))
    
    t2 = time.time()
    dt = t2-t1
    
    print("total time (Python): {}".format(dt))
    print(beta_out[:20,0])
    
    
def test_ridge_regression_cu(X, Y, _lambda):
    n, p = X.shape
    
    # XT = np.array(list(X.transpose())) ####AAAARGHHHHHHH MUST COPY THIS WAY!!!
    # XT = XT.astype(np.float32)
    
    XT = X.astype(np.float32)
    Y = Y.astype(np.float32)
    
    beta_out = np.empty((p,)).astype(np.float32)
    
    t1 = time.time()
    
    # ridge_regression_bridge(float * h_XT, float * h_Y, int n, int p, float lambda, float * beta_out) {
    testlib.ridge_regression_bridge(
                        XT.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        ctypes.c_int(n),
                        ctypes.c_int(p),
                        ctypes.c_float(_lambda),
                        beta_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                        )
    
    t2 = time.time()
    dt = t2-t1
    
    print("total time (CUDA): {}".format(dt))
    print(beta_out[:20])

if __name__ == '__main__':
    
    p = sys.argv[1]

    
    
    data_file = 'data_b/X_%d.csv' % int(p)
    labels_file = 'data_b/Y_%d.csv' % int(p)
    
    # data_file = 'data/X_%d.csv' % int(p)
    # labels_file = 'data/Y_%d.csv' % int(p)
    
    # data_file = 'data/X_1000_4000.csv'
    # labels_file = 'data/Y_1000_4000.csv'
    
    X = np.genfromtxt(data_file)
    Y = np.genfromtxt(labels_file)
    _lambda = 1e-2
    
    test_ridge_regression_cu(X, Y, _lambda)
    test_ridge_regression_py(X, Y, _lambda)
    
    