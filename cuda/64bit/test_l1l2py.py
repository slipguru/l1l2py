import numpy as np
import time

import sys

from algorithms_lite import l1l2_regularization
# from l1l2py.algorithms import l1l2_regularization

import ctypes

#testlib = ctypes.CDLL("./l1l2_regularization.so", mode=ctypes.RTLD_GLOBAL)

def test_l1l2regularization_py(X, Y, mu, tau, k_max, tolerance):
    
    t0 = time.time()
    
    t1 = time.time()
    
    # X = X.astype(np.float32)
    # Y = Y.astype(np.float32)
    
    beta_out, k_final = l1l2_regularization(X, Y, mu, tau, return_iterations=True, kmax = k_max, tolerance = tolerance)
    
    t2 = time.time()
    
    dt = t2-t1
    
    print("total time (python): {}".format(dt))
    print(beta_out[:20])
    
    # print beta[:20]
    
    # print("P = %d %f %f" % (int(p), (t1-t0), (t2-t1)))
    # print("P = %d time = %f" % (int(p), (t2-t1)))
    
    # print("Initialization time: ", (t1-t0))
    # print("Computation time: ", (t2-t1))
    
    print("K final (Python) = {}".format(k_final))
    
    
def test_l1l2regularization_cu(X, Y, mu, tau, k_max, tolerance):
    n, p = X.shape
    
    # XT = np.array(list(X.transpose())) ####AAAARGHHHHHHH MUST COPY THIS WAY!!!
    # XT = XT.astype(np.float32)
    
    XT = X.astype(np.float32)
    Y = Y.astype(np.float32)
    
    beta_out = np.empty((p,)).astype(np.float32)
    
    # print XT
    
    # return
    
    # print("cuda.pinv(X) before:")
    # print Xpinv.T
    
    # print testlib
  
    # testlib.pm(m.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
             # ctypes.c_int(r), ctypes.c_int(c))
    
    # pinv_bridge(float * h_X, float * h_Xpinv, int n, int p)
    
    # (float * XT, float * Y, int n, int p, float mu, float tau, float * beta_out)
    
    t1 = time.time()
    
    k_final = ctypes.c_int32()
    
    testlib.l1l2_regularization_bridge(
                        XT.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        ctypes.c_int(n),
                        ctypes.c_int(p),
                        ctypes.c_float(mu),
                        ctypes.c_float(tau),
                        beta_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        ctypes.byref(k_final),
                        ctypes.c_int(k_max),
                        ctypes.c_float(tolerance)
                        )
    
    t2 = time.time()
    dt = t2-t1
    
    print("K final (CUDA) = {}".format(k_final))
    
    # print("total time (CUDA): {}".format(dt))
    print(beta_out[:20])

if __name__ == '__main__':
    
    # p = sys.argv[1]

    # data_file = 'data_b/X_%d.csv' % int(p)
    # labels_file = 'data_b/Y_%d.csv' % int(p)
    
    data_file = 'data_c/X_600_100000.csv'
    labels_file = 'data_c/Y_600_100000.csv'
    
    # data_file = 'data/X_%d.csv' % int(p)
    # labels_file = 'data/Y_%d.csv' % int(p)
    
    # data_file = 'data/X_1000_4000.csv'
    # labels_file = 'data/Y_1000_4000.csv'
    
    X = np.genfromtxt(data_file)
    Y = np.genfromtxt(labels_file)
    
    # X = X[:, :1000]
    
    mu = 1e-5
    # mu = 1e-2
    tau = 1e-3
    
    k_max = 10000
    tolerance = 1e-5
    
    print("Python:")
    test_l1l2regularization_py(X, Y, mu, tau, k_max, tolerance)
    
    #print("CUDA:")
    #test_l1l2regularization_cu(X, Y, mu, tau, k_max, tolerance)
    
    
    
    
