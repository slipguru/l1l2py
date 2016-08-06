import numpy as np
import ctypes

import numpy.linalg as la

import time

# testlib = ctypes.CDLL("./pinv.so")
testlib = ctypes.CDLL("./pinv.so", mode=ctypes.RTLD_GLOBAL)

def c_bind(X, Xpinv):
    (n,p) = X.shape
    
    # print X
    
    # XT = X.T
    XT = np.array(list(X.transpose())) ####AAAARGHHHHHHH MUST COPY THIS WAY!!!
    
    # print XT
    
    # return
    
    # print("cuda.pinv(X) before:")
    # print Xpinv.T
    
    # print testlib
  
    # testlib.pm(m.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
             # ctypes.c_int(r), ctypes.c_int(c))
    
    # pinv_bridge(float * h_X, float * h_Xpinv, int n, int p)
    testlib.pinv_bridge(
                        XT.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        Xpinv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        ctypes.c_int(n),
                        ctypes.c_int(p)
                        )

    # print("cuda.pinv(X):")
    # print Xpinv.T
    
    return Xpinv.T

def main():
  
    n = 2000
  
    # X = np.array([[1,2,9],[3,4,8], [-1, 6, 2]], np.float32)
    
    X = np.random.normal(size = (n,n))
    X = np.array(X, np.float32)
    
    Xpinv = np.empty(X.T.shape, np.float32)
    
    tic = time.time()
    X_pinv1 = la.pinv(X)
    tac = time.time()
    
    dt1 = tac-tic
    print("la.pinv time: %f" % dt1)
    
    # print("la.pinv(X):")
    # print(X_pinv1)
    
    # print(la.eig(X))
    
    # return
    
    # print("Original matrix (python):")
    # print X
    
    # print X.dtype
    
    tic = time.time()
    X_pinv_cuda = c_bind(X, Xpinv)
    tac = time.time()
    
    dt2 = tac-tic
    print("cuda.pinv time: %f" % dt2)
    
    time_ratio = dt1/dt2
    
    print("Speedup: %f" % time_ratio)
    
    # print("cuda.pinv(X):")
    # print X_pinv_cuda

if __name__ == '__main__':
  main()
