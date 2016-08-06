import numpy as np
import ctypes

import numpy.linalg as la

testlib = ctypes.CDLL("./manycalls.so", mode=ctypes.RTLD_GLOBAL)

def c_call1(h_X, d_X):
    (n,p) = h_X.shape
    
    # pinv_bridge(float * h_X, float * h_Xpinv, int n, int p)
    testlib.call1(
                        h_X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        ctypes.byref(d_X),
                        ctypes.c_int(n)
                        )
    
def c_call2(h_X, d_X):
    (n,p) = h_X.shape
    
    # pinv_bridge(float * h_X, float * h_Xpinv, int n, int p)
    testlib.call2(
                        h_X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        ctypes.byref(d_X),
                        ctypes.c_int(n)
                        )

def main():
    h_X = np.array([[1,2,9],[3,4,8], [-1, 6, 2]], np.float32)
    h_X_cpy = np.empty(h_X.shape, np.float32)
    d_X = ctypes.POINTER(ctypes.c_float)()
    
    print("Before:")
    print(h_X)
    print(h_X_cpy)
    
    c_call1(h_X, d_X)
    
    ### things happen...
    
    c_call2(h_X_cpy, d_X)
    
    print("After:")
    print(h_X)
    print(h_X_cpy)
    

if __name__ == '__main__':
  main()
