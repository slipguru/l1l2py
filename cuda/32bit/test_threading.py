import numpy as np

import sys

from multiprocessing import Process

import threading

import ctypes

def run_tests():
    
    data_file = 'data_c/X_250_1600.csv'
    labels_file = 'data_c/Y_250_1600.csv'
    
    data = np.genfromtxt(data_file)
    labels = np.genfromtxt(labels_file)
    
    mu = 1e-5
    #tau_range = [1e-2,1e0]
    # tau_range = [1e0]
    # tau_range = [1e-1]
    tau_range = np.logspace(-3,0,20)
    #tau_range = np.logspace(-3,0,20)
    
    # tau_range = tau_range[:2]
    
    k_max = 10000
    
    tolerance = 1e-5
    
    # data_file = 'data_c/X_%d_%d.csv' % (int(n), int(p))
    # labels_file = 'data_c/Y_%d_%d.csv' % (int(n), int(p))

    # X = data[:n, :p].copy()
    # Y = labels[:n].copy()
    
    X = data.copy()
    Y = labels.copy()
    
    n, p = X.shape
    
    # XT = np.array(list(X.transpose())) ####AAAARGHHHHHHH MUST COPY THIS WAY!!!
    # XT = XT.astype(np.float32)
    
    XT = X.astype(np.float32)
    Y = Y.astype(np.float32)
    
    n_tau = len(tau_range)
    adaptive = 0
    
    tau_range = np.array(tau_range).astype(np.float32)
    
    beta = np.zeros((p,)).astype(np.float32)
    # out = 6 * np.ones((n_tau,p)).astype(np.float32)
    out = np.empty((n_tau,p)).astype(np.float32)
    
    k_final = ctypes.c_int32()
    n_betas_out = ctypes.c_int32()
    
    # print("K final = {}".format(k_final))
    
    testlib = ctypes.CDLL("./l1l2_path.so", mode=ctypes.RTLD_GLOBAL)
    
    # simple_lib = ctypes.CDLL("./simple_arr.so", mode=ctypes.RTLD_GLOBAL)
    # simple_lib = ctypes.CDLL("./simple_arr.so")
    simple_lib = ctypes.cdll.LoadLibrary("./simple_arr.so")
    
    
    # testlib.l1l2_path_bridge(
    #     XT.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    #     Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    #     ctypes.c_int(n),
    #     ctypes.c_int(p),
    #     ctypes.c_float(mu),
    #     tau_range.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), # float * h_tau_range,
    #     ctypes.c_int(n_tau), # int n_tau,
    #     beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), # float * h_beta,
    #     out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), # float * h_out,
    #     # ctypes.byref(ctypes.c_int(k_final)),
    #     ctypes.byref(n_betas_out),
    #     ctypes.byref(k_final),
    #     ctypes.c_int(k_max), # int kmax,
    #     ctypes.c_float(tolerance), # float tolerance,
    #     # ctypes.c_float(1e-2*tolerance), # float tolerance,
    #     ctypes.c_int(adaptive) # int adaptive
    # )
    
    # simple_lib.simple_arr(
    #     XT.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    #     ctypes.c_int(n),
    #     ctypes.c_int(p)
    # )
    
    # p = Process(target = simple_lib.simple_arr, args=(XT.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    #     ctypes.c_int(n),
    #     ctypes.c_int(p)))
    # p.start()
    # p.join()
    
    p = threading.Thread(target = simple_lib.simple_arr, args=(XT.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(n),
        ctypes.c_int(p)))
    p.start()
    # p.join()
    
    with open('/tmp/aaa.txt', 'w') as f:
        f.write('bbb')
    
    # print("total time (CUDA): {}".format(dt))
    
    # for i in range(len(tau_range)):
        # print out[i,:20]
    
    return -1

    
if __name__ == '__main__':
    
    run_tests()
    # analyze_results()
    # heatmap()