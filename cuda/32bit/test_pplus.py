import numpy as np

import sys

# from algorithms import l1l2_path
# from l1l2py.algorithms import l1l2_path

import pplus

import ctypes

def simplearr_job(pc, X):
    
    n, p = X.shape
    
    X32 = X.astype(np.float32)
    
    res = ctypes.c_float()
    
    try:
        simple_lib = ctypes.CDLL("/home/matteo/projects/L1L2C/32bit/simple_arr.so")
        # simple_lib = ctypes.cdll.LoadLibrary("/home/matteo/projects/L1L2C/32bit/simple_arr.so")
    except Exception as e:
        return e
    
    # return n
    
    simple_lib.simple_arr(
        X32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(n),
        ctypes.c_int(p),
        ctypes.byref(res)
    )
    
    # with open('/tmp/aaa.txt', 'w') as f:
        # f.write('res = {}'.format(res.value))
    
    return res.value

def l1l2path_job(pc, X, Y, mu, tau_range, k_max, tolerance):

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
    
    # testlib = ctypes.CDLL("./l1l2_path.so", mode=ctypes.RTLD_GLOBAL)
    try:
        # testlib = ctypes.CDLL("/home/matteo/projects/L1L2C/32bit/l1l2_path.so")
        testlib = ctypes.CDLL("/home/matteo/projects/L1L2C/32bit/l1l2_path.so", mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        return e
    
    try:
        testlib.l1l2_path_bridge(
            XT.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n),
            ctypes.c_int(p),
            ctypes.c_float(mu),
            tau_range.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), # float * h_tau_range,
            ctypes.c_int(n_tau), # int n_tau,
            beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), # float * h_beta,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), # float * h_out,
            # ctypes.byref(ctypes.c_int(k_final)),
            ctypes.byref(n_betas_out),
            ctypes.byref(k_final),
            ctypes.c_int(k_max), # int kmax,
            ctypes.c_float(tolerance), # float tolerance,
            # ctypes.c_float(1e-2*tolerance), # float tolerance,
            ctypes.c_int(adaptive) # int adaptive
        )
        
        # return 8
    
    except Exception as e:
        return e
        return 9
        
    # with open('/tmp/aaa.txt', 'w') as f:
        # f.write('bbb')
    
    
    # print("K final (CUDA) = {}".format(k_final.value))
    
    # print("total time (CUDA): {}".format(dt))
    
    # for i in range(len(tau_range)):
        # print out[i,:20]
    
    # return k_final.value

    return out[0,:20]

def test_simplearr():
    
    ### Init PPlus
    # pc = pplus.PPlusConnection(debug=True)
    pc = pplus.PPlusConnection(debug=False, workers_servers = ('127.0.0.1',))
    # pc = pplus.PPlusConnection(debug=False, workers_servers = ('10.251.61.233',))
    
    X = np.array([[2,0],[0,2]])
    
    pc.submit(simplearr_job,
                  args=(X,),
                  modules=('numpy as np', 'ctypes'))
    
    pc.submit(simplearr_job,
                  args=(2*X,),
                  modules=('numpy as np', 'ctypes'))
    
    result_keys = pc.collect()
    
    print result_keys

    print("Done")

def dummy_job(pc, arg1):
    
    # with open('/tmp/zzz{}.txt'.format(arg1), 'w') as f:
        # f.write('xxx')
    
    return arg1
    
def test_l1l2path():
    
    X_file = 'data_c/X_200_100.csv'
    Y_file = 'data_c/Y_200_100.csv'
    
    X = np.genfromtxt(X_file)
    Y = np.genfromtxt(Y_file)
    
    mu = 1e-3
    tau_range = np.logspace(-2,0,3)
    k_max = 10000
    tolerance = 1e-4
    
    pc = pplus.PPlusConnection(debug=False, workers_servers = ('127.0.0.1',))
    
    pc.submit(l1l2path_job,
                  args=(X, Y, mu, tau_range, k_max, tolerance),
                  modules=('numpy as np', 'ctypes'))

    result_keys = pc.collect()
    
    print result_keys

    print("Done")
    
def test_dummy():
    
    ### Init PPlus
    pc = pplus.PPlusConnection(debug=True)
    # pc = pplus.PPlusConnection(debug=False, workers_servers = ('127.0.0.1',))
    
    n, p = (100,10)
    
    X = np.random.normal(size=(n,p))
    
    pc.submit(dummy_job, args=(1, ))
    print("Submit1")
    
    pc.submit(dummy_job, args=(2, ))
    print("Submit2")
    
    
    result_keys = pc.collect()

    print("results collected")
    
    print result_keys 
    
    
    
if __name__ == '__main__':
    
    # test_simplearr()
    # test_dummy()
    
    test_l1l2path()
    
    # analyze_results()
    # heatmap()