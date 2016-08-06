import numpy as np
import time

import sys

from algorithms import l1l2_path
# from l1l2py.algorithms import l1l2_path

import ctypes

import matplotlib
matplotlib.use('Agg')

from mpl_toolkits.mplot3d import axes3d

from matplotlib import pyplot as plt

testlib = ctypes.CDLL("./l1l2_path.so", mode=ctypes.RTLD_GLOBAL)

def test_l1l2path_py(X, Y, mu, tau_range, k_max, tolerance):
    
    t0 = time.time()
    
    n, p = X.shape
    
    beta = np.zeros((p,))
    
    # X = X.astype(np.float32)
    # Y = Y.astype(np.float32)
    # beta = beta.astype(np.float32)
    
    t1 = time.time()
    
    out = l1l2_path(X, Y, mu, tau_range, beta=beta, kmax=k_max, tolerance=tolerance, adaptive=False)
    
    t2 = time.time()
    
    dt = t2-t1
    
    print("total time (python): {}".format(dt))
    
    # for i in range(len(tau_range)):
        # print(out[i][:20,0])

    # print(out[i].shape)
    
    # print("P = %d %f %f" % (int(p), (t1-t0), (t2-t1)))
    # print("P = %d time = %f" % (int(p), (t2-t1)))
    
    # print("Initialization time: ", (t1-t0))
    # print("Computation time: ", (t2-t1))
    
    # print k
    
    return dt
    
    
def test_l1l2path_cu(X, Y, mu, tau_range, k_max, tolerance):
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
    

    t1 = time.time()
    
    k_final = ctypes.c_int32()
    n_betas_out = ctypes.c_int32()
    
    # print("K final = {}".format(k_final))
    
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
    
    t2 = time.time()
    dt = t2-t1
    
    # print("K final (CUDA) = {}".format(k_final.value))
    
    print("total time (CUDA): {}".format(dt))
    
    # for i in range(len(tau_range)):
        # print out[i,:20]
    
    return dt

def run_tests():
    
    # ns = range(200, 250, 50)
    # ps = range(40000, 40100, 100)
    
    # ns = range(200, 650, 50)
    # ps = range(10000, 100100, 100)
    
    ns = [200]
    ps = [100]
    
    data_file = 'data_c/X_200_100.csv'
    labels_file = 'data_c/Y_200_100.csv'
    
    # data_file = 'data_c/X_600_100000.csv'
    # labels_file = 'data_c/Y_600_100000.csv'
    
    py_file_path = 'results_py_l1l2path.txt'
    cu_file_path = 'results_cu_l1l2path.txt'
    
    py_file = open(py_file_path, 'w')
    cu_file = open(cu_file_path, 'w')
    
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
    
    for n in ns:
        for p in ps:
            
            print("n = {}, p = {}".format(n,p))
            print("Reading data...")
            
            # data_file = 'data_c/X_%d_%d.csv' % (int(n), int(p))
            # labels_file = 'data_c/Y_%d_%d.csv' % (int(n), int(p))
    
            X = data[:n, :p].copy()
            Y = labels[:n].copy()
            
            print("\nPerforming python experiment...")
            dt_py = test_l1l2path_py(X, Y, mu, tau_range, k_max, tolerance)
            
            print("\nPerforming CUDA experiment...")
            dt_cu = test_l1l2path_cu(X, Y, mu, tau_range, k_max, tolerance)
            
            py_file.write("{},{},{}\n".format(n,p,dt_py))
            cu_file.write("{},{},{}\n".format(n,p,dt_cu))
            
            print("{},{},{}\n".format(n,p,dt_py))
            print("{},{},{}\n".format(n,p,dt_cu))
            
    
    ### Close files
    py_file.close()        
    cu_file.close()

def analyze_results():
    
    py_file_path = 'results_py_2.txt'
    cu_file_path = 'results_cu_2.txt'
    
    results_py = np.genfromtxt(py_file_path, delimiter = ',')
    results_cu = np.genfromtxt(cu_file_path, delimiter = ',')
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # X, Y, Z = axes3d.get_test_data(0.05)
    
    # print(X)
    # print(Y)
    # print(Z)
    
    x_py = np.array(results_py[:,0])
    y_py = np.array(results_py[:,1])
    z_py = np.array(results_py[:,2])
    
    N_x = len(np.unique(x_py))
    N_y = len(np.unique(y_py))
    
    x_py = x_py.reshape((N_x, N_y))
    y_py = y_py.reshape((N_x, N_y))
    z_py = z_py.reshape((N_x, N_y))
    
    x_cu = np.array(results_cu[:,0])
    y_cu = np.array(results_cu[:,1])
    z_cu = np.array(results_cu[:,2])
    
    x_cu = x_cu.reshape((N_x, N_y))
    y_cu = y_cu.reshape((N_x, N_y))
    z_cu = z_cu.reshape((N_x, N_y))
    
    z_speedup = z_py / z_cu

    # ax.plot_surface(x_py, y_py, z_py, color = 'red', rstride=8, cstride=8, alpha=0.3)
    # ax.plot_surface(x_cu, y_cu, z_cu, color = 'blue', rstride=8, cstride=8, alpha=0.3)
    ax.plot_surface(x_cu, y_cu, z_speedup, color = 'green', rstride=8, cstride=8, alpha=0.3)
    
    # cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    # cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    # cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
    
    plt.savefig('l1l2_speedup_2.pdf')
    
def heatmap():
    
    py_file_path = 'results_py_2.txt'
    cu_file_path = 'results_cu_2.txt'
    
    results_py = np.genfromtxt(py_file_path, delimiter = ',')
    results_cu = np.genfromtxt(cu_file_path, delimiter = ',')
    
    fig, ax = plt.subplots()
    
    x_py = np.array(results_py[:,0])
    y_py = np.array(results_py[:,1])
    z_py = np.array(results_py[:,2])
    
    N_x = len(np.unique(x_py))
    N_y = len(np.unique(y_py))
    
    x_py = x_py.reshape((N_x, N_y))
    y_py = y_py.reshape((N_x, N_y))
    z_py = z_py.reshape((N_x, N_y))
    
    x_cu = np.array(results_cu[:,0])
    y_cu = np.array(results_cu[:,1])
    z_cu = np.array(results_cu[:,2])
    
    x_cu = x_cu.reshape((N_x, N_y))
    y_cu = y_cu.reshape((N_x, N_y))
    z_cu = z_cu.reshape((N_x, N_y))
    
    # z_speedup = z_py / z_cu - 1
    z_speedup = z_py / z_cu
    
    # column_labels = np.unique(x_py)
    # row_labels = np.unique(y_py)
    
    column_labels = x_py[:,0]
    row_labels = y_py[0,:]
    
    p_min = min(row_labels)
    p_max = max(row_labels)
    
    p_delta = p_max - p_min
    
    row_labels = list()
    for i in range(0,6):
        row_labels.append(i*p_delta/4 + p_min)
    
    heatmap = ax.pcolor(z_speedup, cmap=plt.cm.seismic)
    fig.colorbar(heatmap)
    
    plt.title('Speedup Python VS CUDA')
    
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    
    plt.savefig('speedup_heatmap.pdf')
    
if __name__ == '__main__':
    
    run_tests()
    # analyze_results()
    # heatmap()
    
