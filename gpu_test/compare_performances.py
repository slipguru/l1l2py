import numpy as np
import time

import sys

from algorithms_lite import l1l2_regularization
from l1l2py.cu_algorithms import l1l2_regularization as cu_l1l2reg

import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt

KMAX = 10000

def test_l1l2regularization_py(X, Y, mu, tau):

    # t0 = time.time()

    # X = X.astype(np.float32)
    # Y = Y.astype(np.float32)

    t1 = time.time()
    # beta_out, k = l1l2_regularization(X, Y, mu, tau, return_iterations=True, kmax = 800)
    beta_out, k = l1l2_regularization(X, Y, mu, tau, return_iterations=True, kmax = KMAX)
    t2 = time.time()
    dt = t2-t1

    print("total time: {}  - iterations :{} (python)".format(dt, k))
    # print(beta_out[:20])


    # print("P = %d %f %f" % (int(p), (t1-t0), (t2-t1)))
    # print("P = %d time = %f" % (int(p), (t2-t1)))

    # print("Initialization time: ", (t1-t0))
    # print("Computation time: ", (t2-t1))

    # print k

    return dt


def test_l1l2regularization_cu(X, Y, mu, tau):
    # n, p = X.shape


    t1 = time.time()
    beta_out, k = cu_l1l2reg(X, Y, mu, tau, return_iterations = True, kmax = KMAX)
    t2 = time.time()
    dt = t2-t1

    print("total time: {}  - iterations :{} (CUDA)".format(dt, k))
    # print(beta_out[:20])

    return dt

def run_tests():
    # ns = range(50, 4050, 50)

    # ns = range(50, 1050, 50)

    ns = range(200, 650, 50)
    ps = range(5000, 100100, 100)

    py_file_path = 'results_py.txt'
    cu_file_path = 'results_cu.txt'

    py_file = open(py_file_path, 'w')
    cu_file = open(cu_file_path, 'w')

    data_file = '/home/matteo/projects/L1L2C/data_c/X_600_100000.csv'
    labels_file = '/home/matteo/projects/L1L2C/data_c/Y_600_100000.csv'

    data = np.genfromtxt(data_file)
    labels = np.genfromtxt(labels_file)

    mu = 1e-2
    tau = 1e0

    for n in ns:
        for p in ps:
            print("\n\n*********************************************\n\n")
            print("n = {}, p = {}".format(n,p))
            print("Reading data...\n\n")

            # data_file = 'data_c/X_%d_%d.csv' % (int(n), int(p))
            # labels_file = 'data_c/Y_%d_%d.csv' % (int(n), int(p))

            X = data[:n, :p].copy()
            Y = labels[:n].copy()

            print("--- Performing python experiment ---\n")
            dt_py = test_l1l2regularization_py(X, Y, mu, tau)
            print("--- Performing CUDA experiment ---\n\n")
            dt_cu = test_l1l2regularization_cu(X, Y, mu, tau)

            py_file.write("{},{},{}\n".format(n,p,dt_py))
            cu_file.write("{},{},{}\n".format(n,p,dt_cu))

            # print("{},{},{}\n".format(n,p,dt_py))
            # print("{},{},{}\n".format(n,p,dt_cu))

    ### Close files
    py_file.close()
    cu_file.close()

def analyze_results():

    py_file_path = 'results_py.txt'
    cu_file_path = 'results_cu.txt'

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

    plt.savefig('speedup_heatmap.pdf')

if __name__ == '__main__':

    run_tests()
    # analyze_results()
