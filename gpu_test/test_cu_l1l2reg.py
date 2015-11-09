#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np

import time
import cPickle as pkl
import matplotlib.pyplot as plt

from algorithms_lite import l1l2_regularization
from l1l2py.pycu_algorithms import l1l2_regularization as cu_l1l2reg

from glados.utils.gendata import grouped

def test_l1l2_regularization(n_samples = 100, n_features_array = np.linspace(1e2, 1e5, 10)):

    cpu_time = []
    gpu_time = []

    for n_features in n_features_array:
        print("p = {}".format(n_features))

        X, Y = grouped(n_samples, n_features)

        _tau = 1
        _mu = 0.01
        kmax = 10000

        # CPU 64bit
        tic = time.time()
        cpu_sol64, k = l1l2_regularization(data = X, labels = Y, mu = _mu,
                                         tau = _tau, return_iterations = True,
                                         tolerance = 1e-5,
                                         kmax = kmax)
        tac = time.time()
        print("CPU[64 bit]: l1l2 solution = {}, elapsed time = {}, iterations = {}".format(np.sum(np.abs(cpu_sol64)), tac-tic, k))
        cpu_time.append(tac-tic)
        # -------------------------------------------------------------------- #

        # GPU
        tic = time.time()
        gpu_sol, k = cu_l1l2reg(data = X, labels = Y,
                                mu = _mu, tau = _tau,
                                tolerance = 1e-5,
                                return_iterations = True,
                                kmax = kmax)
        tac = time.time()
        print("GPU[32 bit]: l1l2 solution = {}, elapsed time = {}, iterations = {}".format(np.sum(np.abs(gpu_sol)), tac-tic, k))
        gpu_time.append(tac-tic)
        # -------------------------------------------------------------------- #

        print("CPU sel feat: {}".format(np.sum(cpu_sol64 != 0)))
        print("GPU sel feat: {}".format(np.sum(gpu_sol != 0)))

        THRESH = 1e-4
        if np.sum(cpu_sol64 - gpu_sol) > THRESH:
            print("*** GPU sol - CPU sol > {} ***".format(THRESH))

        print("-----------------------------------------------------------------")

    return cpu_time, gpu_time

def test_real_scenario(rootFolder):
    cpu_time = []
    gpu_time = []
    xaxis = []
    yaxis = []

    _tau = 0.000765100671141
    _mu = 0.05
    # kmax = 10000
    kmax = 800
    for f in sorted(os.listdir(rootFolder)):
        if f.startswith('X'):
            data = np.genfromtxt(os.path.join(rootFolder,f), delimiter = ' ')
            n, p = data.shape
            labels = np.genfromtxt(os.path.join(rootFolder,'Y_'+str(n)+'_'+str(p)+'.csv'), delimiter = ' ')

            X = data
            Y = labels

            X32 = X.astype(np.float32)
            Y32 = Y.astype(np.float32)

            # CPU
            tic = time.time()
            cpu_sol64, k = l1l2_regularization(data = X, labels = Y, mu = _mu,
                                               tau = _tau, return_iterations = True,
                                               tolerance = 1e-5,
                                               kmax = kmax)
            # cpu_sol = _sigma(X, mu = 1e-2)
            tac = time.time()
            print("CPU[64 bit]: l1l2 solution = {}, elapsed time = {}, iterations = {}".format(np.sum(np.abs(cpu_sol64)), tac-tic, k))
            cpu_time.append(tac-tic)

            # GPU
            d_X = gpuarray.to_gpu_async(X32)
            d_Y = gpuarray.to_gpu_async(Y32.reshape((n,1)))

            tic = time.time()
            gpu_sol, k = cu_l1l2_regularization(gpu_data = d_X, gpu_labels = d_Y,
                                                mu = _mu, tau = _tau,
                                                tolerance = 1e-5,
                                                return_iterations = True,
                                                kmax = kmax)
            tac = time.time()
            print("GPU[32 bit]: l1l2 solution = {}, elapsed time = {}, iterations = {}".format(np.sum(np.abs(gpu_sol)), tac-tic, k))
            gpu_time.append(tac-tic)

            xaxis.append(p)
            yaxis.append(n)

            print("CPU sel feat: {}".format(np.sum(cpu_sol64 != 0)))
            print("GPU sel feat: {}".format(np.sum(gpu_sol != 0)))

            if np.sum(cpu_sol64 - gpu_sol) > 1e-5:
                print("*** GPU solution != CPU solution ***")

            print("-----------------------------------------------------------------")

    return cpu_time, gpu_time, xaxis

def show_speedup(cpu_time, gpu_time, xaxis, yaxis):
    # Dump results
    pkl.dump({'cpu': cpu_time, 'gpu': gpu_time, 'n_features': xaxis}, open( "test_pycuda.pkl", "wb" ))

    plt.figure()
    plt.plot(np.array(xaxis), np.array(cpu_time), 'ob', label = 'CPU')
    plt.plot(np.array(xaxis), np.array(gpu_time), 'og', label = 'GPU')
    plt.xlabel('n_features')
    plt.ylabel('time [sec]')
    plt.grid('on')
    plt.legend()

    plt.figure()
    plt.plot(np.array(xaxis), np.array(cpu_time) / np.array(gpu_time), 'or')
    plt.grid('on')
    plt.title('Speedup')
    plt.xlabel('n_features')

def main():
    n_samples = 200
    n_features_array = np.arange(500, 50000, 1000)
    xaxis = n_features_array
    yaxis = []

    cpu_time, gpu_time = test_l1l2_regularization(n_samples, n_features_array)
    # cpu_time, gpu_time, xaxis, yaxis = test_real_scenario(rootFolder = '/home/matteo/projects/L1L2C/data_c')

    show_speedup(cpu_time, gpu_time, xaxis, yaxis)
    plt.show()


if __name__ == '__main__':
    main()
