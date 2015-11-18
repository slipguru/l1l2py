#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np

import time
import cPickle as pkl
import matplotlib.pyplot as plt

from algorithms_lite import l1l2_path
from l1l2py.pycu_algorithms import l1l2_path as cu_l1l2_path

from test_cu_l1l2reg import show_speedup
from glados.utils.gendata import grouped


def test_l1l2_path(n_samples, n_features_array, tau_range):

    cpu_time = []
    gpu_time = []

    for n_features in n_features_array:
        print("p = {}".format(n_features))

        X, Y = grouped(n_samples, n_features)

        _mu = 0.01
        _kmax = 10000

        # CPU 64 bit
        tic = time.time()
        cpu_sol64 = l1l2_path(X, Y, _mu, tau_range, kmax = _kmax)
        tac = time.time()
        cpu_sol64 = np.array(cpu_sol64)
        print("CPU[64 bit]: l1l2 path solution = {}, elapsed time = {}".format(np.sum(np.abs(cpu_sol64)), tac-tic))
        cpu_time.append(tac-tic)
        # -------------------------------------------------------------------- #

        # GPU 32 bit
        tic = time.time()
        gpu_sol = cu_l1l2_path(X, Y, _mu, tau_range, kmax = _kmax)
        tac = time.time()
        gpu_sol = np.array(gpu_sol)
        print("GPU[32 bit]: l1l2 path solution = {}, elapsed time = {}".format(np.sum(np.abs(gpu_sol)), tac-tic))
        cpu_time.append(tac-tic)
        # -------------------------------------------------------------------- #

        THRESH = 1e-4
        if np.sum(cpu_sol64 - gpu_sol) > THRESH:
            print("*** GPU sol - CPU sol > {} ***".format(THRESH))

        print("-----------------------------------------------------------------")

    return cpu_time, gpu_time


def main():
    n_samples = 200
    n_features_array = np.arange(500, 50000, 1000)
    tau_range = np.arange(1e-4,1,1e-1) # 10 elements
    xaxis = n_features_array
    yaxis = []

    cpu_time, gpu_time = test_l1l2_path(n_samples, n_features_array, tau_range)

    # show_speedup(cpu_time, gpu_time, xaxis, yaxis)
    # plt.show()


if __name__ == '__main__':
    main()
