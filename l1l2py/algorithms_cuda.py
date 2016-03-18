"""Internal algorithms implementations.

This module contains the functions strictly related with the statistical
elaboration of the data.

"""
## This code is written by Salvatore Masecchia <salvatore.masecchia@unige.it>
## and Annalisa Barla <annalisa.barla@unige.it>
## Copyright (C) 2010 SlipGURU -
## Statistical Learning and Image Processing Genoa University Research Group
## Via Dodecaneso, 35 - 16146 Genova, ITALY.
##
## This file is part of L1L2Py.
##
## L1L2Py is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## L1L2Py is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with L1L2Py. If not, see <http://www.gnu.org/licenses/>.

__all__ = ['l1_bound', 'ridge_regression', 'l1l2_regularization', 'l1l2_path']

from math import sqrt

from .algorithms import l1l2_regularization, l1_bound, _sigma, ridge_regression, emergency_log ### check this one

import os

import numpy as np
try:
    from scipy import linalg as la
except ImportError:
    from numpy import linalg as la
    
import ctypes

algorithms_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)),'l1l2_path.so'), mode=ctypes.RTLD_GLOBAL)

def l1l2_path(data, labels, mu, tau_range, beta=None, kmax=100000,
              tolerance=1e-5, adaptive=False, input_key = None):
    r"""Efficient solution of different `l1l2` regularization problems on
    increasing values of the `l1-norm` parameter.

    Finds the `l1l2` regularization path for each value in ``tau_range`` and
    fixed value of ``mu``.

    The values in ``tau_range`` are used during the computation in reverse
    order, while the output path has the same ordering of the `tau` values.

    .. note ::

        For efficency purposes, if ``mu = 0.0`` and the number of non-zero
        values is higher than `N` for a given value of tau (that means algorithm
        has reached the limit of allowed iterations), the following solutions
        (for smaller values of ``tau``) are simply the least squares solutions.

    .. warning ::

        The number of solutions can differ from ``len(tau_range)``.
        The function returns only the solutions with at least one non-zero
        element.
        For values higher than *tau_max* a solution have all zero values.

    Parameters
    ----------
    data : (N, P) ndarray
        Data matrix.
    labels : (N,) or (N, 1) ndarray
        Labels vector.
    mu : float
        `l2-norm` penalty.
    tau_range : array_like of float
        `l1-norm` penalties in increasing order.
    beta : (P,) or (P, 1) ndarray, optional (default is `None`)
        Starting value of the iterations.
        If `None`, then iterations starts from the empty model.
    kmax : int, optional (default is `1e5`)
        Maximum number of iterations.
    tolerance : float, optional (default is `1e-5`)
        Convergence tolerance.
    adaptive : bool, optional (default is `False`)
        If `True`, minimization is performed calculating an adaptive step size
        for each iteration.

    Returns
    -------
    beta_path : list of (P,) or (P, 1) ndarrays
        `l1l2` solutions with at least one non-zero element.

    """
    
    if not input_key is None:
        emergency_log_file = '/tmp/{}.txt'.format(input_key)
    else:
        emergency_log_file = None
    
    emergency_log("l1l2_path_cuda [1]\n", emergency_log_file)
    
    from collections import deque
    
    n, p = data.shape
    
    XT = data.astype(np.float32)
    Y = labels.astype(np.float32)
    
    n_tau = len(tau_range)
    adaptive = int(adaptive)
    
    tau_range = np.array(tau_range).astype(np.float32)
    
    beta = np.zeros((p,)).astype(np.float32)
    # out = 6 * np.ones((n_tau,p)).astype(np.float32)
    out = np.empty((n_tau,p)).astype(np.float32)
    
    k_final = ctypes.c_int32()
    n_betas_out = ctypes.c_int32()
    
    emergency_log("l1l2_path_cuda [2]\n", emergency_log_file)
    
    algorithms_lib.l1l2_path_bridge(
        XT.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(n),
        ctypes.c_int(p),
        ctypes.c_float(mu),
        tau_range.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), # float * h_tau_range,
        ctypes.c_int(n_tau), # int n_tau,
        beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), # float * h_beta,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), # float * h_out,
        ctypes.byref(n_betas_out),
        ctypes.byref(k_final),
        ctypes.c_int(kmax), # int kmax,
        ctypes.c_float(tolerance), # float tolerance,
        ctypes.c_int(adaptive), # int adaptive
        ctypes.c_char_p(emergency_log_file)
    )
    
    emergency_log("l1l2_path_cuda [3]\n", emergency_log_file)
    
    out_list = list()
    
    for i in range(n_tau - n_betas_out.value, n_tau):
        out_list.append(out[i,:])
        
    emergency_log("l1l2_path_cuda [4]\n", emergency_log_file)
    
    return out_list

