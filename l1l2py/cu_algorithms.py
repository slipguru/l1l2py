#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import skcuda.linalg as linalg
from skcuda.misc import multiply as cu_mul
from skcuda.misc import maxabs as cu_maxabs
from skcuda.misc import subtract as cu_sub
from skcuda.misc import add as cu_add
from math import sqrt

import time, sys

linalg.init()

from l1l2_kernels import mod as l1l2_kernels
BLOCK_MAX = 1024

def cu_ridge_regression(gpu_data, gpu_labels, mu=0.0):
    """
    See l1l2py.algorithms.ridge_regression.

    Parameters
    ----------
    gpu_data : (N, P) pycuda.gpuarray.GPUarray
        Data matrix.
    labels : (N,)  or (N, 1) pycuda.gpuarray.GPUarray
        Labels vector.
    mu : float, optional (default is `0.0`)
        `l2-norm` penalty.

    Returns
    --------
    beta : (P, 1) ndarray
        Ridge regression solution.
    """
    n, p = gpu_data.shape

    if n < p:
        gpu_tmp = linalg.dot(gpu_data, gpu_data, transb = 'T')
        gpu_eye_mun = linalg.eye(n) # temporary eye matrix
        linalg.scale(mu*n, gpu_eye_mun)
        gpu_tmp += gpu_eye_mun
        gpu_tmp = linalg.pinv(gpu_tmp)
        gpu_out = linalg.dot(linalg.dot(gpu_data, gpu_tmp, transa = 'T'), gpu_labels)
    else:
        gpu_tmp = linalg.dot(gpu_data, gpu_data, transa = 'T')
        gpu_eye_mup = linalg.eye(p)
        linalg.scale(mu*p, gpu_eye_mup)
        gpu_tmp += gpu_eye_mup
        gpu_tmp = linalg.pinv(gpu_tmp)
        gpu_out = linalg.dot(gpu_tmp, linalg.dot(gpu_data, gpu_labels, transa = 'T'))

    return linalg.transpose(gpu_out).get()

def cu_l1l2_regularization(gpu_data, gpu_labels, mu, tau, beta=None, kmax=100000,
                        tolerance=1e-5, return_iterations=False,
                        adaptive=False):
    """
    See l1l2py.algorithms.l1l2_regularization.

    Parameters
    ----------
    gpu_data : (N, P) pycuda.gpuarray.GPUarray
        Data matrix.
    gpu_labels : (N,) or (N, 1) pycuda.gpuarray.GPUarray
        Labels vector.
    mu : float
        `l2-norm` penalty.
    tau : float
        `l1-norm` penalty.
    beta : (P,) or (P, 1) ndarray, optional (default is `None`)
        Starting value for the iterations.
        If `None`, then iterations starts from the empty model.
    kmax : int, optional (default is `1e5`)
        Maximum number of iterations.
    tolerance : float, optional (default is `1e-5`)
        Convergence tolerance.
    return_iterations : bool, optional (default is `False`)
        If `True`, returns the number of iterations performed.
        The algorithm has a predefined minimum number of iterations
        equal to `10`.
    adaptive : bool, optional (default is `False`)
        If `True`, minimization is performed calculating an adaptive step size
        for each iteration.

    Returns
    -------
    beta : (P, 1) ndarray
        `l1l2` solution.
    k : int, optional
        Number of iterations performed.
    """
    n, d = gpu_data.shape

    # beta starts from 0 and we assume also that the previous value is 0
    if beta is None:
        gpu_beta = gpuarray.zeros((d,1),np.float32) # single precision
    else:
        gpu_beta = gpuarray.to_gpu(beta.reshape((d,1)).astype(np.float32))

    # Easier to read
    gpu_X = gpu_data
    gpu_Y = gpu_labels

    if n > d:
        gpu_XTY = linalg.dot(gpu_X, gpu_Y, transa = 'T')

    # First iteration with standard sigma
    sigma = _cu_sigma(gpu_X, mu)

    if sigma < np.finfo(float).eps: # is zero...
        return np.zeros(d), 0

    mu_s = (mu / sigma).astype(np.float32)
    tau_s = (tau / (2.0 * sigma)).astype(gpu_X.dtype)
    nsigma = (n * sigma).astype(np.float32)

    # Starting conditions
    gpu_aux_beta = gpu_beta.copy() ### check if copy is necessary
    t = 1.

    # Get the soft thresh kernel
    cu_soft_thresholding = l1l2_kernels.get_function('soft_thresholding')

    block_dim = min(BLOCK_MAX, d)
    num_block = max(n,d) // block_dim

    for k in xrange(kmax):
        # Pre-calculated "heavy" computation
        if n > d:
            gpu_precalc = gpu_XTY - linalg.dot(gpu_X, linalg.dot(gpu_X, gpu_aux_beta), transa = 'T')
        else:
            gpu_precalc = linalg.dot(gpu_X, gpu_Y - linalg.dot(gpu_X, gpu_aux_beta), transa = 'T')

        ######## SOFT THRESHOLDING ####################################################
        gpu_beta_next = gpuarray.empty_like(gpu_beta) ## check for
        cu_soft_thresholding(gpu_precalc, np.uint32(d), nsigma, mu_s, tau_s,
                             gpu_aux_beta, gpu_beta_next,
                             block = (block_dim, 1, 1), grid = (num_block+1, 1))

        ######## FISTA ####################################################
        gpu_beta_diff = cu_sub(gpu_beta_next, gpu_beta) ## check if minus is faster

        # Convergence value
        max_diff = cu_maxabs(gpu_beta_diff).get()
        max_coef = cu_maxabs(gpu_beta_next).get()

        t_next = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t*t))
        linalg.scale(((t - 1.0)/t_next), gpu_beta_diff)
        gpu_aux_beta = cu_add(gpu_beta_next, gpu_beta_diff) ## check +

        # Values update
        t = t_next
        gpu_beta = gpu_beta_next

        # Stopping rule (exit even if beta_next contains only zeros)
        if np.allclose(max_coef, 0) or (max_diff / max_coef) <= tolerance: break

    if return_iterations:
        return gpu_beta.get(), k+1
    return gpu_beta.get()


def _cu_sigma(gpu_matrix, mu):
    """
    np.float = _cu_sigma(pycuda.gpuarray.GPUarray, np.float)

    Parameters
    ----------
    gpu_matrix : (N, P) pycuda.gpuarray.GPUarray
        Data matrix (X).
    mu : float
        `l2-norm` penalty.

    Returns
    -------
    sigma : float
        {[Max singular value of X X^T (or X^T X)] / n} + mu
    """
    n, p = gpu_matrix.shape

    if p > n:
        gpu_tmp = linalg.dot(gpu_matrix, gpu_matrix, transb = 'T')
    else:
        gpu_tmp = linalg.dot(gpu_matrix, gpu_matrix, transa = 'T')

    gpu_norm = linalg.svd(gpu_tmp, 'N', 'N')[0]
    linalg.scale(1./n, gpu_norm)

    return gpu_norm.get() + mu
