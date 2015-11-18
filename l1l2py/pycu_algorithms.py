#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import skcuda.linalg as linalg
from skcuda.misc import maxabs as cu_maxabs
import pycuda.driver as cuda
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

def cu_l1l2_regularization(gpu_X, gpu_Y, mu, tau, beta=None, kmax=100000,
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
    n, d = gpu_X.shape

    # beta starts from 0 and we assume also that the previous value is 0
    if beta is None:
        gpu_beta = gpuarray.zeros((d,1),np.float32) # single precision
    else:
        gpu_beta = gpuarray.to_gpu(beta.reshape((d,1)).astype(np.float32))

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
    gpu_aux_beta = gpu_beta.copy()
    t = np.float32(1.0)

    # Get the soft thresh kernel
    # cu_soft_thresholding = l1l2_kernels.get_function('soft_thresholding')
    cu_soft_thresholdingFISTA = l1l2_kernels.get_function('soft_thresholdingFISTA')

    block_dim = min(BLOCK_MAX, d)
    num_block = max(n,d) // block_dim

    # CUDA dummy init
    t_next = gpuarray.to_gpu(np.array(np.float32(1.0)))
    _d = np.uint32(d)
    gpu_beta_next = gpuarray.empty_like(gpu_beta)
    gpu_beta_diff = gpuarray.empty_like(gpu_beta)


    for k in xrange(kmax):
        # Pre-calculated "heavy" computation
        if n > d:
            gpu_precalc = gpu_XTY - linalg.dot(gpu_X, linalg.dot(gpu_X, gpu_aux_beta), transa = 'T')
        else:
            print gpu_X.shape
            print gpu_Y.shape
            print gpu_aux_beta.shape
            gpu_precalc = linalg.dot(gpu_X, gpu_Y - linalg.dot(gpu_X, gpu_aux_beta), transa = 'T')

        ######## SOFT THRESHOLDING ####################################################

        # cu_soft_thresholding(gpu_precalc, np.uint32(d), nsigma, mu_s, tau_s,
        #                      gpu_aux_beta, gpu_beta_next,
        #                      block = (block_dim, 1, 1), grid = (num_block+1, 1))

        # ######## FISTA ####################################################
        # gpu_beta_diff = gpu_beta_next - gpu_beta

        ######## SOFT THRESHOLDING + FISTA #############################################

        cu_soft_thresholdingFISTA(gpu_precalc, _d, nsigma, mu_s, tau_s,
                                  gpu_beta, gpu_aux_beta, gpu_beta_next, t,
                                  t_next, gpu_beta_diff,
                                  block = (block_dim, 1, 1), grid = (num_block+1, 1))
        # Convergence value
        max_diff = cu_maxabs(gpu_beta_diff).get()
        max_coef = cu_maxabs(gpu_beta_next).get()

        # t_next = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t*t))
        # linalg.scale(((t - 1.0)/t_next), gpu_beta_diff)
        # gpu_aux_beta = gpu_beta_next + gpu_beta_diff

        # Values update
        t = t_next.get()
        gpu_beta = gpu_beta_next.copy()

        # Stopping rule (exit even if beta_next contains only zeros)
        # print("t:{}\n\n".format(t))
        if np.allclose(max_coef, 0.0) or (max_diff / max_coef) <= tolerance: break

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

def l1l2_regularization(data, labels, mu, tau, beta=None, kmax=100000,
                        tolerance=1e-5, return_iterations=False,
                        adaptive=False):
    r"""[PYCUDA]Implementation of the Fast Iterative Shrinkage-Thresholding Algorithm
    to solve a least squares problem with `l1l2` penalty.

    It solves the `l1l2` regularization problem with parameter ``mu`` on the
    `l2-norm` and parameter ``tau`` on the `l1-norm`.

    Parameters
    ----------
    data : (N, P) ndarray
        Data matrix.
    labels : (N,) or (N, 1) ndarray
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

    Examples
    --------
    >>> X = numpy.array([[0.1, 1.1, 0.3], [0.2, 1.2, 1.6], [0.3, 1.3, -0.6]])
    >>> beta = numpy.array([0.1, 0.1, 0.0])
    >>> Y = numpy.dot(X, beta)
    >>> beta = l1l2py.algorithms.l1l2_regularization(X, Y, 0.1, 0.1)
    >>> len(numpy.flatnonzero(beta))
    1

    """
    tic = time.time()
    d_data = gpuarray.to_gpu_async(data.astype(np.float32))
    d_labels = gpuarray.to_gpu_async(labels.astype(np.float32).reshape((data.shape[0],1)))
    print("\t[*** GPU data transfer time: {} ***]".format(time.time()-tic))

    return cu_l1l2_regularization(d_data, d_labels, mu, tau, beta=beta, kmax=kmax, tolerance=tolerance, return_iterations=return_iterations, adaptive=adaptive)

def l1l2_path(data, labels, mu, tau_range, beta=None, kmax=100000,
              tolerance=1e-5, adaptive=False):
    r"""[PYCUDA] Implementation of l1l2py.l1l2_path.

    Efficient solution of different `l1l2` regularization problems on
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
    from collections import deque
    n, p = data.shape

    if mu == 0.0:
        beta_ls = ridge_regression(data, labels)
    if beta is None:
        beta = np.zeros((p, 1))

    out = deque()
    nonzero = 0
    for tau in reversed(tau_range):
        if mu == 0.0 and nonzero >= n: # lasso saturation
            beta_next = beta_ls
        else:
            beta_next = l1l2_regularization(data, labels, mu, tau, beta,
                                            kmax, tolerance, adaptive=adaptive)

        nonzero = len(beta_next.nonzero()[0])
        if nonzero > 0:
            out.appendleft(beta_next)

        beta = beta_next

    return out
