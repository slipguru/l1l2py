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

import numpy as np
try:
    from scipy import linalg as la
except ImportError:
    from numpy import linalg as la
    
from itertools import izip
import math

def l1_bound(data, labels):
    r"""Estimation of an useful maximum bound for the `l1` penalty term.

    Fixing ``mu`` close to `0.0` and using the maximum value calculated with
    this function as ``tau`` in the `l1l2` regularization, the solution vector
    contains only zero elements.

    For each value of ``tau`` smaller than the maximum bound the solution vector
    contains at least one non zero element.

    .. warning

        That is, bounds are right if you run the `l1l2` regularization
        algorithm with the same data matrices.

    Parameters
    ----------
    data : (N, P) ndarray
        Data matrix.
    labels : (N,)  or (N, 1) ndarray
        Labels vector.

    Returns
    -------
    tau_max : float
        Maximum ``tau``.

    Examples
    --------
    >>> X = numpy.array([[0.1, 1.1, 0.3], [0.2, 1.2, 1.6], [0.3, 1.3, -0.6]])
    >>> beta = numpy.array([0.1, 0.1, 0.0])
    >>> Y = numpy.dot(X, beta)
    >>> tau_max = l1l2py.algorithms.l1_bound(X, Y)
    >>> l1l2py.algorithms.l1l2_regularization(X, Y, 0.0, tau_max).T
    array([[ 0.,  0.,  0.]])
    >>> beta = l1l2py.algorithms.l1l2_regularization(X, Y, 0.0, tau_max - 1e-5)
    >>> len(numpy.flatnonzero(beta))
    1

    """
    n = data.shape[0]
    corr = np.abs(np.dot(data.T, labels))

    tau_max = (corr.max() * (2.0/n))

    return tau_max

def ridge_regression(data, labels, mu=0.0):
    r"""Implementation of the Regularized Least Squares solver.

    It solves the ridge regression problem with parameter ``mu`` on the
    `l2-norm`.

    Parameters
    ----------
    data : (N, P) ndarray
        Data matrix.
    labels : (N,)  or (N, 1) ndarray
        Labels vector.
    mu : float, optional (default is `0.0`)
        `l2-norm` penalty.

    Returns
    --------
    beta : (P, 1) ndarray
        Ridge regression solution.

    Examples
    --------
    >>> X = numpy.array([[0.1, 1.1, 0.3], [0.2, 1.2, 1.6], [0.3, 1.3, -0.6]])
    >>> beta = numpy.array([0.1, 0.1, 0.0])
    >>> Y = numpy.dot(X, beta)
    >>> beta = l1l2py.algorithms.ridge_regression(X, Y, 1e3).T
    >>> len(numpy.flatnonzero(beta))
    3

    """
    n, p = data.shape

    if n < p:
        tmp = np.dot(data, data.T)
        if mu:
            tmp += mu*n*np.eye(n)
        tmp = la.pinv(tmp)

        return np.dot(np.dot(data.T, tmp), labels.reshape(-1, 1))
    else:
        tmp = np.dot(data.T, data)
        if mu:
            tmp += mu*n*np.eye(p)
        tmp = la.pinv(tmp)

        return np.dot(tmp, np.dot(data.T, labels.reshape(-1, 1)))

def l1l2_path(data, labels, mu, tau_range, beta=None, kmax=1e5,
              tolerance=1e-5):
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
                                            kmax, tolerance)

        nonzero = len(beta_next.nonzero()[0])
        if nonzero > 0:
            out.appendleft(beta_next)

        beta = beta_next

    return out

def l1l2_regularization(data, labels, mu, tau,
                        beta=None, kmax=1e5,
                        tolerance=1e-5,
                        continuation=True,
                        adaptive=True,
                        nonzero=False):
    r"""Implementation of the Fast Iterative Shrinkage-Thresholding Algorithm
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
    n, d = data.shape
    
    # beta starts from 0 and we assume also that the previous value is 0
    if beta is None:
        beta = np.zeros(d)
    else:
        beta.ravel()

    # Useful quantities
    X = data
    Y = labels.ravel()
    XTn = data.T / n
    
    if n > d:
        XTYn = np.dot(XTn, Y)
    
    # Continuation strategy
    if continuation:
        tau_max = l1_bound(X, Y) * 0.8
        if tau < tau_max:
            taus_len = int(10**np.log10(tau_max) - 10**np.log10(tau))
            taus = np.logspace(np.log10(tau), np.log10(tau_max), taus_len)[::-1]
        else:
            taus_len = 1
            taus = [tau]
    else:
        taus_len = 1
        taus = [tau]
    iterations = 0
    
    # First iteration with standard sigma
    sigma_0 = _sigma(data, mu)
    mu_s_0 = mu / sigma_0
        
    for i in xrange(taus_len):    
        # Restart conditions
        tau = taus[i]
        aux_beta = beta # warm-start
        sigma = sigma_0
        mu_s = mu_s_0
        tau_s = tau / (2.0*sigma_0)
        t = 1.
           
        for k in xrange(kmax):            
            # Pre-calculated "heavy" computation
            if n > d:
                precalc = XTYn - np.dot(XTn, np.dot(X, aux_beta))
            else:
                precalc = np.dot(XTn, Y - np.dot(X, aux_beta))
            
            # Soft-Thresholding
            value = (precalc / sigma) + ((1.0 - mu_s) * aux_beta)
            beta_next = value * np.maximum(0, 1 - tau_s/np.abs(value))
                   
            ######## Adaptive step size #######################################
            if adaptive:
                beta_diff = (aux_beta - beta_next)
                
                # Only if there is an increment of the solution
                # we can calculate the adaptive step-size
                if np.any(beta_diff):                    
                    # grad_diff = np.dot(XTn, np.dot(X, beta_diff))
                    # num = np.dot(beta_diff, grad_diff)
                    tmp = np.dot(X, beta_diff) ##### <-- heavy
                    num = np.dot(tmp, tmp) / n                    
                    
                    sigma = (num / np.dot(beta_diff, beta_diff))
                    mu_s = mu / sigma
                    tau_s = tau / (2.0*sigma)
                
                    # Soft-Thresholding
                    value = (precalc / sigma) + ((1.0 - mu_s) * aux_beta)
                    beta_next = value * np.maximum(0, 1 - tau_s/np.abs(value))
            
            ######## FISTA ####################################################
            beta_diff = (beta_next - beta)
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t*t))
            aux_beta = beta_next + ((t - 1.0)/t_next)*beta_diff
            
            # Convergence values        
            max_diff = np.abs(beta_diff).max()
            max_coef = np.abs(beta_next).max()
            tol = tolerance if i == (taus_len-1) else 1e-2
            
            # Stopping rule
            if (max_diff / max_coef) <= tol: break
    
            # Values update
            t = t_next
            beta = beta_next
        
        iterations += k

    return beta, iterations, None
    #return beta, k, energy

def _functional(X, Y, beta, tau, mu):
    n = X.shape[0]

    loss = Y - np.dot(X, beta)
    loss_quadratic_norm = np.linalg.norm(loss) ** 2
    beta_quadratic_norm = np.linalg.norm(beta) ** 2
    beta_l1_norm = np.abs(beta).sum()

    return (((1./n) * loss_quadratic_norm)
             + mu * beta_quadratic_norm
             + tau * beta_l1_norm)

def _sigma(matrix, mu):
    n, p = matrix.shape

    if p > n:
        tmp = np.dot(matrix, matrix.T)
    else:
        tmp = np.dot(matrix.T, matrix)

    return (la.norm(tmp, 2)/n) + mu


def _soft_thresholding(x, th):
    return np.sign(x) * np.maximum(0, np.abs(x) - th)#/2.0)
    #return np.sign(x) * np.clip(np.abs(x) - th, 0, np.inf)
