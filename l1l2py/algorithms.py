r"""Internal algorithms implementations.

This module contains the functions strictly related with the statistical
elaboration of the data.

"""

__all__ = ['l1_bound', 'ridge_regression', 'l1l2_regularization', 'l1l2_path']

import numpy as np
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
    data : (N, D) ndarray
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
    >>> l1l2py.algorithms.l1l2_regularization(X, Y, 0.0, tau_max - 1e-5).T
    array([[  0.00000000e+00,   3.45622120e-06,   0.00000000e+00]])

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
    data : (N, D) ndarray
        Data matrix.
    labels : (N,)  or (N, 1) ndarray
        Labels vector.
    mu : float, optional (default is `0.0`)
        `l2-norm` penalty.

    Returns
    --------
    beta : (D, 1) ndarray
        Ridge regression solution.

    Examples
    --------
    >>> X = numpy.array([[0.1, 1.1, 0.3], [0.2, 1.2, 1.6], [0.3, 1.3, -0.6]])
    >>> beta = numpy.array([0.1, 0.1, 0.0])
    >>> Y = numpy.dot(X, beta)
    >>> l1l2py.algorithms.ridge_regression(X, Y, 1e3).T
    array([[  2.92871765e-05,   1.69054825e-04,   5.45274610e-05]])
    >>> beta_ls = l1l2py.algorithms.ridge_regression(X, Y).T
    >>> numpy.allclose(beta, beta_ls.squeeze())
    True
    """
    n, d = data.shape

    if n < d:
        tmp = np.dot(data, data.T)
        if mu:
            tmp += mu*n*np.eye(n)
        tmp = np.linalg.pinv(tmp)

        return np.dot(np.dot(data.T, tmp), labels.reshape(-1, 1))
    else:
        tmp = np.dot(data.T, data)
        if mu:
            tmp += mu*n*np.eye(d)
        tmp = np.linalg.pinv(tmp)

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
        values is higher than `N` for a given value of tau (that means k has
        reached the limit of allowed iterations), the following solutions (for
        smaller values of ``tau``) are simply the least squares solutions.

    .. warning ::

        The number of solutions can differ from ``len(tau_range)``.
        The function returns only the solutions with at least one non-zero
        element.
        For values higher than *tau_max* a solution could have all zero values.

    Parameters
    ----------
    data : (N, D) ndarray
        Data matrix.
    labels : (N,) or (N, 1) ndarray
        Labels vector.
    mu : float
        `l2-norm` penalty.
    tau_range : array_like of float
        `l1-norm` penalties in increasing order.
    beta : (D,) or (D, 1) ndarray, optional (default is `None`)
        Starting value of the iterations.
        If `None`, then iterations starts from the empty model.
    kmax : int, optional (default is `1e5`)
        Maximum number of iterations.
    tolerance : float, optional (default is `1e-6`)
        Convergence tolerance.

    Returns
    -------
    beta_path : list of (D,) or (D, 1) ndarrays
        `l1l2` solutions with at least one non-zero element.

    """
    from collections import deque
    n, d = data.shape

    if mu == 0.0:
        beta_ls = ridge_regression(data, labels)
    if beta is None:
        beta = np.zeros((d, 1))

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

def l1l2_regularization(data, labels, mu, tau, beta=None, kmax=1e5,
                        tolerance=1e-5, return_iterations=False):
    r"""Implementation of the Iterative Shrinkage-Thresholding Algorithm
    to solve a least squares problem with `l1l2` penalty.

    It solves the `l1l2` regularization problem with parameter ``mu`` on the
    `l2-norm` and parameter ``tau`` on the `l1-norm`.

    Parameters
    ----------
    data : (N, D) ndarray
        Data matrix.
    labels : (N,) or (N, 1) ndarray
        Labels vector.
    mu : float
        `l2-norm` penalty.
    tau : float
        `l1-norm` penalty.
    beta : (D,) or (D, 1) ndarray, optional (default is `None`)
        Starting value for the iterations.
        If `None`, then iterations starts from the empty model.
    kmax : int, optional (default is `1e5`)
        Maximum number of iterations.
    tolerance : float, optional (default is `1e-6`)
        Convergence tolerance.
    return_iterations : bool, optional (default is `False`)
        If `True`, returns the number of iterations performed.
        The algorithm has a predefined minimum number of iterations
        equal to `10`.

    Returns
    -------
    beta : (D, 1) ndarray
        `l1l2` solution.
    k : int, optional
        Number of iterations performed.

    Examples
    --------
    >>> X = numpy.array([[0.1, 1.1, 0.3], [0.2, 1.2, 1.6], [0.3, 1.3, -0.6]])
    >>> beta = numpy.array([0.1, 0.1, 0.0])
    >>> Y = numpy.dot(X, beta)
    >>> l1l2py.algorithms.l1l2_regularization(X, Y, 0.1, 0.1).T
    array([[ 0.        ,  0.07715517,  0.        ]])
    >>> beta_ls = l1l2py.algorithms.l1l2_regularization(X, Y, 0.0, 0.0).T
    >>> numpy.allclose(beta, beta_ls.squeeze())   
    True

    """
    n = data.shape[0]

    # Useful quantities
    sigma = _sigma(data, mu)
    mu_s = mu / sigma
    tau_s = tau / sigma
    XT = data.T / (n * sigma)
    XTY = np.dot(XT, labels.reshape(-1, 1))

    if beta is None:
        beta = np.zeros_like(XTY)
    else:
        beta = beta.reshape(-1, 1)

    k, kmin = 0, 100
    th, difference = -np.inf, np.inf
    while k < kmin or ((difference > th).any() and k < kmax):
        k += 1

        value = beta +  XTY - np.dot(XT, np.dot(data, beta))
        beta_next = _soft_thresholding(value, tau_s) / (1.0 + mu_s)

        # Convergence values
        difference = np.abs(beta_next - beta)
        th = np.abs(beta) * (tolerance / k)

        beta = beta_next

    if return_iterations:
        return beta, k
    else:
        return beta

def _sigma(matrix, mu):
    n, d = matrix.shape

    if d > n:
        tmp = np.dot(matrix, matrix.T)
        num = np.linalg.eigvalsh(tmp).max()
    else:
        tmp = np.dot(matrix.T, matrix)
        evals = np.linalg.eigvalsh(tmp)
        num = evals.max() + evals.min()

    return (num/(2.*n)) + mu

def _soft_thresholding(x, th):
    return np.sign(x) * np.maximum(0, np.abs(x) - th/2.0)
