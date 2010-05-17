r"""Internal algorithms implementation.

The :mod:`algorithms` module defines core numerical optimizazion algorithms:

* :func:`ridge_regression`
* :func:`l1_bounds`
* :func:`l1l2_regularization`
* :func:`l1l2_path`

"""

__all__ = ['l1_bounds', 'ridge_regression', 'l1l2_regularization', 'l1l2_path']

import numpy as np
import math

def l1_bounds(data, labels, eps=1e-5):
    r"""Estimation of useful bounds for the l1 penalty term.
    
    Finds the minimum and maximum value for the :math:`tau` parameter in the
    :math:`\ell_1\ell_2` regularization.
    
    Fixing :math:`\mu` close to `0.0`, and using the maximum value,
    the model will contain only one variable, instead
    using the minimum value, the model will contain (approximately) a number
    of variables equal to the number of different correlated groups
    of variables.
    
    .. warning
    
        ``data`` and ``labels`` parameters are assumed already normalized.
        That is, bounds are right if you run the :math:`\ell_1\ell_2`
        regularization algorithm with the same data.
    
    Parameters
    ----------
    data : (N, D) ndarray
        Data matrix.
    labels : (N,)  or (N, 1) ndarray
        Labels vector.
    eps : float, optional (default is `1e-5`)
        Correction parametr (see `Returns`).
        
    Returns
    -------
    tau_min : float
        Minimum tau + ``eps``
    tau_max : float
        Maximum tau - ``eps``
        
    Raises
    ------
    ValueError
        If ``tau_max`` or ``tau_min`` are negative or ``tau_max`` <= ``tau_min``
    
    """
    n = data.shape[0]
    corr = np.abs(np.dot(data.T, labels))

    tau_min = (corr.min() * (2.0/n)) + eps    
    tau_max = (corr.max() * (2.0/n)) - eps
      
    if tau_max < 0:
        raise ValueError("'eps' has produced negative 'tau_max'")
    if tau_min < 0:
        raise ValueError("'eps' has produced negative 'tau_min'")
    if tau_max <= tau_min:
        raise ValueError("'eps' has produced 'tau_max' less or equal 'tau_min'")
    
    return tau_min, tau_max

def ridge_regression(data, labels, mu=0.0):
    r"""Implementation of Regularized Least Squares.

    Finds the RLS model with ``mu`` parameter associated with its
    :math:`\ell_2` norm (see `Notes`).

    Parameters
    ----------
    data : (N, D) ndarray
        Data matrix.
    labels : (N,)  or (N, 1) ndarray
        Labels vector.
    mu : float, optional (default is `0.0`)
        :math:`\ell_2` norm penalty.

    Returns
    --------
    beta : (D,) or (D, 1) ndarray
        RLS model.

    Examples
    --------
    >>> X = numpy.array([[0.1, 1.1, 0.3], [0.2, 1.2, 1.6], [0.3, 1.3, -0.6]])
    >>> beta = numpy.array([0.1, 0.1, 0.0])
    >>> y = numpy.dot(X, beta)
    >>> beta_rls = biolearning.algorithms.ridge_regression(X, y)
    >>> numpy.allclose(beta, beta_rls)
    True

    """
    n, d = data.shape

    if n < d:
        tmp = np.dot(data, data.T)
        if mu:
            tmp += mu*n*np.eye(n)
        tmp = np.linalg.pinv(tmp)

        return np.dot(np.dot(data.T, tmp), labels)
    else:
        tmp = np.dot(data.T, data)
        if mu:
            tmp += mu*n*np.eye(d)
        tmp = np.linalg.pinv(tmp)

        return np.dot(tmp, np.dot(data.T, labels))

def l1l2_path(data, labels, mu, tau_range, beta=None, kmax=1e5,
              tolerance=1e-6):
    r"""Implementation of Regularized Least Squares path with
    :math:`\ell_1\ell_2` penalty.

    Finds the :math:`\ell_1\ell_2` regularization path for each value in
    ``tau_range`` and fixed value of ``mu``.

    The values in ``tau_range`` are used during the computation in reverse
    order, while the output path has the same ordering of the :math:`\tau`
    values.

    .. warning ::

        The number of models can differ the number of :math:`\tau` values.
        The functions returns only the model with at least one non-zero feature.
        For very high value of :math:`\tau` a model could have all `0s`.

    Parameters
    ----------
    data : (N, D) ndarray
        Data matrix.
    labels : (N,) or (N, 1) ndarray
        Labels vector.
    mu : float
        :math:`\ell_2` norm penalty.
    tau_range : array_like of float
        :math:`\ell_1` norm penalties.
    beta : (D,) or (D, 1) ndarray, optional (default is `None`)
        Starting value of the iterations
        (see :func:`l1l2_regularization` `Notes`).
        If `None`, then iterations starts from the empty model.
    kmax : int, optional (default is :math:`10^5`)
        Maximum number of iterations.
    tolerance : float, optional (default is :math:`10^{-6}`)
        Convergence tolerance.
        (see :func:`l1l2_regularization` `Notes`).

    Returns
    -------
    beta_path : list of (D,) or (D, 1) ndarrays
        :math:`\ell_1\ell_2` models with at least one nonzero feature.

    """
    from collections import deque
    n, d = data.shape

    if mu == 0.0:
        beta_ls = ridge_regression(data, labels)
    if beta is None:
        beta = np.zeros((d, 1))

    out = deque()
    nonzero = 0 # <<---- CHECK!!!
    for tau in reversed(tau_range):
        if mu == 0.0 and nonzero >= n: # lasso saturation
            beta_next = beta_ls
        else:
            beta_next = l1l2_regularization(data, labels, mu, tau, beta,
                                            kmax, tolerance)

        if len(beta_next.nonzero()[0]) > 0:
            out.appendleft(beta_next)

        beta = beta_next

    return out

def l1l2_regularization(data, labels, mu, tau, beta=None, kmax=1e5,
                        tolerance=1e-5, returns_iterations=False):
    r"""Implementation of Regularized Least Squares with
    :math:`\ell_1\ell_2` penalty.

    Finds the :math:`\ell_1\ell_2` model with ``mu`` parameter associated with
    its :math:`\ell_2` norm and ``tau`` parameter associated with its
    :math:`\ell_1` norm (see `Notes`).
    
    Parameters
    ----------
    data : (N, D) ndarray
        Data matrix.
    labels : (N,) or (N, 1) ndarray
        Labels vector.
    mu : float
        :math:`\ell_2` norm penalty.
    tau : float
        :math:`\ell_1` norm penalty.
    beta : (D,) or (D, 1) ndarray, optional (default is `None`)
        Starting value of the iterations (see `Notes`).
        If `None`, then iterations starts from the empty model.
    kmax : int, optional (default is :math:`10^5`)
        Maximum number of iterations.
    tolerance : float, optional (default is :math:`10^{-6}`)
        Convergence tolerance (see `Notes`).
    returns_iterations : bool, optional (default is `False`)
        If `True`, returns the number of iterations performed.
        The algorithm has a predefined minimum number of iterations
        equal to `10`.
        
    Returns
    -------
    beta : (D,) or (D, 1) ndarray
        :math:`\ell_1\ell_2` model.
    k : int, optional
        Number of iterations performed.
        
   
    Examples
    --------
    >>> X = numpy.array([[0.1, 1.1, 0.3], [0.2, 1.2, 1.6], [0.3, 1.3, -0.6]])
    >>> beta = numpy.array([0.1, 0.1, 0.0])
    >>> y = numpy.dot(X, beta)
    >>> beta_rls = biolearning.algorithms.l1l2_regularization(X, y, 0.0, 1e-5)
    >>> numpy.allclose(beta, beta_rls)
    True
    >>> biolearning.algorithms.l1l2_regularization(X, y, 0.1, 0.1)
    array([ 0.        ,  0.04482757,  0.        ])

    """
    n = data.shape[0]

    # Useful quantities
    sigma = _sigma(data, mu)
    mu_s = mu / sigma
    tau_s = tau / sigma
    XT = data.T / (n * sigma)
    XTY = np.dot(XT, labels)

    if beta is None:
        beta = np.zeros_like(XTY)

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

    if returns_iterations:
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
