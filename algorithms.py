r"""Internal algorithms implementation.

The :mod:`algorithms` module defines core numerical optimizazion algorithms:

* :func:`ridge_regression`
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
    -------
    beta : (D,) or (D, 1) ndarray
        RLS model.

    Notes
    -----
    RLS minimizes the following objective function:

    .. math::

        \frac{1}{N} \| Y - X\beta \|_2^2 + \mu \|\beta\|_2^2

    finding the optimal model :math:`\beta^*`, where :math:`X` is the ``data``
    matrix and :math:`Y` contains the ``labels``.

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

    See Also
    --------
    l1l2_regularization

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

        if len(beta_next.nonzero()[0]) > 0:
            out.appendleft(beta_next)

        beta = beta_next

    return out

def l1l2_regularization(data, labels, mu, tau, beta=None, kmax=1e5,
                        tolerance=1e-6, returns_iterations=False):
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

    See Also
    --------
    l1l2_path

    Notes
    -----
    :math:`\ell_1\ell_2` minimizes the following objective function:

    .. math::

        \frac{1}{N} \| Y - X\beta \|_2^2 + \mu \|\beta\|_2^2 + \tau \|\beta\|_1

    finding the optimal model :math:`\beta^*`, where :math:`X` is the ``data``
    matrix and :math:`Y` contains the ``labels``.

    The computation is iterative, each step updates the value of :math:`\beta`
    until the convergence is reached [DeMol08]_:

    .. math::

        \beta^{(k+1)} = \mathbf{S}_{\frac{\tau}{\sigma}} (
                            (1 - \frac{\mu}{\sigma})\beta^k +
                            \frac{1}{n\sigma}X^T[Y - X\beta^k]
                        )

    where, :math:`\mathbf{S}_{\gamma > 0}` is the soft-thresholding function

    .. math::

        \mathbf{S}_{\gamma}(x) = sign(x) max(0, |x| - \frac{\gamma}{2})

    Moreover, the function implements a *MFISTA* [Beck09]_ modification, wich
    increases with quadratic factor the convergence rate of the algorithm.

    The constant :math:`\sigma` is a (theorically optimal) step size wich
    depends by the data:

    .. math::

        \sigma = \frac{\|X^T X\|}{N} + \mu

    The convergence is reached when:

    .. math::

        \|\beta^k - \beta^{k-1}\| \leq \|\beta^k\| * tolerance

    but the algorithm will be stop when the maximum number of iteration
    is reached.

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
    sigma = _maximum_eigenvalue(data)/n + mu
    mu_s = mu / sigma
    tau_s = tau / sigma
    XT = data.T / (n * sigma)
    XTY = np.dot(XT, labels)

    # beta starts from 0 and we assume the previous value is also 0
    if beta is None:
        beta = np.zeros_like(XTY)
    beta_prev = beta
    value_prev = _functional(data, labels, beta_prev, tau, mu)

    # Auxiliary beta (FISTA implementation), starts from 0
    aux_beta = beta
    t, t_next = 1., None     # t values initializations

    k, kmin = 0, 10
    th, distance = -np.inf, np.inf
    while k < kmin or (distance > th and k < kmax):
        k += 1

        # New solution
        value = (1.0 - mu_s)*aux_beta + XTY - np.dot(XT, np.dot(data, aux_beta))
        beta_temp = _soft_thresholding(value, tau_s)
        value_temp = _functional(data, labels, beta_temp, tau, mu)

        # (M)FISTA step
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t*t))
        difference = (beta_temp - beta_prev)
        distance = np.linalg.norm(difference)

        # MFISTA monotonicity check
        if value_temp <= value_prev:
            beta = beta_temp
            value = value_temp
            
            aux_beta = beta + ((t - 1.0)/t_next)*difference
        else:
            beta = beta_prev
            value = value_prev

            aux_beta = beta + (t/t_next)*difference

        # Convergence threshold
        th = np.linalg.norm(beta) * tolerance

        # Values update
        beta_prev = beta
        value_prev = value
        t = t_next

    if returns_iterations:
        return beta, k
    else:
        return beta

def _maximum_eigenvalue(matrix):
    n, d = matrix.shape

    if d > n:
        tmp = np.dot(matrix, matrix.T)
    else:
        tmp = np.dot(matrix.T, matrix)

    return np.linalg.eigvalsh(tmp).max()

def _soft_thresholding(x, th):
    out = x - (np.sign(x) * th/2.0)
    out[np.abs(x) < th/2.0] = 0.0
    return out

def _functional(X, Y, beta, tau, mu):
    n = X.shape[0]

    loss = Y - np.dot(X, beta)
    loss_quadratic_norm = (loss * loss).sum()
    beta_quadratic_norm = (beta * beta).sum()
    beta_l1_norm = np.abs(beta).sum()

    return (loss_quadratic_norm/n + mu  * beta_quadratic_norm
                                  + tau * beta_l1_norm)
