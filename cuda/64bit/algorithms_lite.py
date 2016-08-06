import numpy as np
import numpy.linalg as la

from math import sqrt

def l1l2_regularization(data, labels, mu, tau, beta=None, kmax=800,
                        tolerance=1e-5, return_iterations=False,
                        adaptive=False):
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
    n, d = data.shape

    # beta starts from 0 and we assume also that the previous value is 0
    if beta is None:
        beta = np.zeros(d)
    else:
        beta = beta.ravel()

    # Useful quantities
    X = data
    Y = labels.ravel()

    if n > d:
        XTY = np.dot(X.T, Y)

    # First iteration with standard sigma
    sigma = _sigma(data, mu)
    
    print sigma
    
    if sigma < np.finfo(float).eps: # is zero...
        return np.zeros(d), 0

    mu_s = mu / sigma
    tau_s = tau / (2.0 * sigma)
    nsigma = n * sigma

    # Starting conditions
    aux_beta = beta
    t = 1.

    for k in xrange(kmax):
        # Pre-calculated "heavy" computation
        if n > d:
            precalc = XTY - np.dot(X.T, np.dot(X, aux_beta))
        else:
            precalc = np.dot(X.T, Y - np.dot(X, aux_beta))

        # Soft-Thresholding
        value = (precalc / nsigma) + ((1.0 - mu_s) * aux_beta)
        beta_next = np.sign(value) * np.clip(np.abs(value) - tau_s, 0, np.inf)

        ######## FISTA ####################################################
        beta_diff = (beta_next - beta)
        t_next = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t*t))
        aux_beta = beta_next + ((t - 1.0)/t_next)*beta_diff

        # Convergence values
        # max_diff = np.abs(beta_diff).max()
        # max_coef = np.abs(beta_next).max()

        # Values update
        t = t_next
        beta = beta_next

        # Stopping rule (exit even if beta_next contains only zeros)
        # if max_coef == 0.0 or (max_diff / max_coef) <= tolerance: break

    if return_iterations:
        return beta.reshape(-1, 1), k+1
    return beta.reshape(-1, 1)

def _sigma(matrix, mu):
    n, p = matrix.shape

    if p > n:
        tmp = np.dot(matrix, matrix.T)
    else:
        tmp = np.dot(matrix.T, matrix)

    return (la.norm(tmp, 2)/n) + mu
