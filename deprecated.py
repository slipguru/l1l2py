from algorithms import *
from algorithms import _soft_thresholding
from limbo import _functional, _sigma

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


def l1l2_regularization(data, labels, mu, tau, beta=None, kmax=1e5,
                        tolerance=1e-5, return_iterations=False):
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
    n, p = matrix.shape

    if p > n:
        tmp = np.dot(matrix, matrix.T)
        num = np.linalg.eigvalsh(tmp).max()
    else:
        tmp = np.dot(matrix.T, matrix)
        evals = np.linalg.eigvalsh(tmp)
        num = evals.max() + evals.min()

    return (num/(2.*n)) + mu
    
def l1l2_regularization_FISTA(data, labels, mu, tau, beta=None, kmax=1e5,
                              tolerance=1e-5, returns_iterations=False):
    n, d = data.shape

    # Useful quantities
    sigma = _sigma_F(data, mu)
    mu_s = mu / sigma
    tau_s = tau / sigma
    XT = data.T / (n * sigma)
    XTY = np.dot(XT, labels.reshape(-1, 1))

    # beta starts from 0 and we assume also that the previous value is 0
    if beta is None:
        beta = np.zeros_like(XTY)
    else:
        beta = beta.reshape(-1, 1)
    beta_prev = beta

    # Auxiliary beta (FISTA implementation), starts from 0
    aux_beta = beta
    t, t_next = 1., None     # t values initializations

    k, kmin = 0, 10
    th, difference = -np.inf, np.inf
    while k < kmin or ((difference > th).any() and k < kmax):
        k += 1

        # New solution
        value = (1.0 - mu_s)*aux_beta + XTY - np.dot(XT, np.dot(data, aux_beta))
        beta = _soft_thresholding(value, tau_s)

        # New auxiliary beta (FISTA)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t*t))
        difference = (beta - beta_prev)
        aux_beta = beta + ((t - 1.0)/t_next)*difference

        # Convergence values
        difference = np.abs(difference)
        th = np.abs(beta) * (tolerance / math.sqrt(k))
        
        # Values update
        beta_prev, t = beta, t_next

    if returns_iterations:
        return beta, k
    else:
        return beta
    
    
def l1l2_regularization_MFISTA(data, labels, mu, tau, beta=None, kmax=1e5,
                               tolerance=1e-5, returns_iterations=False):
    n = data.shape[0]

    # Useful quantities
    sigma = _sigma_F(data, mu)
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
    th, difference = -np.inf, np.inf
    while k < kmin or ((difference > th).any() and k < kmax):
        k += 1

        # New solution
        value = (1.0 - mu_s)*aux_beta + XTY - np.dot(XT, np.dot(data, aux_beta))
        beta_temp = _soft_thresholding(value, tau_s)
        value_temp = _functional(data, labels, beta_temp, tau, mu)

        # (M)FISTA step
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t*t))
        difference = (beta_temp - beta_prev)

        # MFISTA monotonicity check
        if value_temp <= value_prev:
            beta = beta_temp
            value = value_temp
            
            aux_beta = beta + ((t - 1.0)/t_next)*difference
            
        else:          
            beta = beta_prev
            value = value_prev
            
            aux_beta = beta + (t/t_next)*difference
            
        difference = np.abs(difference)
        th = np.abs(beta) * (tolerance / math.sqrt(k))
              
        # Values update
        beta_prev = beta
        value_prev = value
        t = t_next

    if returns_iterations:
        return beta, k, values
    else:
        return beta


def _sigma_F(matrix, mu):
    n, d = matrix.shape

    if d > n:
        tmp = np.dot(matrix, matrix.T)
    else:
        tmp = np.dot(matrix.T, matrix)

    return (np.linalg.eigvalsh(tmp).max()/n) + mu