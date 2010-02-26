r"""Internal algorithms implementations.

The :mod:`algorithms` module defines core numerical optimizazion algorithms:

* :func:`ridge_regression`
* :func:`elastic_net`
* :func:`elastic_net_regpath`

"""

import numpy as np

__all__ = ['ridge_regression', 'elastic_net', 'elastic_net_regpath']

def ridge_regression(data, labels, mu=0.0):
    r"""Regularized Least Squares.

    Finds the RLS model with ``mu`` parameter associated with its
    :math:`\ell_2` norm (see `Notes`).

    Parameters
    ----------
    data : (N, D) ndarray
        Data matrix.
    labels : (N,) ndarray
        Labels vector.
    mu : float, optional (default is `0.0`)
        :math:`\ell_2` norm penalty.
    
    Returns
    -------
    beta : (D,) ndarray
        RLS model.
    
    Notes
    -----
    RLS minimizes the following objective function:    
    
    .. math::
    
        \frac{1}{N} \| Y - X\beta \|_2^2 + \mu \|\beta\|_2^2
    
    finding the optimal model :math:`\beta^*`, where
    
    =============== ===============
    :math:`X`       ``data``
    --------------- ---------------
    :math:`Y`       ``labels``
    --------------- ---------------
    :math:`\mu`     ``mu``      
    --------------- ---------------
    :math:`\beta^*` ``beta``    
    =============== ===============
    
    Using ``mu`` = `0.0` the algorithm performs Ordinary Least Squares (OLS).
       
    Examples
    --------
    >>> X = numpy.array([[0.1, 1.1, 0.3], [0.2, 1.2, 1.6], [0.3, 1.3, -0.6]])
    >>> beta = numpy.array([0.1, 0.1, 0.0])
    >>> y = numpy.dot(X, beta)
    >>> biolearning.algorithms.ridge_regression(X, y)
    array([  1.00000000e-01,   1.00000000e-01,   6.62515290e-17])
    
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

def elastic_net_regpath(data, labels, mu, tau_range, beta=None, kmax=np.inf):
    """
    TODO: Add docstring
    reg_path
    Is sufficient document the possibility to get
    a shorter list of beta without using annoying warnings
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
            beta_next = elastic_net(data, labels, mu, tau, beta, kmax)
          
        if len(beta_next.nonzero()[0]) > 0:
            out.appendleft(beta_next)
            
        beta = beta_next
                   
    return out

def elastic_net(data, labels, mu, tau, beta=None, kmax=1e5, step_size=None,
                returns_iterations=False):
    """
    
    """
    n = data.shape[0]
    
    if step_size is None:
        step_size = _step_size(data)    # 1/C
    mu_s = (n * mu) * step_size         # (n mu)/C
    tau_s = (n * tau) * step_size       # (n tau)/C
    
    # Initializations
    XT = data.T * step_size             # X^T / C
    XTY = np.dot(XT, labels)            # (X^T Y) / C
           
    if beta is None:
        beta = ridge_regression(data, labels)
    #--------------------------------------------------------------------------
    # The loop is slower than matlab in the worst case (saturation)!
    # Need to push down (C/C++ code)!
    k, kmin, tol = 0, 100, 0.01
    th, difference = -np.inf, np.inf
    while k < kmin or ((difference > th).any() and k < kmax):
        k += 1
        
        value = beta +  XTY - np.dot(XT, np.dot(data, beta))
        beta_next = _soft_thresholding(value, tau_s) / (1.0 + mu_s)
                        
        difference = np.abs(beta_next - beta)
        th = np.abs(beta) * (tol / k)
        beta = beta_next
    #--------------------------------------------------------------------------
    if returns_iterations:
        return beta_next, k
    else:
        return beta

def _step_size(matrix):
    n, d = matrix.shape
    
    if d > n:
        eigvals = np.linalg.eigvalsh(np.dot(matrix, matrix.T))
        max_eig, min_eig = eigvals.max(), 0.0
    else:
        eigvals = np.linalg.eigvalsh(np.dot(matrix.T, matrix))
        max_eig, min_eig = eigvals.max(), eigvals.min()
        
    return 2.0 / (max_eig + min_eig)
      
def _soft_thresholding(x, th):
    """ TODO: Add docstring """
    out = x - (np.sign(x) * (th / 2.0))
    out[np.abs(x) < (th / 2.0)] = 0.0
    return out