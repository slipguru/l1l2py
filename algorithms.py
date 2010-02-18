"""
TODO: Add docstring
"""

import numpy as np

__all__ = ['ridge_regression', 'elastic_net', 'elastic_net_regpath']

def ridge_regression(data, labels, penalty=0.0):
    """ TODO: Add docstring """
    n, d = data.shape
        
    if n < d:
        tmp = np.dot(data, data.T)
        if penalty:
            tmp += penalty*n*np.eye(n)
        tmp = np.linalg.pinv(tmp)
        
        return np.dot(np.dot(data.T, tmp), labels)
    else:
        tmp = np.dot(data.T, data)
        if penalty:
            tmp += penalty*n*np.eye(d)
        tmp = np.linalg.pinv(tmp)
        
        return np.dot(tmp, np.dot(data.T, labels))

def elastic_net(data, labels, mu, tau, beta=None, kmax=1e5):
    """ TODO: Add docstring """
    n = data.shape[0]
    
    sigma_0 = _get_sigma(data)
    mu = mu*sigma_0
    sigma = sigma_0 + mu
    
    mu_s = mu / sigma
    tau_s = tau / sigma
    dataT = data.T / (n*sigma)
        
    if beta is None:
        beta = ridge_regression(data, labels)
    #--------------------------------------------------------------------------
    # The loop is slower than matlab in the worst case (saturation)!
    # Need to push down (C/C++ code)!
    k, kmin, tol = 0, 100, 0.01
    th, difference = -np.inf, np.inf
    while k < kmin or ((difference > th).any() and k < kmax):
        k += 1
        
        value = beta*(1 - mu_s) + np.dot(dataT, (labels - np.dot(data, beta))) 
        beta_next = _soft_thresholding(value, tau_s)
        
        difference = np.abs(beta_next - beta)
        th = np.abs(beta) * (tol / k)
        beta = beta_next
    #--------------------------------------------------------------------------
    return beta_next, k
  
def elastic_net_regpath(data, labels, mu, tau_range, beta=None, kmax=np.inf):
    """
    TODO: Add docstring
    reg_path
    Is sufficient document the possibility to get
    a shorter list of beta without using annoying warnings
    """
    from collections import deque
    n = data.shape[0]
    
    beta_ls = ridge_regression(data, labels)
    if beta is None:
        beta = beta_ls
    
    out = deque()
    nonzero = 0
    for tau in reversed(tau_range):
        if mu == 0.0 and nonzero >= n: # lasso + saturation
            beta_next = beta_ls                
        else:
            beta_next = elastic_net(data, labels, mu, tau, beta, kmax)[0]
        
        nonzero = np.sum(beta_next != 0)
        if nonzero > 0:
            out.appendleft(beta_next)
            
        beta = beta_next
                   
    return out

def _get_sigma(matrix):
    """ TODO: Add docstring """
    n, d = matrix.shape
    
    if d > n:
        a = np.linalg.norm(np.dot(matrix, matrix.T), 2)
        b = 0
    else:
        aval = np.linalg.svd(np.dot(matrix.T, matrix),
                             full_matrices=False, compute_uv=False)
        a, b = aval[(0, -1)]
    
    return (a+b)/(n*2.0)
      
def _soft_thresholding(x, th):
    """ TODO: Add docstring """
    out = x - (np.sign(x) * (th/2.0))
    out[np.abs(x) < (th/2.0)] = 0.0
    return out