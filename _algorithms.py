import numpy as np

__all__ = ['ridge_regression', 'elastic_net', 'elastic_net_regpath']

def ridge_regression(X, Y, penalty=0.0):
    n, d = X.shape
        
    if n < d:
        tmp = np.dot(X, X.T)
        if penalty: tmp += penalty*n*np.eye(n)
        tmp = np.linalg.pinv(tmp)
        
        return np.dot(np.dot(X.T, tmp), Y)
    else:
        tmp = np.dot(X.T, X)
        if penalty: tmp += penalty*n*np.eye(d)
        tmp = np.linalg.pinv(tmp)
        
        return np.dot(tmp, np.dot(X.T, Y))

def elastic_net(X, Y, mu, tau, beta=None, kmax=1e5):
    n, d = X.shape
    
    sigma_0 = _get_sigma(X)
    mu = mu*sigma_0
    sigma = sigma_0 + mu
    mu_s = mu / sigma
    tau_s = tau / sigma
    XT = X.T / (n*sigma)
    
    kmin = 100
    k = 0
    tol = 0.01
    
    if beta is None:
        beta = ridge_regression(X, Y)
      
    #--------------------------------------------------------------------------
    # The loop is 3x slower than matlab!
    # Need to push down (C/C++ code)!    
    value = beta * (1 - mu_s) + np.dot(XT, (Y - np.dot(X, beta)))
    beta_next = _soft_thresholding(value, tau_s)
    log = True
    
    while k < kmin or (k < kmax and log is True):
        th = np.abs(beta) * (tol / (k+1))
        if (np.abs(beta_next - beta) <= th).all(): log = False
        
        beta = beta_next
        value = beta * (1 - mu_s) + np.dot(XT, (Y - np.dot(X, beta)))      
        beta_next = _soft_thresholding(value, tau_s)
        k += 1
    #--------------------------------------------------------------------------
    
    return beta_next, k
  
def elastic_net_regpath(X, Y, mu, tau_range, beta=None, kmax=np.inf):
    """ reg_path """
    n, d = X.shape
    
    beta_ls = ridge_regression(X, Y)
    if beta is None:
        beta = beta_ls
        
    out = np.empty((len(tau_range), beta.size))    
    sparsity = 0
    for i, t in _reverse_enumerate(tau_range):
        if mu == 0.0 and sparsity >= n: #??
            beta_next = beta_ls                
        else:
            beta_next, k = elastic_net(X, Y, mu, t, beta, kmax)
        out[i,:] = beta_next.squeeze()
        sparsity = np.sum(beta_next != 0)
        beta = beta_next
       
    return out

def _get_sigma(X):
    n, d = X.shape
    
    if d > n:
        a = np.linalg.norm(np.dot(X, X.T), 2)
        b = 0
    else:
        aval = np.linalg.svd(np.dot(X.T, X),
                             full_matrices=False, compute_uv=False)
        a, b = aval[(0, -1)]
    
    return (a+b)/(n*2.0)
    
def _reverse_enumerate(iterable):
    from itertools import izip
    return izip(reversed(xrange(len(iterable))), reversed(iterable))
    
def _soft_thresholding(x, th):
    out = x - (np.sign(x) * (th/2.0))
    out[np.abs(x) < (th/2.0)] = 0.0
    return out