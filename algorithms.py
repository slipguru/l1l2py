from __future__ import division
import tools
import numpy as np
  
def soft_thresholding(x, th):
    out = x - (np.sign(x) * (th/2.0))
    out[np.abs(x) < (th/2.0)] = 0.0
    return out

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

def elastic_net(X, Y, mu, tau, beta=None, kmax=np.inf):
    n, d = X.shape
    
    sigma_0 = _get_sigma(X)
    mu = mu*sigma_0
    sigma = sigma_0 + mu
    mu_s = mu / sigma
    tau_s = tau/sigma
    XT = X.T / (n*sigma)
    
    kmin = 100
    k = 0
    tol = 0.01
    
    if beta is None:
        beta = ridge_regression(X, Y)
        
    value = beta * (1 - mu_s) + np.dot(XT, (Y - np.dot(X, beta)))
    beta_next = soft_thresholding(value, tau_s)
    log = True
    while k < kmin or (k < kmax and log is True):
    
        th = np.abs(beta) * (tol / (k+1))
        if np.all(np.abs(beta_next - beta) <= th): log = False
        
        beta = beta_next;
        value = beta * (1 - mu_s) + np.dot(XT, (Y - np.dot(X, beta)))
        beta_next = soft_thresholding(value, tau_s)
        k = k+1
    
    return beta_next, k

def elastic_net_regpath(X, Y, mu, tau_range, kmax=np.inf):
    """ reg_path """
    n, d = X.shape
    
    beta_ls = ridge_regression(X, Y) # np.dot(np.dot(X.T, X).I, np.dot(X.T, Y))
    beta = beta_ls # np.dot(np.dot(X.T, X).I, np.dot(X.T, Y))
    import collections
    out = collections.deque()
    sparsity = 0
    for i, t in zip(reversed(xrange(10)), tau_range[::-1]):
        
        if mu == 0.0 and sparsity >= n:
            beta_next = beta_ls                
        else:
            beta_next, k = elastic_net(X, Y, mu, t, beta, kmax)
        out.appendleft(beta_next)
        sparsity = np.sum(beta_next != 0)
        beta = beta_next
    
    return np.asarray(out) #very inefficient! right?!

def _get_sigma(X):
    n, d = X.shape
    
    if d > n:
        a = np.linalg.norm(np.dot(X, X.T), 2)
        b = 0
    else:
        aval = np.linalg.svd(np.dot(X.T, X),
                             full_matrices=False, compute_uv=False)
        a = aval[0]
        b = aval[-1]
    
    return (a+b)/(n*2.0)