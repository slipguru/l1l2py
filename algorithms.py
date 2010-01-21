from __future__ import division
import tools
import numpy as np


def stage_I(X, Y, mu_fact, tau_range, lambda_range=np.empty(0),
            k=0, experiment_type='classification',
            standardize_X=True, center_Y=True):
    """
    Ia: mu_fact, tau_range
    Ib: lambda_range on selected
        ** step II -> out **
        
    Normalization inside!
    Is needed to normalize the single sub-subset
    """
    int_cv_sets = tools.kcv_indexes(Y, k, experiment_type)
    
    # --------------------------------------------------
    print 'Numero fold interni:', len(int_cv_sets)
    print 'Coppie sets (interno):', int_cv_sets
    # --------------------------------------------------
    
    for train_idxs, test_idxs in int_cv_sets:
        print 'Train: ', train_idxs, 'on: ', test_idxs
        
        # This command makes copy of the data!
        if standardize_X:
            Xtr, Xts = tools.standardize(X[train_idxs,:], X[test_idxs,:])
        else:
            Xtr, Xts = X[train_idxs,:], X[test_idxs,:]
            
        if center_Y:
            Ytr, Yts = tools.center(Y[train_idxs,:], Y[test_idxs,:])
        else:
            Ytr, Yts = Y[train_idxs,:], Y[test_idxs,:]
            
        # REG_PATH mu_0 and tau_range!!
        beta_casc = stage_Ia(Xtr, Ytr, mu_fact, tau_range)

    #Xtrain, Ytrain = expressions[train_idxs,:], labels[train_idxs,:]
    #Xtest,  Ytest  = expressions[test_idxs, :], labels[test_idxs, :]
       
    return None

def stage_Ia(X, Y, mu, tau_range, kmax=np.inf):
    """ reg_path """
    n, d = X.shape
    
    beta_ls = ols(X, Y) # np.dot(np.dot(X.T, X).I, np.dot(X.T, Y))
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
        #...
        sparsity = np.sum(beta_next != 0)
        beta = beta_next
    
    return np.asarray(out) #very inefficient! right?!

def elastic_net(X, Y, mu, tau, beta, kmax=np.inf):
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
    
    return beta, k
    
    
def soft_thresholding(x, th):
    out = x - (np.sign(x) * (th/2.0))
    out[np.abs(x) < (th/2.0)] = 0.0
    return out

def ols(X, Y):
    n, d = X.shape
    tmp = np.linalg.pinv(np.dot(X, X.T))
    
    if n < d:
        return np.dot(np.dot(X.T, tmp), Y)
    else:
        return np.dot(tmp, np.dot(X.T, Y))

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
    