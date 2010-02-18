import numpy as np

__all__ = ['kcv_model_selection', 'ridge_regression',
           'elastic_net', 'elastic_net_regpath']

def kcv_model_selection(data, labels,
                        mu, tau_range, lambda_range, cv_sets,
                        error_function, return_mean_errors=False,
                        data_normalizer=None, labels_normalizer=None):
      
    err_ts = list()
    err_tr = list()
    max_tau_num = len(tau_range)
    
    for i, (train_idxs, test_idxs) in enumerate(cv_sets):        
        # First create a view and then normalize (eventually)
        data_tr, data_ts = data[train_idxs,:], data[test_idxs,:]
        if not data_normalizer is None:
            data_tr, data_ts = data_normalizer(data_tr, data_ts)
            
        labels_tr, labels_ts = labels[train_idxs,:], labels[test_idxs,:]
        if not labels_normalizer is None:
            labels_tr, labels_ts = labels_normalizer(labels_tr, labels_ts)
            
        # Builds a classifier for each value of tau
        beta_casc = elastic_net_regpath(data_tr, labels_tr, mu,
                                        tau_range[:max_tau_num])
        
        if len(beta_casc) < max_tau_num: max_tau_num = len(beta_casc)
        _err_ts = np.empty((len(beta_casc), len(lambda_range)))
        _err_tr = np.empty_like(_err_ts)
        
        # For each sparse model builds a
        # rls classifier for each value of lambda
        for j, b in enumerate(beta_casc):
            selected = (b.flat != 0)
            for k, lam in enumerate(lambda_range):
                beta = ridge_regression(data_tr[:,selected], labels_tr, lam)
                
                prediction = np.dot(data_ts[:,selected], beta)
                _err_ts[j, k] = error_function(labels_ts, prediction)
    
                prediction = np.dot(data_tr[:,selected], beta)
                _err_tr[j, k] = error_function(labels_tr, prediction)
        
        err_ts.append(_err_ts)
        err_tr.append(_err_tr)
    
    # cut columns and computes the mean
    err_ts = np.asarray([a[:max_tau_num] for a in err_ts]).mean(axis=0)
    err_tr = np.asarray([a[:max_tau_num] for a in err_tr]).mean(axis=0)
       
    tau_opt_idx, lambda_opt_idx = np.where(err_ts == err_ts.min())
    tau_opt = tau_range[tau_opt_idx[0]]             # ?? [0] or [-1]
    lambda_opt = lambda_range[lambda_opt_idx[0]]
    
    if return_mean_errors:
        return tau_opt, lambda_opt, err_ts, err_tr
    else:
        return tau_opt, lambda_opt

def ridge_regression(data, labels, penalty=0.0):
    n, d = data.shape
        
    if n < d:
        tmp = np.dot(data, data.T)
        if penalty: tmp += penalty*n*np.eye(n)
        tmp = np.linalg.pinv(tmp)
        
        return np.dot(np.dot(data.T, tmp), labels)
    else:
        tmp = np.dot(data.T, data)
        if penalty: tmp += penalty*n*np.eye(d)
        tmp = np.linalg.pinv(tmp)
        
        return np.dot(tmp, np.dot(data.T, labels))

def elastic_net(data, labels, mu, tau, beta=None, kmax=1e5):
    n, d = data.shape
    
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
        
        value = beta * (1 - mu_s) + np.dot(dataT, (labels - np.dot(data, beta))) 
        beta_next = _soft_thresholding(value, tau_s)
        
        difference = np.abs(beta_next - beta)
        th = np.abs(beta) * (tol / k)
        beta = beta_next
    #--------------------------------------------------------------------------
    return beta_next, k
  
def elastic_net_regpath(data, labels, mu, tau_range, beta=None, kmax=np.inf):
    """ reg_path
    Is sufficient document the possibility to get
    a shorter list of beta without using annoying warnings
    """
    from collections import deque
    n, d = data.shape
    
    beta_ls = ridge_regression(data, labels)
    if beta is None:
        beta = beta_ls
    
    out = deque()
    nonzero = 0
    for t in reversed(tau_range):
        if mu == 0.0 and nonzero >= n: # lasso + saturation
            beta_next = beta_ls                
        else:
            beta_next, k = elastic_net(data, labels, mu, t, beta, kmax)
        
        nonzero = np.sum(beta_next != 0)
        if nonzero > 0:
            out.appendleft(beta_next)
            
        beta = beta_next
                   
    return out

def _get_sigma(matrix):
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
    out = x - (np.sign(x) * (th/2.0))
    out[np.abs(x) < (th/2.0)] = 0.0
    return out