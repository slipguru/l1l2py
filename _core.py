import numpy as np
from biolearning.algorithms import *

__all__ = ['double_optimization', 'model_selection']
   
def double_optimization(Xtr, Ytr, tau_opt, lambda_opt, m, error_function=None):
    beta, k = alg.elastic_net(Xtr, Ytr, m, tau_opt)
    selected = (beta.flat != 0)
    
    beta = alg.ridge_regression(Xtr[:,selected], Ytr, lambda_opt)
           
    if error_function:    
        prediction = np.dot(Xtr[:,selected], beta)
        err_tr[i] = error_function(Ytr, prediction)
        return beta, selected, err_tr
    else:
        return beta, selected

def model_selection(*args, **kwargs):
    return stage_I(*args, **kwargs)

def stage_I(data, labels, mu, tau_range, lambda_range, cv_sets,
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
    
# Work in progress!! ----------------------------------------------------------
def models_family(Xtr, Ytr, tau_opt, lambda_opt,
                  mu_range, error_function=None):
    
    beta_opt = list()
    selected_opt = list()
    if error_function:
        err_tr = list()
    
    for m in enumerate(mu_range):
        beta, selected, err = double_optimization(Xtr, Ytr, tau_opt, lambda_opt, m,
                                                   error_function)        
        beta_opt.append(beta)
        selected_opt.append(selected)
        if error_function:
            err_tr.append(err) 
              
    if error_function:
        return beta_opt, selected_opt, err_tr
    else:
        return beta_opt, selected_opt

