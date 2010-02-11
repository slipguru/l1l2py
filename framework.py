from __future__ import division
import numpy as np

import algorithms as alg
import tools

def select_features(X, Y, mu, tau_range, lambda_range, cv_sets,
                    error_function,
                    data_normalizer=None, labels_normalizer=None):
      
    err_ts = np.empty((len(cv_sets), tau_range.size, lambda_range.size))
    err_tr = np.empty_like(err_ts)
    
    for i, (train_idxs, test_idxs) in enumerate(cv_sets):
        # First create a view and then normalize (eventually)
        Xtr, Xts = X[train_idxs,:], X[test_idxs,:]
        if not data_normalizer is None:
            Xtr, Xts, meanX, stdX = data_normalizer(Xtr, Xts)
            
        Ytr, Yts = Y[train_idxs,:], Y[test_idxs,:]
        if not labels_normalizer is None:
            Ytr, Yts, meanY = labels_normalizer(Ytr, Yts)
            
        # Builds a classifier for each value of tau
        beta_casc = alg.elastic_net_regpath(Xtr, Ytr, mu, tau_range)
        
        # For each sparse model builds a rls classifier
        # for each value of lambda
        for j, b in enumerate(beta_casc):
            selected = (b != 0)
            for k, lam in enumerate(lambda_range):
                beta = alg.ridge_regression(Xtr[:,selected], Ytr, lam)
                
                prediction = np.dot(Xts[:,selected], beta)
                err_ts[i, j, k] = error_function(Yts, prediction)
                
                prediction = np.dot(Xtr[:,selected], beta)
                err_tr[i, j, k] = error_function(Ytr, prediction)
    
    err_ts = err_ts.mean(axis=0)
    err_tr = err_tr.mean(axis=0)
       
    tau_opt_idx, lambda_opt_idx = np.where(err_ts == err_ts.min())
    tau_opt = tau_range[tau_opt_idx[0]]             # ?? [0] or [-1]
    lambda_opt = lambda_range[lambda_opt_idx[0]]
          
    return tau_opt, lambda_opt, err_ts, err_tr

def build_classifier(Xtr, Ytr, Xts, Yts, tau_opt, lambda_opt,
                     mu_range, error_function):
    
    err_ts = np.empty(mu_range.size)
    #err_tr = np.empty_like(err_ts)
    
    beta_opt = list()
    selected_opt = list()
    for i, m in enumerate(mu_range):
        beta, k = alg.elastic_net(Xtr, Ytr, m, tau_opt)
        selected = (beta.flat != 0)
        beta_opt.append(alg.ridge_regression(Xtr[:,selected], Ytr, lambda_opt))
        selected_opt.append(selected)
        
        prediction = np.dot(Xts[:,selected], beta_opt[-1])
        err_ts[i] = error_function(Yts, prediction)
        
        #prediction = np.dot(Xtr[:,selected], beta_opt[-1])
        #err_tr[i] = error_function(Ytr, prediction)
  
    #print '*'*20
    #print mu_range
    #print err_ts
  
    #mu_opt_idx, = np.where(err_ts == err_ts.min())
    #mu_opt = mu_range[mu_opt_idx[0]]
        
    #return beta_opt
    return beta_opt, selected_opt, err_ts

