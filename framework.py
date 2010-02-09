from __future__ import division
import numpy as np

import algorithms as alg
import tools

# l1l2_kcv!
def select_features(X, Y, mu, tau_range, lambda_range, cv_sets,
                    error_function,
                    data_normalizer=None, labels_normalizer=None):
      
    err_ts = np.empty((len(cv_sets), tau_range.size, lambda_range.size))
    err_tr = np.empty_like(err_ts)
    
    for i, (train_idxs, test_idxs) in enumerate(cv_sets):
        # First create a view and then normalize (eventually)
        Xtr, Xts = X[train_idxs,:], X[test_idxs,:]
        if not data_normalizer is None:
            Xtr, Xts = data_normalizer(Xtr, Xts)
            
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
          
    return tau_opt, lambda_opt

#def build_classifier(Xtr, Ytr, Xts, Yts, tau_opt, lambda_opt, mu_range, experiment_type,
#             standardize_X=True, center_Y=True):
#    
#    # This command makes copy of the data!
#    if standardize_X:
#        Xtr, Xts = tools.standardize(Xtr, Xts)
#    if center_Y:
#        Ytr, Yts, meanY = tools.center(Ytr, Yts)
#    
#    # IIa
#    beta_0 = ridge_regression(Xtr, Ytr)
#    beta, k = elastic_net(Xtr, Ytr, mu_range[0], tau_opt, beta_0)
#    selected = (beta.flat != 0)
#    
#    # IIb
#    beta_opt = ridge_regression(Xtr[:,selected], Ytr, lambda_opt)
#    
#    labels = Yts + meanY
#    predicted = np.dot(Xts[:,selected], beta_opt) + meanY
#    err_test = prediction_error(labels, predicted, experiment_type)
#    #sums sums sums
#    
#    for m in mu_range[1:]:
#        pass
#    
#    mu_opt = None
#    return mu_opt

