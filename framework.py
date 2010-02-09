from __future__ import division
import numpy as np

import algorithms as alg
import tools

# l1l2_kcv!
def select_features(X, Y, mu_fact, tau_range, lambda_range=np.empty(0),
                    int_cv_sets=None, experiment_type='classification',
                    standardize_X=True, center_Y=True):
    """
    Ia: mu_fact, tau_range
    Ib: lambda_range on selected
        ** step II -> out **
        
    Normalization inside!
    Is needed to normalize single sub-subsets
    """
    if experiment_type =='classification':
    #    int_cv_sets = tools.classification_splits(Y, k)
        compute_error = tools.classification_error
    else:
    #    int_cv_sets = tools.regression_splits(Y, k)
        compute_error = tools.regression_error
    
    # --------------------------------------------------
    #print 'Numero fold interni:', len(int_cv_sets)
    #print 'Coppie sets (interno):', int_cv_sets
    # --------------------------------------------------
    
    err_ts = np.empty((len(int_cv_sets), tau_range.size, lambda_range.size))
    err_tr = np.empty_like(err_ts)
    
    for i, (train_idxs, test_idxs) in enumerate(int_cv_sets):
        #print 'Train: ', np.asarray(train_idxs)+1, 'on: ', np.asarray(test_idxs)+1
        
        # This command makes copy of the data!
        if standardize_X:
            Xtr, Xts = tools.standardize(X[train_idxs,:], X[test_idxs,:])
        else:
            Xtr, Xts = X[train_idxs,:], X[test_idxs,:]
            
        if center_Y:
            Ytr, Yts, meanY = tools.center(Y[train_idxs,:], Y[test_idxs,:])
        else:
            Ytr, Yts = Y[train_idxs,:], Y[test_idxs,:]
            
        # REG_PATH mu_0 and tau_range!!
        beta_casc = alg.elastic_net_regpath(Xtr, Ytr, mu_fact, tau_range)
        
        # STAGE Ib!!!
        for j, b in enumerate(beta_casc):
            selected = (b.flat != 0)
            for k, l in enumerate(lambda_range):
                beta = alg.ridge_regression(Xtr[:,selected], Ytr, l)
                
                labelsTs = Yts + meanY
                predictedTs = np.dot(Xts[:,selected], beta) + meanY
                err_ts[i, j, k] = compute_error(labelsTs,
                                                   predictedTs)
                
                labelsTr = Ytr + meanY
                predictedTr = np.dot(Xtr[:,selected], beta) + meanY
                err_tr[i, j, k] = compute_error(labelsTr,
                                                predictedTr)
    #print err_ts
    err_ts = err_ts.mean(axis=0)
    err_tr = err_tr.mean(axis=0)
       
    tau_opt_idx, lambda_opt_idx = np.where(err_ts == err_ts.min())
    tau_opt = tau_range[tau_opt_idx[0]]
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

