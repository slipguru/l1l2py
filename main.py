#!/usr/bin/env python

from optparse import OptionParser
import io, tools
import framework as fw

import numpy as np

def main():
    usage = "usage: %prog configuration-file"
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.error('incorrect number of arguments')
    config_file_path = args[0]

    conf = io.Configuration(config_file_path)
    print conf

    expressions, labels = conf.expressions, conf.labels
    if conf.experiment_type == 'regression':
        error_function = tools.regression_error
    else:
        error_function = tools.classification_error

    # --------------------------------------------------
    print 'Data shape:', expressions.shape
    print 'Labels shape:', labels.shape
    assert expressions.shape[0] == labels.shape[0]
    assert labels.shape[1] == 1
    # --------------------------------------------------
    
    ext_cv_sets = tools.kfold_splits(labels, conf.external_k)
    
    print
    print 'Splitting data in %d pairs' % len(ext_cv_sets)
    
    err_ts = np.empty(len(ext_cv_sets))
    beta_list = list()
    selected_list = list()
    for i, (train_idxs, test_idxs) in enumerate(ext_cv_sets):
        print 'Learning on the pair #%d...' % (i+1)
        
        Xtr, Ytr = expressions[train_idxs,:], labels[train_idxs,:]
        Xts,  Yts  = expressions[test_idxs, :], labels[test_idxs, :]
             
        tau_opt, lambda_opt = fw.select_features(
                                Xtr, Ytr,
                                conf.mu_range[0],
                                conf.tau_range,
                                conf.lambda_range,
                                tools.kfold_splits(Ytr, conf.internal_k),
                                error_function,
                                data_normalizer=tools.standardize,
                                labels_normalizer=tools.center)
            
        beta_opt, selected, mu_opt = fw.build_classifier(
                                Xtr, Ytr, Xts, Yts,
                                tau_opt, lambda_opt,
                                conf.mu_range,
                                error_function,
                                data_normalizer=tools.standardize,
                                labels_normalizer=tools.center)
        
        print '...optimal parameters:', tau_opt, mu_opt, lambda_opt

        beta_list.append(beta_opt)
        selected_list.append(selected)
        
        beta = np.zeros((selected.size, 1))
        beta[selected] = beta_opt
        prediction = np.dot(Xts, beta)
        err_ts[i] = error_function(Yts, prediction)
    
    opt_idx, = np.where(err_ts == err_ts.min()) # errore medio!
    beta_opt = beta_list[opt_idx[0]]
    sel_opt = selected_list[opt_idx[0]]
    
    # frequency - recombine.
    
    # Add zeros
    beta = np.zeros((sel_opt.size, 1))
    beta[sel_opt] = beta_opt
    
    print
    print 'Best model'
    print beta.T
    

if __name__ == '__main__':
    main()
