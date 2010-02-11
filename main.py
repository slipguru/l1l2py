#!/usr/bin/env python

from optparse import OptionParser
import io, tools
import framework as fw

import numpy as np
#import pylab as pl
#from mpl_toolkits.mplot3d import Axes3D


def create_jdf_model_selection():
    f = open('listajob_gse2990_MF_ER-0p1.jdf','w') #nome nel config!
    f.write('job :\n')
    f.write('label : "Analisi gse2990_MF_ER "\n')
    
    locations = ['unige']
    os = ['linux']
    
    envs = '||'.join('( environment == %s )' % x for x in locations)
    os = '||'.join('( os == %s )' % x for x in os)
    f.write('requirements :  %s' %  '&&'.join((lenv, os)))

    #tgz_file = 'gse2990_MF_ER.tgz';
    #[s, r]= system('tar cvzf gse2990_MF_ER.tgz MATLAB MATLAB_FUN info_range.csv lunghezzeMF.txt LABELS');

    ext_cv_sets = tools.kfold_splits(labels, conf.external_k)
    
    print
    print 'Splitting data in %d pairs' % len(ext_cv_sets)
    
    err_ts = np.empty(len(ext_cv_sets))
    beta_list = list()
    selected_list = list()
    error_list = list()
    for i, (train_idxs, test_idxs) in enumerate(ext_cv_sets):
        f.write('task: init: put %s %s\n', config.expression_file_path, config.expression_file);
        f.write('            put  %s %s\n',tgz_file,tgz_file);
        f.write('            put run.sh run\n');
        f.write('      remote: run %s l1l2_all.m %s %f %d %s %s %f %f %d %d > output.log 2>&1\n',tgz_file,task[t], eps(i),s,['../' resultfile_name], ['../' files(f).name],tau_min,tau_max, Kint, Kext
        f.write('      final: get output.log %s/$PROC-$JOB-$TASK.log\n',outdir);
        f.write('             get %s %s/%s\n',resultfile_name ,outdir,[resultfile_name '.mat']);
        
def create_jdf_features_family():
    pass

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
    error_list = list()
    for i, (train_idxs, test_idxs) in enumerate(ext_cv_sets):
        print 'Learning on the pair #%d...' % (i+1)
        
        Xtr, Ytr = expressions[train_idxs,:], labels[train_idxs,:]
        Xts,  Yts  = expressions[test_idxs, :], labels[test_idxs, :]
             
        tau_opt, lambda_opt, err_tr, err_ts = fw.select_features(
                                Xtr, Ytr,
                                conf.mu_range[0],
                                conf.tau_range,
                                conf.lambda_range,
                                tools.kfold_splits(Ytr, conf.internal_k),
                                error_function,
                                data_normalizer=tools.standardize,
                                labels_normalizer=tools.center)
        # salvare conca
        
        Xtr, Xts, meanX, stdX = tools.standardize(Xtr, Xts)       
        Ytr, Yts, meanY = tools.center(Ytr, Yts)
            
        beta_opt_list, selected_opt_list, err_ts = fw.build_classifier(
                                Xtr, Ytr, Xts, Yts,
                                tau_opt, lambda_opt,
                                conf.mu_range,
                                error_function)
        # salvare errori
        
        print '...optimal parameters:', tau_opt, lambda_opt
        
        beta_list.append(beta_opt_list)
        selected_list.append(selected_opt_list)
        error_list.append(err_ts)
    
    foo = np.asarray(selected_list)
    print foo.shape
    freq = foo.sum(axis=0)
    print freq.shape
    min = freq.min(axis=0)
    max = freq.max(axis=0)
    print min
    print max
    
    print np.asarray(error_list)
    print np.asarray(error_list).mean(axis=0)
    
    
    # Save: freq ed err_ts
    
    
    #light
    
    
    
       
    

if __name__ == '__main__':
    main()
