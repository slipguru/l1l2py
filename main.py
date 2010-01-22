#!/usr/bin/env python

from optparse import OptionParser
import io, framework, tools

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

    # --------------------------------------------------
    print 'Data shape:', expressions.shape
    print 'Labels shape:', labels.shape
    assert expressions.shape[0] == labels.shape[0]
    assert labels.shape[1] == 1
    # --------------------------------------------------
    
    ext_cv_sets = tools.kcv_indexes(labels, conf.external_k, conf.experiment_type)
    train_idxs, test_idxs = ext_cv_sets[conf.split_index]
    
    # --------------------------------------------------
    print 'Numero fold:', len(ext_cv_sets)
    print 'Indici training set (esterno):', train_idxs
    print 'Indici test set (esterno):', test_idxs
    assert len(train_idxs + test_idxs) == labels.size
    # --------------------------------------------------

    Xtrain, Ytrain = expressions[train_idxs,:], labels[train_idxs,:]
    Xtest,  Ytest  = expressions[test_idxs, :], labels[test_idxs, :]
    assert Xtrain.shape == (20, 40)
    assert Ytrain.shape == (20, 1)
    assert Xtest.shape == (10, 40)
    assert Ytest.shape == (10, 1)
         
    tau_opt, lambda_opt = framework.stage_I(Xtrain, Ytrain,
                                             conf.mu_range[0],
                                             conf.tau_range,
                                             conf.lambda_range,
                                             conf.internal_k,
                                             conf.experiment_type)
    
    mu_opt = framework.stage_II(Xtrain, Ytrain, Xtest, Ytest,
                                 tau_opt,
                                 lambda_opt,
                                 conf.mu_range,
                                 conf.experiment_type)
    

if __name__ == '__main__':
    main()
