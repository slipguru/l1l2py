#!/usr/bin/env python

from optparse import OptionParser
import io, algorithms

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

    # Starting Protocol
    import tools
    tau_range =  conf.tau_range
    lambda_range = conf.lambda_range
    mu_range = conf.mu_range

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
      
    tau_opt, lambda_opt = algorithms.stage_I(Xtrain, Ytrain, mu_range[0],
                                             tau_range, lambda_range,
                                             conf.internal_k,
                                             conf.experiment_type)
    mu_opt = algorithms.stage_II(Xtrain, Ytrain, Xtest, Ytest,
                                 tau_opt, lambda_opt, mu_range,
                                 conf.experiment_type)
    

    # STAGE I
    #functions.l1l2_kvc(Xtrain, Ytrain,
    #                   tau_values,      # EN, stage Ia
    #                   mu_values[0],    # EN, stage Ia
    #                   lambda_values,    # RLS, stage Ib
    #                   input['internal_k'],
    #                   input['experiment_type']) #can I normalize externally?
    #
    #

if __name__ == '__main__':
    main()
