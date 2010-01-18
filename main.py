#!/usr/bin/env python

from optparse import OptionParser
import io

def main():
    usage = "usage: %prog configuration-file"
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.error('incorrect number of arguments')
    config_file_path = args[0]

    conf = io.Configuration(config_file_path)
    print conf

    expressions, labels = conf.data_matrices()

    # --------------------------------------------------
    print 'Data shape:', expressions.shape
    print 'Labels shape:', labels.shape
    assert expressions.shape[0] == labels.shape[0]
    assert labels.shape[1] == 1
    # --------------------------------------------------

    # Starting Protocol
    import tools
    tau_values =    tools.parameter_range(conf['parameters_tau-range-type'],
                                          float(conf['parameters_tau-min']),
                                          float(conf['parameters_tau-max']),
                                          int(conf['parameters_tau-number']))
    lambda_values = tools.parameter_range(conf['parameters_lambda-range-type'],
                                          float(conf['parameters_lambda-min']),
                                          float(conf['parameters_lambda-max']),
                                          int(conf['parameters_lambda-number']))
    mu_values =     tools.parameter_range(conf['parameters_mu-range-type'],
                                          float(conf['parameters_mu-min']),
                                          float(conf['parameters_mu-max']),
                                          int(conf['parameters_mu-number']))

    ext_cv_sets = tools.kcv_indexes(labels, int(conf['parameters_external-k']),
                                            conf['experiment_type'])
    train_idxs = ext_cv_sets[int(conf['parameters_split-index'])][0]
    test_idxs = ext_cv_sets[int(conf['parameters_split-index'])][1]

    # --------------------------------------------------
    print 'Numero fold:', len(ext_cv_sets)
    print 'Indici training set (esterno):', train_idxs
    print 'Indici test set (esterno):', test_idxs
    assert len(train_idxs + test_idxs) == labels.size
    # --------------------------------------------------

    Xtrain, Ytrain = expressions[train_idxs,:], labels[train_idxs,:]
    Xtest,  Ytest  = expressions[test_idxs, :], labels[test_idxs, :]

    # STAGE I
    #functions.l1l2_kvc(Xtrain, Ytrain,
    #                   tau_values,      # EN, stage Ia
    #                   mu_values[0],    # EN, stage Ia
    #                   lambda_values,    # RLS, stage Ib
    #                   input['internal_k'],
    #                   input['experiment_type']) #can I normalize externally?
    #
    #
    ##KCV_grid(expressions, labels, lambda_min,lambda_max,all_eps,Kint,Kext,split_idx,resultfile)

if __name__ == '__main__':
    main()
