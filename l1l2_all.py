#!/usr/bin/env python

from optparse import OptionParser
from ConfigParser import SafeConfigParser

def main():
    usage = "usage: %prog configuration-file"
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.error('incorrect number of arguments')
    config_file_path = args[0]

    config = SafeConfigParser()
    if not config.read(config_file_path):
        raise RuntimeError('incorrect configuration file')
        
    # To generalize!!
    input = {'experiment_type': config.get('experiment', 'type'),
             'data_path': config.get('data', 'path'),
             'data_name': config.get('data', 'name'),
             
             'labels_path': config.get('labels', 'path'),
             'labels_name': config.get('labels', 'name'),
             'labels_type': config.get('labels', 'type'),
            
             'result_path': config.get('output', 'result'),
            
             'tau_min': config.getfloat('parameters', 'tau-min'),
             'tau_max': config.getfloat('parameters', 'tau-max'),
             'lambda_min': config.getfloat('parameters', 'lambda-min'),
             'lambda_max': config.getfloat('parameters', 'lambda-max'),
             'mu_min': config.getfloat('parameters', 'mu-min'),
             'mu_max': config.getfloat('parameters', 'mu-max'),
             
             'split_idx': config.getint('parameters', 'split-index'),
             'external_k': config.getint('parameters', 'external-k'),
             'internal_k': config.getint('parameters', 'internal-k'),            
            }
      
    print_input(input)
    
    # if is a matlab file
    from scipy.io import loadmat
    from os import path
    
    # Configuration file directory is the root directory
    config_path = path.split(config_file_path)[0]
    
    tmp = loadmat(path.join(config_path, input['data_path']),
                  struct_as_record=True)
    expressions = tmp[input['data_name']]
    
    if input['data_path'] != input['labels_path']:
        tmp = loadmat(input['labels_path'], struct_as_record=True)
    labels = tmp[input['labels_name']]
    
    # --------------------------------------------------
    print 'Data shape:', expressions.shape
    print 'Labels shape:', labels.shape
    assert expressions.shape[0] == labels.shape[0]
    assert labels.shape[1] == 1
    # --------------------------------------------------
    
    # Starting Protocol
    import functions
    import numpy as np
    sf = functions.scaling_factor(expressions)
    tau_values =    functions.get_range(input['tau_min'],
                                        input['tau_max'],
                                        20, sf)
    lambda_values = functions.get_range(input['lambda_min'],
                                        input['lambda_max'],
                                        20, sf)
    mu_values =     functions.get_range(input['mu_min'],
                                        input['mu_max'],
                                        20, sf)
    
    ext_cv_sets = functions.get_split_indexes(labels, input['external_k'],
                                                      input['experiment_type'])
    train_idxs = ext_cv_sets[input['split_idx']][0]
    test_idxs = ext_cv_sets[input['split_idx']][1]
    
    print len(ext_cv_sets)
    print train_idxs
    print test_idxs
    print len(train_idxs + test_idxs)
    Xtrain, Ytrain = expressions[train_idxs,:], labels[train_idxs,:]
    Xtest,  Ytest  = expressions[test_idxs, :], labels[test_idxs, :]
    
    # STAGE I
    functions.l1l2_kvc(Xtrain, Ytrain,
                       tau_values,      # EN, stage Ia
                       mu_values[0],    # EN, stage Ia
                       lambda_values,    # RLS, stage Ib
                       input['internal_k'],
                       input['experiment_type']) #can I normalize externally?
    
    
    #KCV_grid(expressions, labels, lambda_min,lambda_max,all_eps,Kint,Kext,split_idx,resultfile)



def print_input(input):
    width = 32
    just = (width/2)-1
    # Header
    print
    print '+',      '-'*width,         '+'
    print '|', 'Input'.center(width),  '|'
    print '+',      '-'*width,         '+'
    
    # Content
    for k in sorted(input.keys()):
        print '| %s: %s |' % (k.rjust(just), str(input[k]).ljust(just))
    
    # Footer
    print '+',      '-'*width,         '+'
    print
    
if __name__ == '__main__':
    main()



        

