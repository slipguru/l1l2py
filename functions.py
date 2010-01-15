import numpy as np
import mlpy

def normalize(X, Y):
    """ This function simulate the normalization
        function in the matlab code with norm_mean=norm_col=1"""
    Xnorm = mlpy.data_standardize(X)
    Ynorm = Y - Y.mean(axis=0)
    
    return Xnorm, Ynorm

def scaling_factor(X):
    Xtmp = normalize(X, np.empty(0))[0]
    return np.sqrt( np.linalg.norm( np.dot(Xtmp, Xtmp.T), 2) )
    
def get_range(min, max, number, scaling_factor):
    return np.linspace(min, max, number) * scaling_factor
    
def get_split_indexes(labels, k, experiment_type):    
    if experiment_type == 'classification':
        return mlpy.kfoldS(labels, k)
    elif experiment_type == 'regression':
        return mlpy.kfold(labels.size, k)
    else:
        raise Exception()
        
def l1l2_kvc(*args):
    return args