import numpy as np
import mlpy

def linear_range(min, max, number):
    return np.linspace(min, max, number)

def geometric_range(min, max, number):
    ratio = (max/float(min))**(1.0/(number-1))
    return min * (ratio ** np.arange(number))

def standardize(matrix, p=None):
    """ This function simulate the normalization
        function in the matlab code with norm_mean=norm_col=1"""
    return mlpy.data_standardize(matrix, p)

def center(matrix, p=None):
    mean = matrix.mean(axis=0)
    if p is None:
        return matrix - mean, mean
    else:
        return matrix - mean, p - mean, mean

def scaling_factor(X):
    Xtmp = standardize(X)
    return np.sqrt( np.linalg.norm( np.dot(Xtmp, Xtmp.T), 2) )

def kcv_indexes(labels, k, experiment_type):
    if experiment_type == 'classification':
        return mlpy.kfoldS(labels, k)
    elif experiment_type == 'regression':
        return mlpy.kfold(labels.size, k)
    else:
        raise RuntimeError()