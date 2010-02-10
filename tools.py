import numpy as np
import mlpy

def linear_range(min, max, number):
    return np.linspace(min, max, number)

def geometric_range(min, max, number):
    ratio = (max/float(min))**(1.0/(number-1))
    return min * (ratio ** np.arange(number))
    
def reverse_enumerate(iterable):
    from itertools import izip
    return izip(reversed(xrange(iterable.size)), reversed(iterable))

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

def kfold_splits(labels, k, rseed=0):
    return mlpy.kfold(labels.size, k, rseed)

def stratified_kfold_splits(labels, k, rseed=0):
    return mlpy.kfoldS(labels, k, rseed)
    
def classification_error(labels, predicted):
    difference = (np.sign(labels) != np.sign(predicted))
    return labels[difference].size / float(labels.size)
    
def regression_error(labels, predicted):
    norm = np.linalg.norm(labels - predicted, 2)
    return (norm * norm) / float(labels.size)
