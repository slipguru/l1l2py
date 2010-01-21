import numpy as np
import mlpy

def standardize(X, p=None):
    """ This function simulate the normalization
        function in the matlab code with norm_mean=norm_col=1"""
    return mlpy.data_standardize(X, p)

def center(X, p=None):
    mean = X.mean(axis=0)
    if p is None:
        return X - mean, mean
    else:
        return X - mean, p - mean, mean

def scaling_factor(X):
    Xtmp = standardize(X)
    return np.sqrt( np.linalg.norm( np.dot(Xtmp, Xtmp.T), 2) )


def parameter_range(type, *args, **kwargs):
    if type == 'linear':
        return linear_range(*args, **kwargs)
    elif type == 'geometric':
        return geometric_range(*args, **kwargs)
    else:
        raise RuntimeError()

def linear_range(min, max, number):
    return np.linspace(min, max, number)

def geometric_range(min, max, number):
    """
    Examples
    --------
    >>> get_geometric_range(2, 10, 20)
    array([  2.        ,   2.1767968 ,   2.36922216,   2.57865761,
             2.80660682,   3.05470637,   3.32473753,   3.61863901,
             3.93852091,   4.28667986,   4.6656155 ,   5.07804845,
             5.52693981,   6.01551245,   6.54727413,   7.1260427 ,
             7.75597347,   8.44158913,   9.1878121 ,  10.        ])
    >>>
    """
    ratio = (max/float(min))**(1.0/(number-1))
    return min * (ratio ** np.arange(20))

def kcv_indexes(labels, k, experiment_type):
    if experiment_type == 'classification':
        return mlpy.kfoldS(labels, k)
    elif experiment_type == 'regression':
        return mlpy.kfold(labels.size, k)
    else:
        raise RuntimeError()