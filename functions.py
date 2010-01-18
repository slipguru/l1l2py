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

def get_geometric_range(min, max, number):
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

def get_split_indexes(labels, k, experiment_type):
    if experiment_type == 'classification':
        return mlpy.kfoldS(labels, k)
    elif experiment_type == 'regression':
        return mlpy.kfold(labels.size, k)
    else:
        raise Exception()

def l1l2_kvc(*args):
    return args