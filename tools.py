""" TODO: Add docstring """

import numpy as np

__all__ = ['linear_range', 'geometric_range',
           'standardize', 'center',
           'classification_error', 'balanced_classification_error',
                                   'regression_error',
            'kfold_splits', 'stratified_kfold_splits']

# Ranges functions ------------------------------------------------------------
def linear_range(min_value, max_value, number):
    """ TODO: Add docstring """
    return np.linspace(min_value, max_value, number)

def geometric_range(min_value, max_value, number):
    """ TODO: Add docstring """
    ratio = (max_value/float(min_value))**(1.0/(number-1))
    return min_value * (ratio ** np.arange(number))

# Normalization ---------------------------------------------------------------
def standardize(matrix, optional_matrix=None, return_factors=False):
    """ TODO: Add docstring """
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)

    # Simple case
    if optional_matrix is None and return_factors is False:
        return (matrix - mean)/std
                
    if optional_matrix is None: # than return_factors is True
        return (matrix - mean)/std, mean, std
        
    if return_factors is False: # ... with p not None
        return (matrix - mean)/std, (optional_matrix - mean)/std
        
    # Full case
    return (matrix - mean)/std, (optional_matrix - mean)/std, mean, std
    
def center(matrix, optional_matrix=None, return_mean=False):
    """ TODO: Add docstring """
    mean = matrix.mean(axis=0)
    
    # Simple case
    if optional_matrix is None and return_mean is False:
        return matrix - mean
    
    if optional_matrix is None: # than return_mean is True
        return (matrix - mean, mean)
    
    if return_mean is False: # ...with p not None
        return (matrix - mean, optional_matrix - mean)
    
    # Full case
    return (matrix - mean, optional_matrix - mean, mean)
    
# Error functions -------------------------------------------------------------
def classification_error(labels, predicted):
    """ TODO: Add docstring """
    difference = (np.sign(labels) != np.sign(predicted))
    return labels[difference].size / float(labels.size)
    
def balanced_classification_error(labels, predicted):
    """ TODO: Add docstring """
    balance_factors = np.abs(center(labels)[0])
   
    errors = (np.sign(labels) != np.sign(predicted)) * balance_factors
    return errors.sum() / float(labels.size)
    
def regression_error(labels, predicted):
    """ TODO: Add docstring """
    norm = np.linalg.norm(labels - predicted, 2)
    return (norm * norm) / float(labels.size)
    
# KCV tools -------------------------------------------------------------------
def kfold_splits(labels, k, rseed=0):
    """ TODO: Add docstring """
    import mlpy
    return mlpy.kfold(labels.size, k, rseed)

def stratified_kfold_splits(labels, k, rseed=0):
    """ TODO: Add docstring """
    import mlpy
    return mlpy.kfoldS(labels, k, rseed)