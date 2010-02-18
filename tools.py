import numpy as np

__all__ = ['linear_range', 'geometric_range',
           'standardize', 'center',
           'classification_error', 'balanced_classification_error',
                                   'regression_error',
            'kfold_splits', 'stratified_kfold_splits']

# Ranges functions ------------------------------------------------------------    
def linear_range(min, max, number):
    return np.linspace(min, max, number)

def geometric_range(min, max, number):
    ratio = (max/float(min))**(1.0/(number-1))
    return min * (ratio ** np.arange(number))

# Normalization ---------------------------------------------------------------
def standardize(matrix, p=None, return_factors=False):
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)

    # Simple case
    if p is None and return_factors is False:
        return (matrix - mean)/std
                
    if p is None: # than return_factors is True
        return (matrix - mean)/std, mean, std
        
    if return_factors is False: # ... with p not None
        return (matrix - mean)/std, (p - mean)/std
        
    # Full case
    return (matrix - mean)/std, (p - mean)/std, mean, std
    
def center(matrix, p=None, return_mean=False):
    mean = matrix.mean(axis=0)
    
    # Simple case
    if p is None and return_mean is False:
        return matrix - mean
    
    if p is None: # than return_mean is True
        return (matrix - mean, mean)
    
    if return_mean is False: # ...with p not None
        return (matrix - mean, p - mean)
    
    # Full case
    return (matrix - mean, p - mean, mean)
    
# Error functions -------------------------------------------------------------    
def classification_error(labels, predicted):
    difference = (np.sign(labels) != np.sign(predicted))
    return labels[difference].size / float(labels.size)
    
def balanced_classification_error(labels, predicted):
    balance_factors = np.abs(center(labels)[0])
   
    errors = (np.sign(labels) != np.sign(predicted)) * balance_factors
    return errors.sum() / float(labels.size)
    
def regression_error(labels, predicted):
    norm = np.linalg.norm(labels - predicted, 2)
    return (norm * norm) / float(labels.size)
    
# KCV tools -------------------------------------------------------------------    
def kfold_splits(labels, k, rseed=0):
    import mlpy
    return mlpy.kfold(labels.size, k, rseed)

def stratified_kfold_splits(labels, k, rseed=0):
    import mlpy
    return mlpy.kfoldS(labels, k, rseed)