import numpy as np

def linear_range(min, max, number):
    return np.linspace(min, max, number)

def geometric_range(min, max, number):
    ratio = (max/float(min))**(1.0/(number-1))
    return min * (ratio ** np.arange(number))

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