import numpy as np

def linear_range(min, max, number):
    return np.linspace(min, max, number)

def geometric_range(min, max, number):
    ratio = (max/float(min))**(1.0/(number-1))
    return min * (ratio ** np.arange(number))

def standardize(matrix, p=None, return_factors=False):
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
        
    if p is None:
        out = ((matrix - mean)/std,)
    else:
        out = ((matrix - mean)/std, (p - mean)/std)
        
    if return_factors:
        return out + (mean, std)
    else:
        return out

def center(matrix, p=None, return_mean=False):
    mean = matrix.mean(axis=0)
    
    if p is None:
        out = (matrix - mean,)
    else:
        out = (matrix - mean, p - mean)
        
    if return_mean:
        return out + (mean,)
    else:
        return out