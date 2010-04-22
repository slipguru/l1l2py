import numpy as np

from algorithms import _maximum_eigenvalue, ridge_regression

def correlated_dataset(samples, groups, variables, true_model):
    """
    >>> correlated_dataset(30, (5, 5, 5), 40, [3.0]*15 + [0.0]*(40-15))
    ...
    """
    #print len(true_model), variables
    assert sum(groups) <= variables
    assert len(true_model) == variables

    X = np.zeros((samples, variables))
    i = 0
    for g in groups:
        variable = np.random.normal(scale=1.0, size=(samples,))
        for j in xrange(g):
            error = np.random.normal(scale=0.01, size=(samples,))
            X[:,i] = variable + error
            i+=1

    noisy = variables - sum(groups)
    X[:, i:] = np.random.normal(scale=1.0, size=(samples, noisy))

    error = np.random.normal(scale=1.0, size=(samples,))
    Y = np.dot(X, true_model) + error

    return X, Y


