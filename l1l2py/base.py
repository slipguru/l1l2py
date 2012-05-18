# Author: Salvatore Masecchia <salvatore.masecchia@disi.unige.it>
# License: New-BSD

import numpy as np
from .data import center

# Fit intercept devo centrare anche la y!
# Altrimenti posso centrare solo la X fuori e NON fittare l'intercetta,
# ma devo ricordarmi di shiftare anche le previsioni rispetto alla media
# della y (appunto)

class AbstractLinearModel(object):
    """ Abstract Linear Model. """
    
    def __init__(self):
        self._beta = None
        self._intercept = None
        
    @property
    def beta(self):
        return self._beta
    
    @property
    def intercept(self):
        return self._intercept
    
    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        Returns
        -------
        pred : array, shape (n_samples,)
            Predicted values
        """
        X = np.asanyarray(X)
        return np.dot(X, self._beta) + self._intercept
        
    def train(self, X, y, fit_intercept=True, *args, **kwargs):
        X = np.asanyarray(X)
        y = np.asanyarray(y)
        
        X, y, Xmean, ymean = _center_data(X, y, fit_intercept)
        self._train(X, y, *args, **kwargs)        
        self._intercept = _set_intercept(Xmean, ymean,
                                         self._beta, fit_intercept)
        return self
        
    def _train(self, X, y, *args, **kwargs):
        raise NotImplementedError

# Useful functions --    
def _center_data(X, y, fit_intercept):
    """
    Centers data to have mean zero along axis 0. This is here because
    nearly all linear models will want their data to be centered.
    """
    if fit_intercept:
        Xmean = X.mean(axis=0)
        X = X - Xmean
        
        ymean = y.mean()
        y = y - ymean
    else:
        Xmean = np.zeros(X.shape[1])
        ymean = 0.
    return X, y, Xmean, ymean

def _set_intercept(Xmean, ymean, beta, fit_intercept):
    """Set the intercept_
    """
    if fit_intercept:
        return ymean - np.dot(Xmean, beta)
    else:
        return 0
    
