# Author: Salvatore Masecchia <salvatore.masecchia@disi.unige.it>
# License: New-BSD

import numpy as np
from numpy import linalg as la

from .data import center

class AbstractLinearModel(object):
    """Abstract Linear Model. """
    
    def __init__(self):
        self._beta = None
        self._intercept = None
        self._trained = False
        
    @property
    def beta(self):
        return self._beta
    
    @property
    def intercept(self):
        return self._intercept
    
    @property
    def trained(self):
        return self._trained
    
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
        if not self._trained:
            raise RuntimeError('model not trained')
        
        X = np.asanyarray(X)
        return np.dot(X, self._beta) + self._intercept
        
    def train(self, X, y, fit_intercept=True, *args, **kwargs):
        X = np.asanyarray(X)
        y = np.asanyarray(y)
        
        # Centering Data
        if fit_intercept:
            X, Xmean = center(X, return_mean=True)
            y, ymean = center(y, return_mean=True)
        
        # Calling the class-specific train method
        self._train(X, y, *args, **kwargs)
        
        # Fitting the intercept if required
        if fit_intercept:
            self._intercept = ymean - np.dot(Xmean, self._beta)
        else:
            self._intercept = 0.0
        
        self._trained = True
        return self
        
    def _train(self, X, y, *args, **kwargs):
        """ Has to set self._beta at least """
        raise NotImplementedError

    
class RidgeRegression(AbstractLinearModel):
    """Ridge regression solved as direct method. """

    def __init__(self, mu=0.0):
        super(RidgeRegression, self).__init__()
        self._mu = mu
        
    @property
    def mu(self):
        return self._mu

    def _train(self, X, y):
        n, d = X.shape

        if n < d:
            tmp = np.dot(X, X.T)
            if self._mu != 0.0:
                tmp += self._mu * n * np.eye(n)
            tmp = la.pinv(tmp)

            self._beta =  np.dot(np.dot(X.T, tmp), y)
        else:
            tmp = np.dot(X.T, X)
            if self._mu != 0.0:
                tmp += self._mu * n * np.eye(d)
            tmp = la.pinv(tmp)

            self._beta = np.dot(tmp, np.dot(X.T, y))