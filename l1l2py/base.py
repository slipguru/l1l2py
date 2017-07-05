# Author: Salvatore Masecchia <salvatore.masecchia@disi.unige.it>
# License: New-BSD

import numpy as np
from numpy import linalg as la

from sklearn.linear_model.base import LinearModel

from .data import center


class AbstractLinearModel(LinearModel):
    """Abstract Linear Model. """

    def fit(self, X, y, *args, **kwargs):
        X = np.asanyarray(X)
        y = np.asanyarray(y)

        # Centering Data
        if self.fit_intercept:
            X, Xmean = center(X, return_mean=True)
            y, ymean = center(y, return_mean=True)

        # Calling the class-specific train method
        self._fit(X, y, *args, **kwargs)

        # Fitting the intercept if required
        if self.fit_intercept:
            self._intercept = ymean - np.dot(Xmean, self.coef_)
        else:
            self._intercept = 0.0

        self._trained = True
        return self


class RidgeRegression(AbstractLinearModel):
    """Ridge regression solved as direct method. """

    def __init__(self, mu=0.0, fit_intercept=True):
        self.mu = mu
        self.fit_intercept = fit_intercept

    def _fit(self, X, y):
        # Calling the class-specific train method
        n, d = X.shape

        if n < d:
            tmp = np.dot(X, X.T)
            if self.mu != 0.0:
                tmp += self.mu * n * np.eye(n)
            tmp = la.pinv(tmp)

            coef_ = np.dot(np.dot(X.T, tmp), y)
        else:
            tmp = np.dot(X.T, X)
            if self.mu != 0.0:
                tmp += self.mu * n * np.eye(d)
            tmp = la.pinv(tmp)

            coef_ = np.dot(tmp, np.dot(X.T, y))
        self.coef_ = coef_
