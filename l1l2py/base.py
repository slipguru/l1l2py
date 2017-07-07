# Author: Salvatore Masecchia <salvatore.masecchia@disi.unige.it>
# License: New-BSD

import numpy as np
from numpy import linalg as la
from scipy import sparse

from sklearn.linear_model.base import LinearModel, _pre_fit


class AbstractLinearModel(LinearModel):
    """Abstract Linear Model. """

    def fit(self, X, y, *args, **kwargs):
        X = np.asanyarray(X)
        y = np.asanyarray(y)

        # Centering Data
        X, y, X_offset, y_offset, X_scale, precompute, Xy = \
            _pre_fit(X, y, None, self.precompute, self.normalize,
                     self.fit_intercept, copy=False)

        # Calling the class-specific train method
        self._fit(X, y, *args, **kwargs)

        # Fitting the intercept if required
        self._set_intercept(X_offset, y_offset, X_scale)

        self._trained = True
        return self


class RidgeRegression(AbstractLinearModel):
    """Ridge regression solved as direct method. """

    def __init__(self, mu=0.0, fit_intercept=True, precompute=False,
                 normalize=False):
        self.mu = mu
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.normalize = normalize

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
