# Author: Salvatore Masecchia <salvatore.masecchia@disi.unige.it>
# License: New-BSD

import numpy as np
from numpy import linalg as la
from numpy.testing import assert_raises, assert_equal, assert_almost_equal, \
    assert_array_almost_equal

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from ..base import AbstractLinearModel, RidgeRegression


class MockModel(AbstractLinearModel):
    def __init__(self, fit_intercept=True, precompute=False, normalize=False):
        self.fit_intercept = fit_intercept
        self.intercept_ = 0.0
        self.coef_ = None
        self.precompute = precompute
        self.normalize = normalize

    def _fit(self, X, y):
        self.coef_ = la.lstsq(X, y)[0]


def test_abstractmodel():
    ab = MockModel()
    assert_raises(NotFittedError, check_is_fitted, ab, 'n_iter_')  # Not trained!

    assert_equal(None, ab.coef_)
    assert_equal(0.0, ab.intercept_)


def test_intercept_value():
    X = [[1, 1, 1], [2, 2, 2]]
    y = [sum(x) + 2. for x in X]

    model = MockModel(fit_intercept=False)
    model.fit(X, y)
    assert_almost_equal(0.0, model.intercept_)

    model = MockModel(fit_intercept=True)
    model.fit(X, y)
    assert_almost_equal(2., model.intercept_)
    assert_almost_equal(np.ones(3), model.coef_)


def test_prediction():
    X = [[1, 1, 1], [2, 2, 2]]
    y = [sum(x) + 2. for x in X]

    T = [[1, 1, 1], [3, 3, 3]]
    Ty = [sum(t) + 2. for t in T]

    Xc = X - np.mean(X, axis=0)
    yc = y - np.mean(y)
    Xm = np.c_[np.ones(2), X]

    # Standard call
    model = MockModel(fit_intercept=True).fit(X, y)
    assert_almost_equal(2., model.intercept_)
    assert_almost_equal([1, 1, 1], model.coef_)
    assert_almost_equal(y, model.predict(X))
    assert_almost_equal(Ty, model.predict(T))

    # Pre-centered X
    model = MockModel(fit_intercept=False).fit(Xc, y)
    assert_almost_equal(0.0, model.intercept_)
    assert_almost_equal([1, 1, 1], model.coef_)
    assert_almost_equal(yc, model.predict(Xc))
    assert_almost_equal(Ty - np.mean(y),
                        model.predict(T - np.mean(X, axis=0)))

    # Ones-column added (included intercept)
    model = MockModel(fit_intercept=False).fit(Xm, y)
    assert_almost_equal(0.0, model.intercept_)
    assert_almost_equal([2, 1, 1, 1], model.coef_)
    assert_almost_equal(y, model.predict(Xm))
    assert_almost_equal(Ty, model.predict(np.c_[np.ones(2), T]))

def test_ridge():
    """Test Ridge regression for different values of mu."""
    # A simple sum function (with intercept)
    X = [[1, 2], [3, 4], [5, 6]]
    y = [sum(x)+1 for x in X]
    T = [[7, 8], [9, 10], [2, 1]]

    model = RidgeRegression(mu=0.0).fit(X, y)  # OLS
    assert_array_almost_equal([1, 1], model.coef_)
    assert_array_almost_equal([16, 20, 4], model.predict(T))
    assert_almost_equal(1.0, model.intercept_)

    # Equivalence with standard numpy least squares
    Xc = X - np.mean(X, axis=0)
    assert_almost_equal(la.lstsq(Xc, y)[0], model.coef_)

    model = RidgeRegression(mu=0.5).fit(X, y)
    assert_array_almost_equal([0.91428571, 0.91428571], model.coef_)
    assert_array_almost_equal([15.31428571, 18.97142857, 4.34285714],
                              model.predict(T))

    model = RidgeRegression(mu=1.0).fit(X, y)
    assert_array_almost_equal([0.84210526, 0.84210526], model.coef_)
    assert_array_almost_equal([14.73684211, 18.10526316, 4.63157895],
                              model.predict(T))
