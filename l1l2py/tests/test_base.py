# Author: Salvatore Masecchia <salvatore.masecchia@disi.unige.it>
# License: New-BSD

import numpy as np
from numpy import linalg as la
from numpy.testing import *

from ..base import AbstractLinearModel, RidgeRegression

class MockModel(AbstractLinearModel):
    def _train(self, X, y):
        self._beta = la.lstsq(X, y)[0]

def test_abstractmodel():
    ab = AbstractLinearModel()
    assert_raises(NotImplementedError, ab.train, np.empty((2, 2)), np.empty(2))
    assert_raises(RuntimeError, ab.predict, np.eye(2)) # Not trained!

    assert_equal(None, ab.beta)
    assert_equal(None, ab.intercept)
    assert_equal(False, ab.trained)
    
def test_intercept_value():    
    X = [[1, 1, 1], [2, 2, 2]]
    y = [sum(x) + 2. for x in X]
    model = MockModel()
    
    model.train(X, y, fit_intercept=False)
    assert_almost_equal(0.0, model.intercept)
    
    model.train(X, y, fit_intercept=True)
    assert_almost_equal(2., model.intercept)
    assert_almost_equal(np.ones(3), model.beta)
    
def test_prediction():    
    X = [[1, 1, 1], [2, 2, 2]]
    y = [sum(x) + 2. for x in X]
    model = MockModel()
    
    T = [[1, 1, 1], [3, 3, 3]]
    Ty = [sum(t) + 2. for t in T]
    
    Xc = X - np.mean(X, axis=0)
    yc = y - np.mean(y)
    Xm = np.c_[np.ones(2), X]
    
    # Standard call
    model.train(X, y, fit_intercept=True)
    assert_almost_equal(2., model.intercept)
    assert_almost_equal([1, 1, 1], model.beta)
    assert_almost_equal(y, model.predict(X))
    assert_almost_equal(Ty, model.predict(T))
       
    # Pre-centered X
    model.train(Xc, y, fit_intercept=False)
    assert_almost_equal(0.0, model.intercept)
    assert_almost_equal([1, 1, 1], model.beta)
    assert_almost_equal(yc, model.predict(Xc))
    assert_almost_equal(Ty - np.mean(y),
                        model.predict(T - np.mean(X, axis=0)))
    
    # Ones-column added (included intercept)
    model.train(Xm, y, fit_intercept=False)
    assert_almost_equal(0.0, model.intercept)
    assert_almost_equal([2, 1, 1, 1], model.beta)
    assert_almost_equal(y, model.predict(Xm))
    assert_almost_equal(Ty, model.predict(np.c_[np.ones(2), T]))

def test_ridge():
    """Test Ridge regression for different values of mu."""
    # A simple sum function (with intercept)
    X = [[1, 2], [3, 4], [5, 6]]
    y = [sum(x)+1 for x in X]
    T = [[7, 8], [9, 10], [2, 1]]
          
    model = RidgeRegression(mu=0.0) # OLS
    model.train(X, y)
    assert_array_almost_equal([1, 1], model.beta)
    assert_array_almost_equal([16, 20, 4], model.predict(T))
    assert_almost_equal(1.0, model.intercept)
    
    # Equivalence with standard numpy least squares
    Xc = X - np.mean(X, axis=0)
    assert_almost_equal(la.lstsq(Xc, y)[0], model.beta)

    model = RidgeRegression(mu=0.5)
    model.train(X, y)
    assert_array_almost_equal([0.91428571, 0.91428571], model.beta)
    assert_array_almost_equal([15.31428571, 18.97142857, 4.34285714],
                              model.predict(T))

    model = RidgeRegression(mu=1.0)
    model.train(X, y)
    assert_array_almost_equal([0.84210526, 0.84210526], model.beta)
    assert_array_almost_equal([14.73684211, 18.10526316, 4.63157895],
                              model.predict(T))
    