import numpy as np
from numpy.testing import *

from ..admm import Lasso#, ElasticNetCV, LassoCV, enet_path
#from ..base import RidgeRegression

def test_lasso_zero():
    """Check that Lasso can handle zero data."""
    X = [[0], [0], [0]]
    y = [0, 0, 0]
    model = Lasso(tau=0).train(X, y)
    pred = model.predict([[1], [2], [3]])
    assert_array_almost_equal(model.beta, [0])
    assert_array_almost_equal(pred, [0, 0, 0])

def test_factor():
    from ..admm import factor
    X = np.array([[1., 2., 3.], [4., 5., 6.]])
    
    # To test!
    #print
    #print factor(X, 1.0)
    
def test_lasso_on_examples():
    """Test Lasso for different values of tau."""

    # A simple sum function
    X = [[1, 2], [3, 4], [5, 6]]
    y = [sum(x) for x in X]
    T = [[7, 8], [9, 10], [2, 1]]

    model = Lasso(tau=0.0).train(X, y) # OLS
    pred = model.predict(T)
    assert_array_almost_equal([1, 1], model.beta)
    assert_array_almost_equal([15, 19, 3], pred)

    model = Lasso(tau=0.5).train(X, y)
    pred = model.predict(T)
    assert_array_almost_equal([.953, .953], model.beta, 3)
    assert_array_almost_equal([14.625, 18.437, 3.187], pred, 3)

    model = Lasso(tau=1.0).train(X, y)
    pred = model.predict(T)
    assert_array_almost_equal([.906, .906], model.beta, 3)
    assert_array_almost_equal([14.25, 17.875, 3.375], pred, 3)
    
def test_lasso_on_examples_fat():
    """Test Lasso for different values of tau."""

    # A simple sum function
    X = [[1, 2, 3, 4], [5, 6, 7, 8]]
    y = [sum(x) for x in X]
    T = [[7, 8, 9, 10], [12, 11, 10, 9]]

    model = Lasso(tau=0.0).train(X, y) # OLS
    pred = model.predict(T)
    assert_array_almost_equal([1, 1, 1, 1], model.beta)
    assert_array_almost_equal([34, 42], pred)

    model = Lasso(tau=0.5).train(X, y)
    pred = model.predict(T)
    assert_array_almost_equal([.984, .984, .984, .984], model.beta, 3)
    assert_array_almost_equal([33.750, 41.625], pred, 3)
    
    model = Lasso(tau=1.0).train(X, y)
    pred = model.predict(T)
    assert_array_almost_equal([.968, .968, .968, .968], model.beta, 3)
    assert_array_almost_equal([33.5, 41.25], pred, 3)