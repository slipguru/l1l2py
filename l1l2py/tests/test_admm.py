import numpy as np
from numpy.testing import *

from ..admm import Lasso, ElasticNet #, ElasticNetCV, LassoCV, enet_path
#from ..base import RidgeRegression

def test_lasso_zero():
    """Check that Lasso can handle zero data."""
    X = [[0], [0], [0]]
    y = [0, 0, 0]
    model = Lasso(tau=0).fit(X, y)
    pred = model.predict([[1], [2], [3]])
    assert_array_almost_equal(model.coef_, [0])
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

    model = Lasso(tau=0.0).fit(X, y) # OLS
    pred = model.predict(T)
    assert_array_almost_equal([1, 1], model.coef_)
    assert_array_almost_equal([15, 19, 3], pred)

    model = Lasso(tau=0.5).fit(X, y)
    pred = model.predict(T)
    assert_array_almost_equal([.953, .953], model.coef_, 3)
    assert_array_almost_equal([14.625, 18.437, 3.187], pred, 3)

    model = Lasso(tau=1.0).fit(X, y)
    pred = model.predict(T)
    assert_array_almost_equal([.906, .906], model.coef_, 3)
    assert_array_almost_equal([14.25, 17.875, 3.375], pred, 3)

def test_lasso_on_examples_fat():
    """Test Lasso for different values of tau."""

    # A simple sum function
    X = [[1, 2, 3, 4], [5, 6, 7, 8]]
    y = [sum(x) for x in X]
    T = [[7, 8, 9, 10], [12, 11, 10, 9]]

    model = Lasso(tau=0.0).fit(X, y) # OLS
    pred = model.predict(T)
    assert_array_almost_equal([1, 1, 1, 1], model.coef_)
    assert_array_almost_equal([34, 42], pred)

    model = Lasso(tau=0.5).fit(X, y)
    pred = model.predict(T)
    assert_array_almost_equal([.984, .984, .984, .984], model.coef_, 3)
    assert_array_almost_equal([33.750, 41.625], pred, 3)

    model = Lasso(tau=1.0).fit(X, y)
    pred = model.predict(T)
    assert_array_almost_equal([.968, .968, .968, .968], model.coef_, 3)
    assert_array_almost_equal([33.5, 41.25], pred, 3)

def test_elasticnet_on_examples():
    """Test Elastic Net for different values of tau and mu."""

    # A simple sum function
    X = [[1, 2], [3, 4], [5, 6]]
    y = [sum(x) for x in X]
    T = [[7, 8], [9, 10], [2, 1]]

    model = ElasticNet(tau=0.0, mu=0.0).fit(X, y) # OLS
    pred = model.predict(T)
    assert_array_almost_equal([1, 1], model.coef_)
    assert_array_almost_equal([15, 19, 3], pred)

    model = ElasticNet(tau=0.5, mu=0.0).fit(X, y) # as Lasso
    pred = model.predict(T)
    assert_array_almost_equal([.953, .953], model.coef_, 3)
    assert_array_almost_equal([14.625, 18.437, 3.187], pred, 3)

    model = ElasticNet(tau=0.0, mu=0.5).fit(X, y) # RLS
    pred = model.predict(T)
    assert_array_almost_equal([.914, .914], model.coef_, 3)
    assert_array_almost_equal([14.314, 17.971, 3.343], pred, 3)

    model = ElasticNet(tau=0.5, mu=0.5).fit(X, y) # default
    pred = model.predict(T)
    assert_array_almost_equal([.871, .871], model.coef_, 3)
    assert_array_almost_equal([13.971, 17.457, 3.514], pred, 3)
