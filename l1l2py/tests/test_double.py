#-*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from nose.tools import *

from ..double import DoubleStepEstimator
from ..estimators import Ridge, Lasso

def test_double_optimization():
    """Test double optimization on a simple example."""
    # A simple sparse-sum function
    X = [[1, 2], [3, 4], [5, 6]]
    y = [sum(x) for x in X]
    T = [[7, 8], [9, 10], [2, 1]]
    
    # noisy variables
    np.random.seed(0)
    X = np.c_[X, np.random.random((3, 100))]
    T = np.c_[T, np.random.random((3, 100))]    
    
    # Select the first 2 variables and calculate a linear model on them
    dstep = DoubleStepEstimator(Lasso(tau=1.0), Ridge(mu=0.0)).fit(X, y)
    
    # Coefficients
    lasso = dstep.selector
    ridge = dstep.estimator
    assert_array_almost_equal([0.90635646, 0.90635646], lasso.coef_[:2])
    assert_array_almost_equal([1.0, 1.0], ridge.coef_)
    assert_array_almost_equal([1.0, 1.0], dstep.coef_[:2])
    
    # Prediction
    y_ = dstep.predict(T)
    assert_array_almost_equal([15., 19., 3.], y_)
    
def test_double_optimization_intercept():
    """Test double optimization soundness with different parameters."""
    # A simple sum function with intercept
    intercept = 2.0
    X = [[1, 2], [3, 4], [5, 6]]
    y = [sum(x)+intercept for x in X]
    
    # OLS cascade
    dstep = DoubleStepEstimator(Ridge(mu=0.0, fit_intercept=True),
                                Ridge(mu=0.0, fit_intercept=True)).fit(X, y)
    assert_almost_equal(intercept, dstep.intercept_)
    assert_array_almost_equal([1., 1.], dstep.coef_)
    
    # OLS cascade: first step without fit_intercept
    dstep = DoubleStepEstimator(Ridge(mu=0.0, fit_intercept=False),
                                Ridge(mu=0.0, fit_intercept=True)).fit(X, y)
    assert_almost_equal(intercept, dstep.intercept_)
    assert_array_almost_equal([1., 1.], dstep.coef_)
    
    # OLS cascade: second step without fit_intercept
    dstep = DoubleStepEstimator(Ridge(mu=0.0, fit_intercept=True),
                                Ridge(mu=0.0, fit_intercept=False)).fit(X, y)
    assert_almost_equal(0.0, dstep.intercept_)
    assert_array_almost_equal([-1., 3.], dstep.coef_)
    assert_equal(False, dstep.fit_intercept)
    
    # OLS cascade: all steps without fit_intercept
    dstep = DoubleStepEstimator(Ridge(mu=0.0, fit_intercept=False),
                                Ridge(mu=0.0, fit_intercept=False)).fit(X, y)
    assert_almost_equal(0.0, dstep.intercept_)
    assert_array_almost_equal([-1., 3.], dstep.coef_)
    
    
    
