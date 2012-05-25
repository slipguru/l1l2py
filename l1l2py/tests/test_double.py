#-*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from nose.tools import *

from ..double import DoubleStepEstimator
from ..base import RidgeRegression
from ..proximal import Lasso

def _test_double_optimization():
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
    dstep = DoubleStepEstimator(Lasso(tau=1.0), RidgeRegression(mu=0.0)).train(X, y)
    
    # Coefficients
    lasso = dstep.selector
    ridge = dstep.estimator
    assert_array_almost_equal([0.90635646, 0.90635646], lasso.beta[:2])
    assert_array_almost_equal([1.0, 1.0], ridge.beta)
    assert_array_almost_equal([1.0, 1.0], dstep.beta[:2])
    
    # Prediction
    y_ = dstep.predict(T)
    assert_array_almost_equal([15., 19., 3.], y_)
    
def _test_double_optimization_intercept():
    """Test double optimization soundness with different parameters."""
    # A simple sum function with intercept
    intercept = 2.0
    X = [[1, 2], [3, 4], [5, 6]]
    y = [sum(x)+intercept for x in X]
    
    # OLS cascade
    dstep = DoubleStepEstimator(RidgeRegression(mu=0.0, fit_intercept=True),
                                RidgeRegression(mu=0.0, fit_intercept=True)).train(X, y)
    assert_almost_equal(intercept, dstep.intercept_)
    assert_array_almost_equal([1., 1.], dstep.beta)
    
    # OLS cascade: first step without fit_intercept
    dstep = DoubleStepEstimator(Ridge(mu=0.0, fit_intercept=False),
                                Ridge(mu=0.0, fit_intercept=True)).train(X, y)
    assert_almost_equal(intercept, dstep.intercept_)
    assert_array_almost_equal([1., 1.], dstep.beta)
    
    # OLS cascade: second step without fit_intercept
    dstep = DoubleStepEstimator(Ridge(mu=0.0, fit_intercept=True),
                                Ridge(mu=0.0, fit_intercept=False)).train(X, y)
    assert_almost_equal(0.0, dstep.intercept_)
    assert_array_almost_equal([-1., 3.], dstep.beta)
    assert_equal(False, dstep.train_intercept)
    
    # OLS cascade: all steps without fit_intercept
    dstep = DoubleStepEstimator(Ridge(mu=0.0, fit_intercept=False),
                                Ridge(mu=0.0, fit_intercept=False)).train(X, y)
    assert_almost_equal(0.0, dstep.intercept_)
    assert_array_almost_equal([-1., 3.], dstep.beta)
    
    
    
