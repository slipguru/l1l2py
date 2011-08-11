#-*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from nose.tools import *

from ..double import DoubleStepEstimator
from ..estimators import Ridge, Lasso

def test_pipe():
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
    ridge = dstep.regressor
    assert_array_almost_equal([0.90635646, 0.90635646], lasso.coef_[:2])
    assert_array_almost_equal([1.0, 1.0], ridge.coef_)
    assert_array_almost_equal([1.0, 1.0], dstep.coef_[:2])
    
    # Prediction
    y_ = dstep.predict(T)
    assert_array_almost_equal([15., 19., 3.], y_)
    
    
