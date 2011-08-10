#-*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from nose.tools import *

from ..double import DoubleStepEstimator
from ..estimators import Ridge, Lasso

class Selector(object):
    def __init__(self, n_sel=None):
        self.n_sel = n_sel
    def fit(self, X, y):
        self.coef_ = np.random.randn(X.shape[1])
        self.coef_[self.n_sel:] = 0.0
    def transform(self, X):
        nonzero = np.flatnonzero(self.coef_)
        return X[:,nonzero]

def test_selector():
    """..."""
    # Data creation
    np.random.seed(0)
    coef = np.random.randn(200)
    X = np.random.randn(50, 200)
    y = np.empty(50) # random labels
    
    sel = Selector(n_sel=10)
    sel.fit(X, y)
    assert_equal(10, len(np.flatnonzero(sel.coef_)))
    assert_array_almost_equal(X[:,:10], sel.transform(X))
    
def test_pipe():
    """..."""
    # Data creation
    np.random.seed(0)
    coef = np.random.randn(200)
    coef[10:] = 0.0 # only the top 10 features are impacting the model
    X = np.random.randn(50, 200)
    y = np.dot(X, coef) # without error
    
    lasso = Lasso(tau=1.0)
    ridge = Ridge(mu=1.0)
    dstep = DoubleStepEstimator(lasso, ridge)
    
    dstep.fit(X, y)
    lasso.fit(X, y)
       
    ridge.fit(lasso.transform(X), y)   
    
    assert_equal(len(lasso.coef_), len(dstep.coef_))
    assert_equal(len(ridge.coef_), len(np.flatnonzero(dstep.coef_)))
    
    ridge_pred = ridge.predict(lasso.transform(X))
    dstep_pred = dstep.predict(X)
    assert_array_almost_equal(ridge_pred, dstep_pred)
    
    
