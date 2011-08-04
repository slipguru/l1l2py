#-*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from ..double import DoubleStepEstimator

class HalfSelection(object):
    def fit(self, X, y):
        self.coef_ = np.ones(X.shape[1])
        self.coef_[:X.shape[1]/2] = 0.0
    def transform(self, X):
        nonzero = np.flatnonzero(self.coef_)
        return X[:,nonzero]

def test_half():
    assert False

def test_mock():
    pass
