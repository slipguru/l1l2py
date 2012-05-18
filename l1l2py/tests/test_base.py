# Author: Salvatore Masecchia <salvatore.masecchia@disi.unige.it>
# License: New-BSD

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from nose.tools import *

from ..base import AbstractLinearModel

def test_abstractmodel():
    ab = AbstractLinearModel()
    assert_raises(NotImplementedError, ab.train, np.empty((2, 2)), np.empty(2))
    assert_raises(TypeError, ab.predict, np.eye(2))

    assert_equal(None, ab.beta)
    assert_equal(None, ab.intercept)


#def test_lasso_cv():
#    """ Test Lasso cross validation."""
#
#    # Data creation
#    np.random.seed(0)
#    coef = np.random.randn(200)
#    coef[10:] = 0.0 # only the top 10 features are impacting the model
#    X = np.random.randn(50, 200)
#    y = np.dot(X, coef) # without error
#
#    # Automatic generation of the taus
#    clf = LassoCV(n_taus=100, eps=1e-3, max_iter=10)
#    clf.fit(X, y)
#    assert_almost_equal(clf.tau, 0.02099, 2)
