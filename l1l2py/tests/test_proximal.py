#-*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from ..proximal import ElasticNet, Lasso, ElasticNetCV, enet_path

def test_lasso_zero():
    """Check that Lasso can handle zero data."""
    X = [[0], [0], [0]]
    y = [0, 0, 0]
    clf = Lasso(tau=0).fit(X, y)
    pred = clf.predict([[1], [2], [3]])
    assert_array_almost_equal(clf.coef_, [0])
    assert_array_almost_equal(pred, [0, 0, 0])


def test_lasso_on_examples():
    """Test Lasso for different values of tau."""

    # A simple sum function
    X = [[1, 2], [3, 4], [5, 6]]
    y = [sum(x) for x in X]
    T = [[7, 8], [9, 10], [2, 1]]

    clf = Lasso(tau=0.0) # OLS
    clf.fit(X, y)
    pred = clf.predict(T)
    assert_array_almost_equal([1, 1], clf.coef_)
    assert_array_almost_equal([15, 19, 3], pred)

    clf = Lasso(tau=0.5)
    clf.fit(X, y)
    pred = clf.predict(T)
    assert_array_almost_equal([.953, .953], clf.coef_, 3)
    assert_array_almost_equal([14.625, 18.437, 3.187], pred, 3)

    clf = Lasso(tau=1.0)
    clf.fit(X, y)
    pred = clf.predict(T)
    assert_array_almost_equal([.906, .906], clf.coef_, 3)
    assert_array_almost_equal([14.25, 17.875, 3.375], pred, 3)

def test_elasticnet_on_examples():
    """Test Elastic Net for different values of tau and mu."""

    # A simple sum function
    X = [[1, 2], [3, 4], [5, 6]]
    y = [sum(x) for x in X]
    T = [[7, 8], [9, 10], [2, 1]]

    clf = ElasticNet(tau=0.0, mu=0.0) # OLS
    clf.fit(X, y)
    pred = clf.predict(T)
    assert_array_almost_equal([1, 1], clf.coef_)
    assert_array_almost_equal([15, 19, 3], pred)

    clf = ElasticNet(tau=0.5, mu=0.0) # as Lasso
    clf.fit(X, y)
    pred = clf.predict(T)
    assert_array_almost_equal([.953, .953], clf.coef_, 3)
    assert_array_almost_equal([14.625, 18.437, 3.187], pred, 3)

    clf = ElasticNet(tau=0.0, mu=0.5) # RLS
    clf.fit(X, y)
    pred = clf.predict(T)
    assert_array_almost_equal([.914, .914], clf.coef_, 3)
    assert_array_almost_equal([14.314, 17.971, 3.343], pred, 3)

    clf = ElasticNet(tau=0.5, mu=0.5) # default
    clf.fit(X, y)
    pred = clf.predict(T)
    assert_array_almost_equal([.871, .871], clf.coef_, 3)
    assert_array_almost_equal([13.971, 17.457, 3.514], pred, 3)

    clf = ElasticNet(tau=0.5, mu=0.5, adaptive_step_size=False) # without adaptive
    clf.fit(X, y)
    pred = clf.predict(T)
    assert_array_almost_equal([.871,  .871], clf.coef_, 3)
    assert_array_almost_equal([13.971, 17.457, 3.514], pred, 3)


def test_intercept():
    """Test for a manual intercept fit."""

    # A simple sum function
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([sum(x) for x in X])
    T = np.array([[7, 8], [9, 10], [2, 1]])

    meanx, meany = X.mean(axis=0), y.mean()
    Xc = X - meanx
    yc = y - meany

    clf = Lasso(tau=0.0, fit_intercept=False) # OLS
    clf.fit(Xc, yc)
    pred = clf.predict(T-meanx) + meany

    assert_array_almost_equal([1, 1], clf.coef_)
    assert_array_almost_equal([15, 19, 3], pred)

def test_enet_path():
    X = [[1, 2], [3, 4], [5, 6]]
    y = [sum(x) for x in X]
    T = [[7, 8], [9, 10], [2, 1]]

    X = np.asarray(X)
    y = np.asarray(y)
    T = np.asarray(T)

    models = enet_path(X, y, mu=0.0, n_taus=10, eps=1e-3)
    assert_equals(10, len(models))
    assert_almost_equal(1e-3, models[-1].tau/models[0].tau)

    # External intercept
    X, y, Xmean, ymean = ElasticNet._center_data(X, y, True)
    models_test = enet_path(X, y, mu=0.0, n_taus=10, eps=1e-3,
                            fit_intercept=False)
    for m, mt in zip(models, models_test):
        assert_array_almost_equal(m.coef_, mt.coef_)

    # Manual taus
    models_test = enet_path(X, y, mu=0.0, taus=[m.tau for m in models])
    for m, mt in zip(models, models_test):
        assert_array_almost_equal(m.coef_, mt.coef_)

def test_enet_cv():
    # Data creation
    np.random.seed(0)
    coef = np.random.randn(200)
    coef[10:] = 0.0 # only the top 10 features are impacting the model
    X = np.random.randn(50, 200)
    y = np.dot(X, coef) # without error

    # Automatic generation of the taus
    clf = ElasticNetCV(n_taus=100, eps=1e-3, mu=1e-1)
    clf.fit(X, y, max_iter=10)

    import pylab as pl
    pl.plot(clf.mse_path_)
    pl.show()

    assert_almost_equal(clf.tau, 0.5575, 2)



# Per il path abbiamo bisogno della RidgeRegression
def _test_ridge():
    from scikits.learn.linear_model import Ridge, ElasticNet as ElasticNetSKL
    X = np.random.random((100, 1001))
    y = np.random.random(100)

    print ridge_regression(X, y, mu=0.0)
    clf = Ridge(alpha=0.0, fit_intercept=False)
    clf.fit(X, y)
    print clf.coef_

    # Nota che tra mu ed alpha c'è una differenza...
    # La nostra ridge regression fornisce gli stessi risultati
    # di enet con tau=0.0, la Ridge invece deve avere in input
    # alpha = N*mu ma è in genere più veloce
    print ridge_regression(X, y, mu=0.5)
    clf = Ridge(alpha=len(y)*0.5, fit_intercept=False) ### nota la relazione
    clf.fit(X, y)
    print clf.coef_

    # L'implementazione cd di enet non può essere
    # portata a quella della ridge regression,
    # ma in ogni caso i lambda sono diversi dai mu... può aver senso
    #en = ElasticNetSKL(alpha=1.0, rho=0.0)
    #en.fit(X, y, fit_intercept=False)
    #print np.asarray(en.coef_)


from l1l2py.algorithms import *
from l1l2py.tests import _TEST_DATA_PATH
from nose.tools import *

class TestAlgorithms(object):

    def setup(self):
        data = np.loadtxt(_TEST_DATA_PATH)
        self.X = data[:,:-1]
        self.Y = data[:,-1]

    def test_data(self):
        assert_equals((30, 40), self.X.shape)
        assert_equals((30, ), self.Y.shape)

    def test_rls(self):
        # case n >= d
        for penalty in np.linspace(0.0, 1.0, 5):
            value = ridge_regression(self.X, self.Y, penalty)
            assert_equal(value.shape, (self.X.shape[1], 1))

        expected = ridge_regression(self.X, self.Y, 0.0)
        value = ridge_regression(self.X, self.Y)
        assert_true(np.allclose(expected, value))

        # case d > n
        X = self.X.T
        Y = self.X[0:1,:].T
        for penalty in np.linspace(0.0, 1.0, 5):
            value = ridge_regression(X, Y, penalty)
            assert_equal(value.shape, (X.shape[1], 1))

        expected = ridge_regression(X, Y, 0.0)
        value = ridge_regression(X, Y)
        assert_true(np.allclose(expected, value))

    def test_l1_bound(self):
        tau_max = l1_bound(self.X, self.Y)

        beta, k = l1l2_regularization(self.X, self.Y, 0.0, tau_max+1e3)
        assert_equals(0, len(np.flatnonzero(beta)))

        beta, k = l1l2_regularization(self.X, self.Y, 0.0, tau_max-1e-3)
        assert_equals(1, len(np.flatnonzero(beta)))
