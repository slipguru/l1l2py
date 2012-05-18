import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from nose.tools import *

from ..estimators import Ridge, ElasticNet, Lasso, \
                         ElasticNetCV, LassoCV, enet_path

def test_ridge_on_examples():
    """Test Ridge regression for different values of mu."""

    # A simple sum function
    X = [[1, 2], [3, 4], [5, 6]]
    y = [sum(x)+1 for x in X]
    T = [[7, 8], [9, 10], [2, 1]]
        
    clf = Ridge(mu=0.0) # OLS
    clf.fit(X, y)
    pred = clf.predict(T)
    assert_array_almost_equal([1, 1], clf.coef_)
    assert_array_almost_equal([16, 20, 4], pred)
    assert_almost_equal(1.0, clf.intercept_)

    clf = Ridge(mu=0.5)
    clf.fit(X, y)
    pred = clf.predict(T)
    assert_array_almost_equal([0.91428571, 0.91428571], clf.coef_)
    assert_array_almost_equal([15.31428571, 18.97142857, 4.34285714], pred)
    
    clf = Ridge(mu=1.0)
    clf.fit(X, y)
    pred = clf.predict(T)
    assert_array_almost_equal([0.84210526, 0.84210526], clf.coef_)
    assert_array_almost_equal([14.73684211, 18.10526316, 4.63157895], pred)


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


def test_estimators_equivalences():
    """ Test the equivalence of the estimators with different parameters."""
    # Data creation
    np.random.seed(0)
    coef = np.random.randn(200)
    coef[10:] = 0.0 # only the top 10 features are impacting the model
    X = np.random.randn(50, 200)
    y = np.dot(X, coef) # without error

    # OLS
    enet = ElasticNet(tau=0.0, mu=0.0).fit(X, y)
    lasso = Lasso(tau=0.0).fit(X, y)
    ridge = Ridge(mu=0.0).fit(X, y)
    assert_array_almost_equal(enet.coef_, lasso.coef_, 3)
    assert_array_almost_equal(enet.coef_, ridge.coef_, 3)
    assert_array_almost_equal(lasso.coef_, ridge.coef_, 3)

    # Ridge
    enet = ElasticNet(tau=0.0, mu=0.5).fit(X, y)
    ridge = Ridge(mu=0.5).fit(X, y)
    assert_array_almost_equal(enet.coef_, ridge.coef_, 3)

    # Lasso
    enet = ElasticNet(tau=0.5, mu=0.0).fit(X, y)
    lasso = Lasso(tau=0.5).fit(X, y)
    assert_array_almost_equal(enet.coef_, lasso.coef_, 3)


def test_enet_path():
    """Test elastic net path."""

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
    """ Test ElasticNet cross validation."""

    # Data creation
    np.random.seed(0)
    coef = np.random.randn(200)
    coef[10:] = 0.0 # only the top 10 features are impacting the model
    X = np.random.randn(50, 200)
    y = np.dot(X, coef) # without error

    # Automatic generation of the taus
    clf = ElasticNetCV(n_taus=100, eps=1e-3, mu=1e2, max_iter=10)
    clf.fit(X, y)

    assert_almost_equal(clf.tau, 0.9743, 2)

def test_lasso_cv():
    """ Test Lasso cross validation."""

    # Data creation
    np.random.seed(0)
    coef = np.random.randn(200)
    coef[10:] = 0.0 # only the top 10 features are impacting the model
    X = np.random.randn(50, 200)
    y = np.dot(X, coef) # without error

    # Automatic generation of the taus
    clf = LassoCV(n_taus=100, eps=1e-3, max_iter=10)
    clf.fit(X, y)
    assert_almost_equal(clf.tau, 0.02099, 2)
