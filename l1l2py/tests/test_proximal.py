import numpy as np
from numpy.testing import *

from ..proximal import ElasticNet, Lasso#, ElasticNetCV, LassoCV, enet_path
from ..base import RidgeRegression

def test_lasso_zero():
    """Check that Lasso can handle zero data."""
    X = [[0], [0], [0]]
    y = [0, 0, 0]
    model = Lasso(tau=0).train(X, y)
    pred = model.predict([[1], [2], [3]])
    assert_array_almost_equal(model.beta, [0])
    assert_array_almost_equal(pred, [0, 0, 0])

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

def test_elasticnet_on_examples():
    """Test Elastic Net for different values of tau and mu."""

    # A simple sum function
    X = [[1, 2], [3, 4], [5, 6]]
    y = [sum(x) for x in X]
    T = [[7, 8], [9, 10], [2, 1]]

    model = ElasticNet(tau=0.0, mu=0.0).train(X, y) # OLS
    pred = model.predict(T)
    assert_array_almost_equal([1, 1], model.beta)
    assert_array_almost_equal([15, 19, 3], pred)

    model = ElasticNet(tau=0.5, mu=0.0).train(X, y) # as Lasso
    pred = model.predict(T)
    assert_array_almost_equal([.953, .953], model.beta, 3)
    assert_array_almost_equal([14.625, 18.437, 3.187], pred, 3)

    model = ElasticNet(tau=0.0, mu=0.5).train(X, y) # RLS
    pred = model.predict(T)
    assert_array_almost_equal([.914, .914], model.beta, 3)
    assert_array_almost_equal([14.314, 17.971, 3.343], pred, 3)

    model = ElasticNet(tau=0.5, mu=0.5).train(X, y) # default
    pred = model.predict(T)
    assert_array_almost_equal([.871, .871], model.beta, 3)
    assert_array_almost_equal([13.971, 17.457, 3.514], pred, 3)

    # with adaptive
    model = ElasticNet(tau=0.5, mu=0.5, adaptive_step_size=True).train(X, y)
    pred = model.predict(T)
    assert_array_almost_equal([.871,  .871], model.beta, 3)
    assert_array_almost_equal([13.971, 17.457, 3.514], pred, 3)


def test_intercept():
    """Test for a manual intercept train."""

    # A simple sum function
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([sum(x) for x in X])
    T = np.array([[7, 8], [9, 10], [2, 1]])

    meanx, meany = X.mean(axis=0), y.mean()
    Xc = X - meanx
    yc = y - meany

    model = Lasso(tau=0.0, fit_intercept=False).train(Xc, yc) # OLS
    pred = model.predict(T - meanx) + meany

    assert_array_almost_equal([1, 1], model.beta)
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
    enet = ElasticNet(tau=0.0, mu=0.0).train(X, y)
    lasso = Lasso(tau=0.0).train(X, y)
    ridge = RidgeRegression(mu=0.0).train(X, y)
    assert_array_almost_equal(enet.beta, lasso.beta, 3)
    assert_array_almost_equal(enet.beta, ridge.beta, 3)
    assert_array_almost_equal(lasso.beta, ridge.beta, 3)

    # RidgeRegression
    enet = ElasticNet(tau=0.0, mu=0.5).train(X, y)
    ridge = RidgeRegression(mu=0.5).train(X, y)
    assert_array_almost_equal(enet.beta, ridge.beta, 3)

    # Lasso
    enet = ElasticNet(tau=0.5, mu=0.0).train(X, y)
    lasso = Lasso(tau=0.5).train(X, y)
    assert_array_almost_equal(enet.beta, lasso.beta, 3)


def _test_enet_path():
    """Test elastic net path."""

    X = [[1, 2], [3, 4], [5, 6]]
    y = [sum(x) for x in X]
    T = [[7, 8], [9, 10], [2, 1]]

    X = np.asarray(X)
    y = np.asarray(y)
    T = np.asarray(T)

    models = enet_path(X, y, mu=0.0, n_taus=10, eps=1e-3)
    assert_equal(10, len(models))
    assert_almost_equal(1e-3, models[-1].tau/models[0].tau)

    # External intercept
    X, y, Xmean, ymean = ElasticNet._center_data(X, y, True)
    models_test = enet_path(X, y, mu=0.0, n_taus=10, eps=1e-3,
                            train_intercept=False)
    for m, mt in zip(models, models_test):
        assert_array_almost_equal(m.beta, mt.beta)

    # Manual taus
    models_test = enet_path(X, y, mu=0.0, taus=[m.tau for m in models])
    for m, mt in zip(models, models_test):
        assert_array_almost_equal(m.beta, mt.beta)

def _test_enet_cv():
    """ Test ElasticNet cross validation."""

    # Data creation
    np.random.seed(0)
    coef = np.random.randn(200)
    coef[10:] = 0.0 # only the top 10 features are impacting the model
    X = np.random.randn(50, 200)
    y = np.dot(X, coef) # without error

    # Automatic generation of the taus
    model = ElasticNetCV(n_taus=100, eps=1e-3, mu=1e2, max_iter=10)
    model.train(X, y)

    assert_almost_equal(model.tau, 0.9743, 2)

def _test_lasso_cv():
    """ Test Lasso cross validation."""

    # Data creation
    np.random.seed(0)
    coef = np.random.randn(200)
    coef[10:] = 0.0 # only the top 10 features are impacting the model
    X = np.random.randn(50, 200)
    y = np.dot(X, coef) # without error

    # Automatic generation of the taus
    model = LassoCV(n_taus=100, eps=1e-3, max_iter=10)
    model.train(X, y)
    assert_almost_equal(model.tau, 0.02099, 2)
