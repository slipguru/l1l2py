import warnings
import numpy as np

from scikits.learn.linear_model.base import LinearModel

class ElasticNet(LinearModel):
    """
    scikits.learn model is:
        (1/2n)*||y - X*b||^2 + alpha*rho*||b||_1 + 0.5*alpha*(1-rho)||b||^2

    l1l2py model is:
        (1/n)*||y - X*b||^2 + tau*||b||_1 + mu*||b||^2

    Now, we keep our definition... we have to think if a different one
    is better or not.

    Notes:
        - with alpha and rho the default parameters (1.0 and 0.5)
          have a meaning: equal balance between l1 and l2 penalties.
          We do not have this balancing because the penalties parameter
          are unrelated.
          For now we choose 0.5 for both (but this value has no meaning
          right now)
        - We have to introduce the precompute behaviour... see Sofia's mail
          about coordinate descent and proximal methods
        - In l1l2py we have max_iter equal 100.000 instead of 1.000 and
          tol equal to 1e-5 instead of 1e-4... are them differences
          between cd and prox???

    """
    def __init__(self, tau=0.5, mu=0.5, fit_intercept=True,
                 precompute='auto', max_iter=100000, tol=1e-5,
                 adaptive=True,
                 continuation=True,
                 nonzero=False):
        self.tau = tau
        self.mu = mu
        self.coef_ = None
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.max_iter = max_iter
        self.tol = tol
        self.adaptive = adaptive
        self.continuation = continuation
        self.nonzero = nonzero

    def fit(self, X, y, Xy=None, coef_init=None, **params):
        """
        Notes: Xy is related to Gram and precompute parameter

        params may be each "init" parameter that the user want to
        override in the specific fit
        """
        from .algorithms import l1l2_regularization

        self._set_params(**params)
        X = np.asanyarray(X, dtype=np.float64)
        y = np.asanyarray(y, dtype=np.float64).ravel()

        X, y, Xmean, ymean = LinearModel._center_data(X, y, self.fit_intercept)

        if coef_init is None:
            self.coef_ = np.zeros(X.shape[1], dtype=np.float64)
        else:
            self.coef_ = coef_init.ravel()

        l1l2_proximal = l1l2_regularization
        self.coef_, self.niter_, self.energy_ = l1l2_proximal(X, y,
                                                    self.mu, self.tau,
                                                    beta=self.coef_,
                                                    kmax=self.max_iter,
                                                    tolerance=self.tol,
                                                    adaptive=self.adaptive,
                                                    continuation=self.continuation,
                                                    nonzero=self.nonzero)
        
        self.coef_ = self.coef_.ravel()

        self._set_intercept(Xmean, ymean)

        if self.niter_ == self.max_iter:
            warnings.warn('Objective did not converge, you might want'
                          ' to increase the number of iterations')

        return self

class Lasso(ElasticNet):
    def __init__(self, tau=0.5, fit_intercept=True,
                 precompute='auto', max_iter=100000, tol=1e-5,
                 adaptive=True,
                 continuation=True,
                 nonzero=False):
        super(Lasso, self).__init__(tau=tau, mu=0.0,
                            fit_intercept=fit_intercept,
                            precompute=precompute,
                            max_iter=max_iter,
                            tol=tol,
                            adaptive=adaptive,
                            continuation=continuation,
                            nonzero=nonzero)

###############################################################################
# Elastic net tests from scikits.learn internals
##############################################################################

from scikits.learn.linear_model.coordinate_descent import ElasticNet as ElasticNetSK
from numpy.testing.utils import assert_array_almost_equal, assert_almost_equal

def conv_(alpha, rho):
    tau, mu = 2.*alpha*rho, alpha*(1-rho)
    return {'tau': tau, 'mu': mu}

def test_enet_toy_both():
    for en in ElasticNetSK, ElasticNet:
        yield _test_enet_toy, en

def _test_enet_toy(ElasticNet):
    """
    Test ElasticNet for various parameters of alpha and rho.

    Actualy, the parameters alpha = 0 should not be alowed. However,
    we test it as a border case.

    ElasticNet is tested with and without precomputed Gram matrix
    """

    X = np.array([[-1.], [0.], [1.]])
    Y = [-1, 0, 1]       # just a straight line
    T = [[2.], [3.], [4.]]  # test sample

    n = X.shape[0]

    # this should be the same as lasso... no OLS
    try:
        clf = ElasticNet(alpha=0, rho=1.0)
    except:
        clf = ElasticNet(**conv_(alpha=0, rho=1.0))
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(pred, [2, 3, 4])
    #assert_almost_equal(clf.dual_gap_, 0)

    try:
        clf = ElasticNet(alpha=0.5, rho=0.3)
    except:
        clf = ElasticNet(**conv_(alpha=0.5, rho=0.3))
    clf.fit(X, Y, max_iter=1000, precompute=False)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    #assert_almost_equal(clf.dual_gap_, 0)
    #
    #clf.fit(X, Y, max_iter=1000, precompute=True) # with Gram
    #pred = clf.predict(T)
    #assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    #assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    #assert_almost_equal(clf.dual_gap_, 0)
    #
    #clf.fit(X, Y, max_iter=1000, precompute=np.dot(X.T, X)) # with Gram
    #pred = clf.predict(T)
    #assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    #assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    #assert_almost_equal(clf.dual_gap_, 0)
    #

    try:
        clf = ElasticNet(alpha=0.5, rho=0.5)
    except:
        clf = ElasticNet(**conv_(alpha=0.5, rho=0.5))
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.45454], 3)
    assert_array_almost_equal(pred, [0.9090, 1.3636, 1.8181], 3)
    #assert_almost_equal(clf.dual_gap_, 0)
