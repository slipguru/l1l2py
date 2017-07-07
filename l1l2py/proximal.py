# Author: Salvatore Masecchia <salvatore.masecchia@disi.unige.it>
#
# License: BSD Style.

import warnings
from math import sqrt

import numpy as np

try:
    from scipy import linalg as la
except ImportError:
    from numpy import linalg as la

from .base import AbstractLinearModel
from .metrics import regression_error
from .cross_val import KFold

##############################################################################
# Algorithm

def l1l2_regularization(data, labels, mu, tau, beta=None, kmax=100000,
                        tolerance=1e-5, return_iterations=False,
                        adaptive=False):
    r"""Implementation of the Fast Iterative Shrinkage-Thresholding Algorithm
    to solve a least squares problem with `l1l2` penalty.

    It solves the `l1l2` regularization problem with parameter ``mu`` on the
    `l2-norm` and parameter ``tau`` on the `l1-norm`.

    Parameters
    ----------
    data : (N, P) ndarray
        Data matrix.
    labels : (N,) or (N, 1) ndarray
        Labels vector.
    mu : float
        `l2-norm` penalty.
    tau : float
        `l1-norm` penalty.
    beta : (P,) or (P, 1) ndarray, optional (default is `None`)
        Starting value for the iterations.
        If `None`, then iterations starts from the empty model.
    kmax : int, optional (default is `1e5`)
        Maximum number of iterations.
    tolerance : float, optional (default is `1e-5`)
        Convergence tolerance.
    return_iterations : bool, optional (default is `False`)
        If `True`, returns the number of iterations performed.
        The algorithm has a predefined minimum number of iterations
        equal to `10`.
    adaptive : bool, optional (default is `False`)
        If `True`, minimization is performed calculating an adaptive step size
        for each iteration.

    Returns
    -------
    beta : (P, 1) ndarray
        `l1l2` solution.
    k : int, optional
        Number of iterations performed.

    Examples
    --------
    >>> X = numpy.array([[0.1, 1.1, 0.3], [0.2, 1.2, 1.6], [0.3, 1.3, -0.6]])
    >>> beta = numpy.array([0.1, 0.1, 0.0])
    >>> Y = numpy.dot(X, beta)
    >>> beta = l1l2py.algorithms.l1l2_regularization(X, Y, 0.1, 0.1)
    >>> len(numpy.flatnonzero(beta))
    1

    """
    n, d = data.shape

    # beta starts from 0 and we assume also that the previous value is 0
    if beta is None:
        beta = np.zeros(d)
    else:
        beta = beta.ravel()

    # Useful quantities
    X = data
    Y = labels.ravel()

    if n > d:
        XTY = np.dot(X.T, Y)

    # First iteration with standard sigma
    sigma = _sigma(data, mu)
    if sigma < np.finfo(float).eps: # is zero...
        return np.zeros(d), 0

    mu_s = mu / sigma
    tau_s = tau / (2.0 * sigma)
    nsigma = n * sigma

    # Starting conditions
    auxcoef_ = beta
    t = 1.

    for k in xrange(kmax):
        # Pre-calculated "heavy" computation
        if n > d:
            precalc = XTY - np.dot(X.T, np.dot(X, auxcoef_))
        else:
            precalc = np.dot(X.T, Y - np.dot(X, auxcoef_))

        # TODO: stopping rule based on r = Y - Xbeta ??

        # Soft-Thresholding
        value = (precalc / nsigma) + ((1.0 - mu_s) * auxcoef_)
        beta_next = np.sign(value) * np.clip(np.abs(value) - tau_s, 0, np.inf)

        ######## Adaptive step size #######################################
        if adaptive:
            beta_diff = (auxcoef_ - beta_next)

            # Only if there is an increment of the solution
            # we can calculate the adaptive step-size
            if np.any(beta_diff):
                # grad_diff = np.dot(XTn, np.dot(X, beta_diff))
                # num = np.dot(beta_diff, grad_diff)
                tmp = np.dot(X, beta_diff) # <-- adaptive-step-size drawback
                num = np.dot(tmp, tmp) / n

                sigma = (num / np.dot(beta_diff, beta_diff))
                mu_s = mu / sigma
                tau_s = tau / (2.0*sigma)
                nsigma = n * sigma

                # Soft-Thresholding
                value = (precalc / nsigma) + ((1.0 - mu_s) * auxcoef_)
                beta_next = value * np.maximum(0, 1 - tau_s/np.abs(value))

        ######## FISTA ####################################################
        beta_diff = (beta_next - beta)
        t_next = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t*t))
        auxcoef_ = beta_next + ((t - 1.0)/t_next)*beta_diff

        # Convergence values
        max_diff = np.abs(beta_diff).max()
        max_coef = np.abs(beta_next).max()

        # Values update
        t = t_next
        beta = beta_next

        # Stopping rule (exit even if beta_next contains only zeros)
        if max_coef == 0.0 or (max_diff / max_coef) <= tolerance: break

    if return_iterations:
        return beta, k+1
    return beta


def _sigma(matrix, mu):
    n, p = matrix.shape

    if p > n:
        tmp = np.dot(matrix, matrix.T)
    else:
        tmp = np.dot(matrix.T, matrix)

    return (la.norm(tmp, 2)/n) + mu


##############################################################################
# Models

class ElasticNet(AbstractLinearModel):
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
    def __init__(self, fit_intercept=True, tau=0.5, mu=0.5,
                 adaptive_step_size=False, max_iter=10000, tol=1e-4,
                 precompute=False, normalize=False):

        self.tau = tau
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol
        self.adaptive_step_size = adaptive_step_size
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.normalize = normalize
        self.intercept_ = 0.0

    def _fit(self, X, y, warm_start=None):
        if warm_start is None:
            self.coef_ = np.zeros(X.shape[1])
        else:
            self.coef_ = np.asanyarray(warm_start)

        l1l2_proximal = l1l2_regularization
        self.coef_, self.niter_ = l1l2_proximal(X, y,
                                                self.mu, self.tau,
                                                beta=self.coef_,
                                                kmax=self.max_iter,
                                                tolerance=self.tol,
                                                return_iterations=True,
                                                adaptive=self.adaptive_step_size)

        if self.niter_ == self.max_iter:
            warnings.warn('Objective did not converge, you might want'
                          ' to increase the number of iterations')

        return self

class Lasso(ElasticNet):
    def __init__(self, tau=0.5, fit_intercept=True,
                 adaptive_step_size=False,
                 max_iter=10000, tol=1e-4, precompute=False, normalize=False):

        super(Lasso, self).__init__(tau=tau, mu=0.0,
                                    fit_intercept=fit_intercept,
                                    adaptive_step_size=adaptive_step_size,
                                    max_iter=max_iter,
                                    tol=tol, precompute=precompute,
                                    normalize=normalize)

##############################################################################
# Paths

#def lasso_path(X, y, eps=1e-3, n_taus=100, taus=None,
#              fit_intercept=True, verbose=False, **fit_params):
#    return enet_path(X, y, mu=0.0, eps=eps, n_taus=n_taus, taus=taus,
#                     fit_intercept=fit_intercept, verbose=verbose,
#                     adaptive_step_size=False, max_iter=10000, tol=1e-4)
#
#def enet_path(X, y, mu=0.5, eps=1e-3, n_taus=100, taus=None,
#              fit_intercept=True, verbose=False,
#              adaptive_step_size=False, max_iter=10000, tol=1e-4):
#    r"""The code is very similar to the scikits one.... mumble mumble"""
#
#    X, y, Xmean, ymean = LinearModel._center_data(X, y, fit_intercept)
#
#    n_samples = X.shape[0]
#    if taus is None:
#        tau_max = np.abs(np.dot(X.T, y)).max() * (2.0 / n_samples)
#        taus = np.logspace(np.log10(tau_max * eps), np.log10(tau_max),
#                           num=n_taus)[::-1]
#    else:
#        taus = np.sort(taus)[::-1]  # make sure taus are properly ordered
#    coef_ = None  # init coef_
#    models = []
#
#    for tau in taus:
#        model = ElasticNet(tau=tau, mu=mu, fit_intercept=False,
#                           adaptive_step_size=adaptive_step_size,
#                           max_iter=max_iter, tol=tol)
#        model.fit(X, y, coef_init=coef_)
#        if fit_intercept:
#            model.fit_intercept = True
#            model._set_intercept(Xmean, ymean)
#        if verbose:
#            print model
#        coef_ = model.coef_
#        models.append(model)
#
#    return models
#
################################################################################
## CV Estimators
#
#class ElasticNetCV(LinearModel):
#    path = staticmethod(enet_path)
#    estimator = ElasticNet
#
#    def __init__(self, mu=0.5, eps=1e-3, n_taus=100, taus=None,
#                 fit_intercept=True, max_iter=10000,
#                 tol=1e-4, cv=None,
#                 adaptive_step_size=False,
#                 loss=None):
#        self.mu = mu
#        self.eps = eps
#        self.n_taus = n_taus
#        self.taus = taus
#        self.fit_intercept = fit_intercept
#        self.max_iter = max_iter
#        self.tol = tol
#        self.cv = cv
#        self.adaptive_step_size = adaptive_step_size
#        self.loss = loss
#        self.coef_ = None
#
#    def fit(self, X, y):
#        X = np.asanyarray(X)
#        y = np.asanyarray(y)
#        n_samples = X.shape[0]
#
#        # Path parmeters creation
#        path_params = self.__dict__.copy()
#        for p in ('cv', 'loss', 'coef_'):
#            del path_params[p]
#
#        # TODO: optional????
#        # Start to compute path on full data
#        models = self.path(X, y, **path_params)
#
#        # Update the taus list
#        taus = [model.tau for model in models]
#        n_taus = len(taus)
#        path_params.update({'taus': taus, 'n_taus': n_taus})
#
#        # init cross-validation generator
#        cv = self.cv if self.cv else KFold(len(y), 5)
#
#        # init loss function
#        loss = self.loss if self.loss else regression_error
#
#        # Compute path for all folds and compute MSE to get the best tau
#        folds = list(cv)
#        loss_taus = np.zeros((len(folds), n_taus))
#        for i, (train, test) in enumerate(folds):
#            models_train = self.path(X[train], y[train], **path_params)
#            for i_tau, model in enumerate(models_train):
#                y_ = model.predict(X[test])
#                loss_taus[i, i_tau] += loss(y_, y[test])
#
#        i_best_tau = np.argmin(np.mean(loss_taus, axis=0))
#        model = models[i_best_tau]
#
#        self.coef_ = model.coef_
#        self.intercept_ = model.intercept_
#        self.tau = model.tau
#        self.taus = np.asarray(taus)
#        self.coef_path_ = np.asarray([model.coef_ for model in models])
#        self.loss_path = loss_taus.T
#        return self
#
#class LassoCV(ElasticNetCV):
#    path = staticmethod(lasso_path)
#    estimator = Lasso
#
#    def __init__(self, eps=1e-3, n_taus=100, taus=None,
#                 fit_intercept=True, max_iter=10000,
#                 tol=1e-4, cv=None,
#                 adaptive_step_size=False,
#                 loss=None):
#        super(LassoCV, self).__init__(mu=0.0,
#                                      eps=eps,
#                                      n_taus=n_taus, taus=taus,
#                                      fit_intercept=fit_intercept,
#                                      max_iter=max_iter,
#                                      tol=tol, cv=cv,
#                                      adaptive_step_size=adaptive_step_size,
#                                      loss=loss)


##############################################################################
# GLMNet models
#try:
#    from sklearn.linear_model import ElasticNet as _GlmElasticNet
#    from sklearn.linear_model import Lasso as _GlmLasso
#    from sklearn.linear_model import ElasticNetCV as _GlmElasticNetCV
#    from sklearn.linear_model import LassoCV as _GlmLassoCV
#
#    ## TODO better.....
#    class GlmElasticNet(ElasticNet):
#        def __init__(self, tau=0.5, mu=0.5, **params):
#            alpha = tau + mu
#            if tau == mu == 0.0:
#                rho = 0.0
#            else:
#                rho = tau / (tau + mu)
#
#            self.tau = tau
#            self.mu = mu
#            self._estimator = _GlmElasticNet(alpha=alpha, rho=rho, **params)
#
#        def fit(self, X, y):
#            return self._estimator.fit(X, y)
#
#        def __getattr__(self, key):
#            return getattr(self._estimator, key)
#
#except:
#    pass
