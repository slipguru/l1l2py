# Author: Salvatore Masecchia <salvatore.masecchia@disi.unige.it>
#
# License: BSD Style.

import warnings
import numpy as np

try:
    from scipy import linalg as la
except ImportError:
    from numpy import linalg as la

from .proximal import l1l2_regularization
from .metrics import regression_error
from .cross_val import KFold

##############################################################################
# Models

class LinearModel(object):
    """TODO"""

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Returns predicted values.
        """
        X = np.asanyarray(X)
        return np.dot(X, self.coef_.T) + self.intercept_

    @staticmethod
    def _center_data(X, y, fit_intercept):
        """
        Centers data to have mean zero along axis 0. This is here because
        nearly all linear models will want their data to be centered.
        """
        if fit_intercept:
            Xmean = X.mean(axis=0)
            X = X - Xmean
            
            ymean = y.mean()
            y = y - ymean
        else:
            Xmean = np.zeros(X.shape[1])
            ymean = 0.
        return X, y, Xmean, ymean

    def _set_intercept(self, Xmean, ymean):
        """Set the intercept_
        """
        if self.fit_intercept:
            self.intercept_ = ymean - np.dot(Xmean, self.coef_.T)
        else:
            self.intercept_ = 0

class Ridge(LinearModel):
    """TODO"""

    def __init__(self, mu=0.5, fit_intercept=True): ## DEFINE better defaults
        self.mu = mu
        self.fit_intercept = fit_intercept
        self.coef_ = None

    def fit(self, X, y):
        X = np.asanyarray(X)
        y = np.asanyarray(y)
        n_samples, n_features = X.shape

        X, y, Xmean, ymean = Ridge._center_data(X, y, self.fit_intercept)

        if n_samples < n_features:
            tmp = np.dot(X, X.T)
            if self.mu != 0.0:
                tmp += self.mu*n_samples*np.eye(n_samples)
            tmp = la.pinv(tmp)

            self.coef_ =  np.dot(np.dot(X.T, tmp), y)
        else:
            tmp = np.dot(X.T, X)
            if self.mu != 0.0:
                tmp += self.mu*n_samples*np.eye(n_features)
            tmp = la.pinv(tmp)

            self.coef_ = np.dot(tmp, np.dot(X.T, y))

        self._set_intercept(Xmean, ymean)

        return self

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
                 adaptive_step_size=False, max_iter=10000, tol=1e-4):
        self.tau = tau
        self.mu = mu
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.adaptive_step_size = adaptive_step_size
        self.coef_ = None

    def fit(self, X, y, coef_init=None):
        X = np.asanyarray(X)
        y = np.asanyarray(y)

        X, y, Xmean, ymean = LinearModel._center_data(X, y, self.fit_intercept)

        if coef_init is None:
            self.coef_ = np.zeros(X.shape[1])
        else:
            self.coef_ = np.asanyarray(coef_init)

        l1l2_proximal = l1l2_regularization
        self.coef_, self.niter_ = l1l2_proximal(X, y,
                                                self.mu, self.tau,
                                                beta=self.coef_,
                                                kmax=self.max_iter,
                                                tolerance=self.tol,
                                                return_iterations=True,
                                                adaptive=self.adaptive_step_size)
        self._set_intercept(Xmean, ymean)

        if self.niter_ == self.max_iter:
            warnings.warn('Objective did not converge, you might want'
                          ' to increase the number of iterations')

        return self

class Lasso(ElasticNet):
    def __init__(self, tau=0.5, fit_intercept=True,
                 adaptive_step_size=False, max_iter=10000, tol=1e-4):

        super(Lasso, self).__init__(tau=tau, mu=0.0,
                                    fit_intercept=fit_intercept,
                                    adaptive_step_size=adaptive_step_size,
                                    max_iter=max_iter,
                                    tol=tol)

##############################################################################
# Paths

def lasso_path(X, y, eps=1e-3, n_taus=100, taus=None,
              fit_intercept=True, verbose=False, **fit_params):
    return enet_path(X, y, mu=0.0, eps=eps, n_taus=n_taus, taus=taus,
                     fit_intercept=fit_intercept, verbose=verbose,
                     adaptive_step_size=False, max_iter=10000, tol=1e-4)

def enet_path(X, y, mu=0.5, eps=1e-3, n_taus=100, taus=None,
              fit_intercept=True, verbose=False,
              adaptive_step_size=False, max_iter=10000, tol=1e-4):
    r"""The code is very similar to the scikits one.... mumble mumble"""

    X, y, Xmean, ymean = LinearModel._center_data(X, y, fit_intercept)

    n_samples = X.shape[0]
    if taus is None:
        tau_max = np.abs(np.dot(X.T, y)).max() * (2.0 / n_samples)
        taus = np.logspace(np.log10(tau_max * eps), np.log10(tau_max),
                           num=n_taus)[::-1]
    else:
        taus = np.sort(taus)[::-1]  # make sure taus are properly ordered
    coef_ = None  # init coef_
    models = []

    for tau in taus:
        model = ElasticNet(tau=tau, mu=mu, fit_intercept=False,
                           adaptive_step_size=adaptive_step_size,
                           max_iter=max_iter, tol=tol)
        model.fit(X, y, coef_init=coef_)
        if fit_intercept:
            model.fit_intercept = True
            model._set_intercept(Xmean, ymean)
        if verbose:
            print model
        coef_ = model.coef_
        models.append(model)

    return models

###############################################################################
# CV Estimators

class ElasticNetCV(LinearModel):
    path = staticmethod(enet_path)
    estimator = ElasticNet

    def __init__(self, mu=0.5, eps=1e-3, n_taus=100, taus=None,
                 fit_intercept=True, max_iter=10000,
                 tol=1e-4, cv=None,
                 adaptive_step_size=False,
                 loss=None):
        self.mu = mu
        self.eps = eps
        self.n_taus = n_taus
        self.taus = taus
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.adaptive_step_size = adaptive_step_size
        self.loss = loss
        self.coef_ = None

    def fit(self, X, y):
        X = np.asanyarray(X)
        y = np.asanyarray(y)
        n_samples = X.shape[0]

        # Path parmeters creation
        path_params = self.__dict__.copy()
        for p in ('cv', 'loss', 'coef_'):
            del path_params[p]

        # TODO: optional????
        # Start to compute path on full data
        models = self.path(X, y, **path_params)

        # Update the taus list
        taus = [model.tau for model in models]
        n_taus = len(taus)
        path_params.update({'taus': taus, 'n_taus': n_taus})

        # init cross-validation generator
        cv = self.cv if self.cv else KFold(len(y), 5)

        # init loss function
        loss = self.loss if self.loss else regression_error

        # Compute path for all folds and compute MSE to get the best tau
        folds = list(cv)
        loss_taus = np.zeros((len(folds), n_taus))
        for i, (train, test) in enumerate(folds):
            models_train = self.path(X[train], y[train], **path_params)
            for i_tau, model in enumerate(models_train):
                y_ = model.predict(X[test])
                loss_taus[i, i_tau] += loss(y_, y[test])

        i_best_tau = np.argmin(np.mean(loss_taus, axis=0))
        model = models[i_best_tau]

        self.coef_ = model.coef_
        self.intercept_ = model.intercept_
        self.tau = model.tau
        self.taus = np.asarray(taus)
        self.coef_path_ = np.asarray([model.coef_ for model in models])
        self.loss_path = loss_taus.T
        return self

class LassoCV(ElasticNetCV):
    path = staticmethod(lasso_path)
    estimator = Lasso

    def __init__(self, eps=1e-3, n_taus=100, taus=None,
                 fit_intercept=True, max_iter=10000,
                 tol=1e-4, cv=None,
                 adaptive_step_size=False,
                 loss=None):
        super(LassoCV, self).__init__(mu=0.0,
                                      eps=eps,
                                      n_taus=n_taus, taus=taus,
                                      fit_intercept=fit_intercept,
                                      max_iter=max_iter,
                                      tol=tol, cv=cv,
                                      adaptive_step_size=adaptive_step_size,
                                      loss=loss)


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
