# Author: Salvatore Masecchia <salvatore.masecchia@disi.unige.it>
#
# License: BSD Style.

import warnings
import numpy as np

from scikits.learn.base import clone
from scikits.learn.linear_model.base import LinearModel
from scikits.learn.linear_model import LinearRegression
from scikits.learn.cross_val import KFold

from .algorithms import l1l2_regularization, l1l2_path

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
                 adaptive_step_size=False, max_iter=100000, tol=1e-5):
        self.tau = tau
        self.mu = mu
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.adaptive_step_size = adaptive_step_size
        self.coef_ = None

    def fit(self, X, y, coef_init=None, **fit_params):
        self._set_params(**fit_params)
        X = np.asanyarray(X)
        y = np.asanyarray(y)

        X, y, Xmean, ymean = LinearModel._center_data(X, y, self.fit_intercept)

        if coef_init is None:
            self.coef_ = np.zeros(X.shape[1])
        else:
            self.coef_ = np.asanyarray(coef_init)

        #from l1l2py.algorithms_orig import l1l2_regularization
        l1l2_proximal = l1l2_regularization
        self.coef_, self.niter_ = l1l2_proximal(X, y,
                                                self.mu, self.tau,
                                                beta=self.coef_,
                                                kmax=self.max_iter,
                                                tolerance=self.tol,
                                                adaptive=self.adaptive_step_size)
        self._set_intercept(Xmean, ymean)

        if self.niter_ == self.max_iter:
            warnings.warn('Objective did not converge, you might want'
                          ' to increase the number of iterations')

        return self

class Lasso(ElasticNet):
    def __init__(self, tau=0.5, fit_intercept=True,
                 adaptive_step_size=True, max_iter=100000, tol=1e-5):

        super(Lasso, self).__init__(tau=tau, mu=0.0,
                                    fit_intercept=fit_intercept,
                                    adaptive_step_size=adaptive_step_size,
                                    max_iter=max_iter,
                                    tol=tol)

##############################################################################
def lasso_path(X, y, eps=1e-3, n_taus=100, taus=None,
              fit_intercept=True, verbose=False, **fit_params):
    return enet_path(X, y, mu=0.0, eps=eps, n_alphas=n_alphas, alphas=alphas,
                     fit_intercept=fit_intercept, verbose=verbose, **fit_params)

def enet_path(X, y, mu=0.5, eps=1e-3, n_taus=100, taus=None,
              fit_intercept=True, verbose=False, **fit_params):
    r"""The code is very similar to the scikits one.... mumble mumble"""

    X, y, Xmean, ymean = LinearModel._center_data(X, y, fit_intercept)

    n_samples = X.shape[0]
    if taus is None:
        tau_max = np.abs(np.dot(X.T, y)).max() * (2.0 / n_samples)
        taus = np.logspace(np.log10(tau_max * eps), np.log10(tau_max),
                           num=n_taus)[::-1]
    else:
        taus = np.sort(taus)[::-1]  # make sure alphas are properly ordered

    coef_ = None  # init coef_
    models = []

    for tau in taus:
        model = ElasticNet(tau=tau, mu=mu, fit_intercept=False)
        model.fit(X, y, coef_init=coef_, **fit_params)
        if fit_intercept:
            model.fit_intercept = True
            model._set_intercept(Xmean, ymean)
        if verbose:
            print model
        coef_ = model.coef_
        models.append(model)

    return models



###############################################################################
#TODO

class ElasticNetCV(LinearModel):
    path = staticmethod(enet_path)
    estimator = ElasticNet

    def __init__(self, mu=0.5, eps=1e-3, n_taus=100, taus=None,
                 fit_intercept=True, max_iter=100000, 
                 tol=1e-5, cv=None, adaptive_step_size=True):
        self.mu = mu
        self.eps = eps
        self.n_taus = n_taus
        self.taus = taus
        self.fit_intercept=fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.adaptive_step_size = adaptive_step_size
        self.coef_ = None        

    def fit(self, X, y, **fit_params):
        self._set_params(**fit_params)
        X = np.asanyarray(X)
        y = np.asanyarray(y)
        
        n_samples = X.shape[0]
        
        ### TODO: this should be optional....
        # Start to compute path on full data
        
        # All LinearModelCV parameters except 'cv' are acceptable
        path_params = self._get_params()
        del path_params['cv']     
        models = self.path(X, y, **path_params)
        
        # But I have to calculate the right set of taus
        taus = [model.tau for model in models]
        n_taus = len(taus)
        path_params.update({'taus':taus, 'n_taus':n_taus})
    
        # This is a cut and paste from enet path ##############################    
        #n_samples = X.shape[0]
        #if taus is None:
        #    tau_max = np.abs(np.dot(X.T, y)).max() * (2.0 / n_samples)
        #    taus = np.logspace(np.log10(tau_max * eps), np.log10(tau_max),
        #                       num=n_taus)[::-1]
        #else:
        #    taus = np.sort(taus)[::-1]  # make sure alphas are properly ordered
        #n_taus = len(taus)
        #######################################################################

        # init cross-validation generator
        cv = self.cv if self.cv else KFold(n_samples, 5)
       
        # Optional parameter for the error function

        # Compute path for all folds and compute MSE to get the best alpha
        folds = list(cv)
        mse_taus = np.zeros((len(folds), n_taus))
        for i, (train, test) in enumerate(folds):
            models_train = self.path(X[train], y[train], **path_params)
            for i_tau, model in enumerate(models_train):
                y_ = model.predict(X[test])
                mse_taus[i, i_tau] += ((y_ - y[test]) ** 2).mean()

        i_best_tau = np.argmin(np.mean(mse_taus, axis=0))
        model = models[i_best_tau]
        
        ## TODO: now, if the full path is not required I have to fit
        ## the model with the best tau and set coef_ etc...

        self.coef_ = model.coef_
        self.intercept_ = model.intercept_
        self.tau = model.tau
        self.taus = np.asarray(taus)
        self.coef_path_ = np.asarray([model.coef_ for model in models])
        self.mse_path_ = mse_taus.T
        return self

class LassoCV(ElasticNetCV):
    path = staticmethod(lasso_path)
    estimator = Lasso
