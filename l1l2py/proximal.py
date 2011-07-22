# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#
# License: BSD Style.

import warnings
import numpy as np

from scikits.learn.linear_model.base import LinearModel

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
                 adaptive_step_size=True, max_iter=10000, tol=1e-5):
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
            self.coef_ = np.zeros(X.shape[1], dtype=np.float64)
        else:
            self.coef_ = np.asanyarray(coef_init)

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
                 adaptive_step_size=True, max_iter=10000, tol=1e-5):

        super(Lasso, self).__init__(tau=tau, mu=0.0,
                                    fit_intercept=fit_intercept,
                                    adaptive_step_size=adaptive_step_size, 
                                    max_iter=max_iter,
                                    tol=tol)

def elasticnet_path():
    pass
    

class ElasticNetCV(LinearModel):
    def __init__(self, taus, mu, cv=None, fit_intercept=True,
                 adaptive_step_size=True, max_iter=10000, tol=1e-5):
        self.taus = taus
        self.mu = mu
        self.cv = cv
        self.fit_intercept=fit_intercept
        self.adaptive_step_size = adaptive_step_size
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        
    def fit(self, X, y, **fit_params):
        pass
        
    @staticmethod
    def path(X, y, taus, mu, fit_intercept=False, **fit_params):
        r"""TODO: rewrite using Lasso/ElasticNet classes"""
        from collections import deque
        from .algorithms import ridge_regression, l1l2_regularization
        from scikits.learn.linear_model import LinearRegression
        from scikits.learn.base import clone
        
        X = np.asanyarray(X)
        y = np.asanyarray(y)
        n, p = X.shape
        
        X, y, Xmean, ymean = LinearModel._center_data(X, y, fit_intercept)
    
        if mu == 0.0:
            model_ls = LinearRegression().fit(X, y)
            if fit_intercept:
                model_ls.fit_intercept = True
                model_ls._set_intercept(Xmean, ymean)
        beta = None #init
    
        out = deque()
        nonzero = 0
        for tau in reversed(taus):
            if mu == 0.0 and nonzero >= n: # lasso saturation
                model_next = clone(model_ls)
                beta_next = model_next.coef_
            else:
                model_next = ElasticNet(tau=tau, mu=mu, fit_intercept=False)
                model_next.fit(X, y, coef_init=beta, **fit_params)
                if fit_intercept:
                    model_next.fit_intercept = True
                    model_next._set_intercept(Xmean, ymean)
                
                beta_next = model_next.coef_
                k = model_next.niter_
    
            nonzero = len(np.flatnonzero(beta_next))
            if nonzero > 0:
                out.appendleft(model_next)
    
            beta = beta_next
    
        return out

class LassoCV(ElasticNetCV):
    pass