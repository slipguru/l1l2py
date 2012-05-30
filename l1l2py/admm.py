import numpy as np
from scipy import linalg as la

from .base import AbstractLinearModel

class Lasso(AbstractLinearModel):
    def __init__(self, fit_intercept=True, tau=0.5,
                 rho=1.0, alpha=1.0,
                 max_iter=1000, abs_tol=1e-6, rel_tol=1e-4):
        super(Lasso, self).__init__(fit_intercept)
        
        self.tau = tau
        self.rho = rho      # step size
        self.alpha = alpha  # over relaxation parameter
        self.max_iter = max_iter
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol          
        
    def _train(self, X, y):
        n, d = X.shape
        
        XTy = np.dot(X.T, y)
        
        x = np.zeros(d)
        z = np.zeros(d)
        u = np.zeros(d)
        
        L, U = factor(X, self.rho)
        
        for k in xrange(self.max_iter):
            #x-update
            q = (2./n) * XTy + self.rho * (z - u)    # temporary value
            
            if n >= d:      # if skinny                
                x = la.solve_triangular(U,
                                       la.solve_triangular(L, q, lower=True),
                                       lower=False)
            else:            # if fat
                tmp = la.solve_triangular(U,
                                          la.solve_triangular(L, np.dot(X, q),
                                                              lower=True),
                                          lower=False)
                x = (q / self.rho) - ( np.dot(X.T, tmp) / (self.rho * self.rho))
        
            # z-update with relaxation
            zold = z
            x_hat = self.alpha * x + (1 - self.alpha) * zold            
            value = x_hat + u            
            z = np.sign(value) * np.clip(np.abs(value) - self.tau/self.rho,
                                         0, np.inf)
            #z = max( 0, value - self.tau/self.rho ) - max( 0, -value - self.tau/self.rho)
            
            # u-update
            u = u + (x_hat - z)
            
            # Stopping
            r_norm  = la.norm(x - z);
            s_norm  = la.norm(-self.rho * (z - zold))
            
            eps_pri = np.sqrt(d) * self.abs_tol + self.rel_tol * max(la.norm(x), la.norm(-z))
            eps_dual= np.sqrt(d) * self.abs_tol + self.rel_tol * la.norm(self.rho * u)
            
            if (r_norm < eps_pri) and (s_norm < eps_dual):
                self._beta = z
                break
        else:
            # anyway
            self._beta = z

def factor(X, rho):
    n, d = X.shape
    
    if n >= d:
        L = la.cholesky((2./n) * np.dot(X.T, X) + rho * np.eye(d), lower=True)
    else:
        L = la.cholesky(np.eye(n) + 1./rho * np.dot(X, X.T), lower=True)
        
    return L, L.T # L, U