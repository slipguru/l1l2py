cimport numpy as np
import numpy as np
import numpy.linalg as la
cimport cython

cdef extern from "math.h":
    #double fabs(double f)
    double sqrt(double f)
    double log10(double f)
    
cdef extern from "cblas.h":
    void daxpy "cblas_daxpy"(int N, double alpha, double *X, int incX,
                             double *Y, int incY)
    void dgemv "cblas_dgemv" (char *order, char* TransA,
                              int M, int N, double alpha,
                              double *A, int lda, double *X,
                              int incX, double beta,
                              double *Y, int incY)

from itertools import izip
#import math

ctypedef np.float64_t DOUBLE

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def l1l2_proximal(np.ndarray[DOUBLE, ndim=2] X,
                  np.ndarray[DOUBLE, ndim=1] Y,
                  double mu, double tau,
                  np.ndarray[DOUBLE, ndim=1] beta,
                  int kmax,
                  double tolerance):

    # Useful quantities    
    cdef unsigned int n = X.shape[0]
    cdef unsigned int d = X.shape[1]
    cdef np.ndarray[DOUBLE, ndim=2] XTn = (X.T / n)
    
    cdef np.ndarray[DOUBLE, ndim=1] out
    cdef np.ndarray[DOUBLE, ndim=2] datanz
    cdef np.ndarray[DOUBLE, ndim=1] aux_beta = np.empty_like(beta)
    
    # Variables
    cdef double tau_max
    cdef unsigned int t_len
    cdef double sigma_0
    cdef double mu_s_0
    cdef double t
    cdef double sigma, mu_s, tau_s
    
    #cdef np.ndarray[DOUBLE, ndim=1] invariant
    #cdef np.ndarray[DOUBLE, ndim=1] taus
    #cdef np.ndarray[DOUBLE, ndim=1] aux_beta
    
    # DEBUG
    energy = [_functional(X, Y, beta, tau, mu)]
    
    # Tau list for continuation strategy
    # Pass it from input
    tau_max = l1_bound(X, Y) * 0.8
    if tau < tau_max:
        t_len =  int(tau_max / (tau_max - tau))
        t_len = int(10**log10(tau_max) - 10**log10(tau))
        taus = np.logspace(log10(tau), log10(tau_max), t_len)[::-1]
        tols = [1e-2]*t_len
        tols[-1] = tolerance
    else:
        taus = np.array([tau])
        tols = [tolerance]
    
    # Precalculated values
    sigma_0 = _sigma(X, mu)
    mu_s_0 = mu / sigma_0

    for tau, tolerance in izip(taus, tols):
        
        # Restart conditions
        aux_beta = beta # warm-start
        sigma = sigma_0
        mu_s = mu_s_0
        tau_s = tau / (2.0*sigma_0)
        t = 1.
           
        for k in xrange(kmax):        
            nonzero = np.flatnonzero(aux_beta)
            datanz = X[:,nonzero] # this is a BIG copy!
            aux_betanz = aux_beta[nonzero]
    
    
            #daxpy(n_samples, w_ii,
              #<DOUBLE*>(X.data + ii * n_samples * sizeof(DOUBLE)), 1,
              #<DOUBLE*>R.data, 1)
    
            # New solution
            #out = np.zeros_like(Y)
            #<DOUBLE*>aux_betanz.data
            #dgemv('C', 'N', n, len(nonzero),
                  #-1., <DOUBLE*>datanz.data, n,
                  #<DOUBLE*>aux_betanz.data, 1,
                  #1, <DOUBLE*>out.data, 1)
            
            invariant = np.dot(XTn, Y - np.dot(datanz, aux_betanz))
            value = invariant / sigma
            value[nonzero] += ((1.0 - mu_s) * aux_betanz)
                   
            # Soft-Thresholding
            beta_next = np.sign(value) * np.maximum(0, np.abs(value) - tau_s)
                   
            ######## Adaptive step size ########################################
            beta_diff2 = (aux_beta - beta_next)
            nonzero2 = np.flatnonzero(beta_diff2)
            Xnz = X[:,nonzero2] # this is a BIG copy!
            beta_diff2nz = beta_diff2[nonzero2]
            
            grad_diff = np.dot(XTn, np.dot(Xnz, beta_diff2nz))
            
            num = np.dot(beta_diff2.ravel(), grad_diff.ravel())
            den = np.sqrt(np.dot(beta_diff2nz, beta_diff2nz)) ### norm                
            sigma = num / (den**2)
            
            mu_s = mu/sigma
            tau_s = tau / (2.0*sigma)
        
            ############ AGAIN!!! ############################    
            # New solution
            value = invariant / sigma
            value[nonzero] += ((1.0 - mu_s) * aux_betanz)
            #value += ((1.0 - mu_s) * aux_betanz)
                   
            #beta = _soft_thresholding(value, tau_s)        
            beta_next = np.sign(value) * np.maximum(0, np.abs(value) - tau_s)
            ######## Adaptive step size ########################################
            
            energy.append(_functional(X, Y, beta_next, tau, mu))
           
            # New auxiliary beta (FISTA)
            beta_diff = (beta_next - beta)
            t_next = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t*t))
            aux_beta = beta_next + ((t - 1.0)/t_next)*beta_diff
            
            # Convergence values        
            max_diff = np.abs(beta_diff).max()
            max_coef = np.abs(beta_next).max()
            tol = tolerance
            
            # Stopping rule
            if (max_diff / max_coef) <= tol:
                break
    
            # Values update
            t = t_next
            beta = beta_next

    return beta, k, energy    

def _functional(X, Y, beta, tau, mu):
    n = X.shape[0]

    loss = Y - np.dot(X, beta)
    loss_quadratic_norm = np.linalg.norm(loss) ** 2
    beta_quadratic_norm = np.linalg.norm(beta) ** 2
    beta_l1_norm = np.abs(beta).sum()

    return (((1./n) * loss_quadratic_norm)
             + mu * beta_quadratic_norm
             + tau * beta_l1_norm)

def _sigma(matrix, mu):
    n, p = matrix.shape

    if p > n:
        tmp = np.dot(matrix, matrix.T)
    else:
        tmp = np.dot(matrix.T, matrix)

    return (la.norm(tmp, 2)/n) + mu
    
def l1_bound(data, labels):
    n = data.shape[0]
    corr = np.abs(np.dot(data.T, labels))

    tau_max = (corr.max() * (2.0/n))

    return tau_max