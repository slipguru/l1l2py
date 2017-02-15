from libc.math cimport fabs
cimport numpy as np
import numpy as np
import numpy.linalg as linalg

cimport cython
from cpython cimport bool
from cython cimport floating
import warnings

ctypedef np.float64_t DOUBLE
ctypedef np.uint32_t UINT32_t
ctypedef floating (*DOT)(int N, floating *X, int incX, floating *Y,
                         int incY) nogil
ctypedef void (*AXPY)(int N, floating alpha, floating *X, int incX,
                      floating *Y, int incY) nogil
ctypedef floating (*ASUM)(int N, floating *X, int incX) nogil

np.import_array()

# from l1l2py.algorithms import l1l2_regularization
try:
    from scipy import linalg as la
except ImportError:
    from numpy import linalg as la
# from l1l2py.algorithms import ridge_regression


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef floating get_lipschitz(np.ndarray[floating, ndim=2, mode='fortran'] X):
    """Get the Lipschitz constant for a specific loss function.

    Only square loss implemented.

    Parameters
    ----------
    data : (n, d) float ndarray
        data matrix
    loss : string
        the selected loss function in {'square', 'logit'}
    Returns
    ----------
    L : float
        the Lipschitz constant
    """

    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int n_features = X.shape[1]

    if n_features > n_samples:
        tmp = np.dot(X, X.T)
    else:
        tmp = np.dot(X.T, X)
    return la.norm(tmp, 2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[floating, ndim=1] least_square_step(
                      np.ndarray[floating, ndim=1, mode='c'] y,
                      np.ndarray[floating, ndim=2, mode='fortran'] X,
                      np.ndarray[floating, ndim=1] Z):
    """
    Returns the point in witch we apply gradient descent
    parameters
    ----------
    y : np-array
        the labels vector
    K : 2D np-array
        the concatenation of all the kernels, of shape
        n_samples, n_kernels*n_samples
    Z : a linear combination of the last two coefficient vectors
    returns
    -------
    res : np-array of shape n_samples*,_kernels
          a point of the space where we will apply gradient descent
    """
    return np.dot(X.transpose(), y - np.dot(X, Z))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[floating, ndim=1] prox_l1(np.ndarray[floating, ndim=1] w, floating alpha):
    r"""Proximity operator for l1 norm.

    :math:`\\hat{\\alpha}_{l,m} = sign(u_{l,m})\\left||u_{l,m}| - \\alpha \\right|_+`
    Parameters
    ----------
    u : ndarray
        The vector (of the n-dimensional space) on witch we want
        to compute the proximal operator
    alpha : float
        regularisation parameter
    Returns
    -------
    ndarray : the vector corresponding to the application of the
             proximity operator to u
    """
    return np.sign(w) * np.maximum(np.abs(w) - alpha, 0.)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def fista_l1l2(np.ndarray[floating, ndim=1] beta, floating tau, floating mu,
               np.ndarray[floating, ndim=2, mode='fortran'] X,
               np.ndarray[floating, ndim=1, mode='c'] y,
               int max_iter, floating tol, object rng, bint random=0,
               bint positive=0, bint adaptive=0):
    """Fista algorithm for l1l2 regularization.

    We minimize
    (1/n) * norm(y - X w, 2)^2 + tau norm(w, 1) + mu norm(w, 2)^2
    """
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int n_features = X.shape[1]

    # XTY = np.dot(Xt, y)
    # XTX = np.dot(Xt, X)
    # if n_samples > n_features:
    #     XTY = np.dot(Xt, y)

    # First iteration with standard sigma
    cdef floating lipschitz_constant = get_lipschitz(X)
    cdef floating sigma = lipschitz_constant / n_samples + mu

    if sigma < np.finfo(float).eps:  # is zero...
        return beta, None, tol, 0

    # mu_s = 1 - mu / sigma
    cdef floating mu_s = 1 - mu * n_samples / (lipschitz_constant + mu * n_samples)
    # tau_s = tau / (2.0 * sigma)
    cdef floating tau_s = tau * n_samples / (2. * lipschitz_constant + mu * n_samples)
    # nsigma = n_samples * sigma
    cdef floating gamma = 1. / (lipschitz_constant + mu * n_samples)

    # Starting conditions
    cdef np.ndarray[floating, ndim=1] aux_beta = np.copy(beta)
    cdef np.ndarray[floating, ndim=1] grad, value, beta_diff
    cdef np.ndarray[floating, ndim=1] beta_next = np.empty(n_features)
    cdef floating t = 1., t_next, max_coef, max_diff

    for n_iter in range(max_iter):
        # Pre-calculated "heavy" computation
        # if n_samples > n_features:
        #     grad = XTY - np.dot(Xt, np.dot(X, aux_beta))
        # else:
        #     grad = np.dot(Xt, y - np.dot(X, aux_beta))
        # grad = XTY - np.dot(Xt, np.dot(X, aux_beta))
        grad = least_square_step(y, X, aux_beta)

        # Soft-Thresholding
        # value = (precalc / nsigma) + (mu_s * aux_beta)
        value = gamma * grad + (mu_s * aux_beta)
        beta_next = prox_l1(value, tau_s)
        # np.maximum(np.abs(value) - tau_s, 0, beta_next)
        # beta_next *= np.sign(value)

        # ## Adaptive step size #######################################
        # if adaptive:
        #     beta_diff = (aux_beta - beta_next)
        #
        #     # Only if there is an increment of the solution
        #     # we can calculate the adaptive step-size
        #     if np.any(beta_diff):
        #         # grad_diff = np.dot(XTn, np.dot(X, beta_diff))
        #         # num = np.dot(beta_diff, grad_diff)
        #         tmp = np.dot(X, beta_diff)  # <-- adaptive-step-size drawback
        #         num = np.dot(tmp, tmp) / n_samples
        #
        #         sigma = (num / np.dot(beta_diff, beta_diff))
        #         mu_s = 1 - mu / sigma
        #         tau_s = tau / (2. * sigma)
        #         nsigma = n_samples * sigma
        #
        #         # Soft-Thresholding
        #         value = grad / nsigma + mu_s * aux_beta
        #         beta_next = prox_l1(value, tau_s)
        #         # np.maximum(np.abs(value) - tau_s, 0, beta_next)
        #         # beta_next *= np.sign(value)

        # FISTA
        beta_diff = (beta_next - beta)
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        aux_beta = beta_next + ((t - 1) / t_next) * beta_diff

        # Convergence values
        max_diff = np.abs(beta_diff).max()
        max_coef = np.abs(beta_next).max()

        # Values update
        t = t_next
        # beta = np.copy(beta_next)
        beta = beta_next

        # Stopping rule (exit even if beta_next contains only zeros)
        if max_coef == 0.0 or (max_diff / max_coef) <= tol:
            break

    return beta, None, tol, n_iter + 1
