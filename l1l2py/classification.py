"""Wrapper for l1l2."""
import numpy as np

from sklearn.linear_model import ElasticNet
from sklearn.linear_model.base import LinearClassifierMixin, _pre_fit
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.feature_selection.base import SelectorMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_X_y, deprecated
from sklearn.feature_selection.from_model import _get_feature_importances
from sklearn.feature_selection.from_model import _calculate_threshold

# from l1l2py.algorithms import l1l2_regularization
try:
    from scipy import linalg as la
except ImportError:
    from numpy import linalg as la
# from l1l2py.algorithms import ridge_regression


def get_lipschitz(data):
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
    n, p = data.shape

    if p > n:
        tmp = np.dot(data, data.T)
    else:
        tmp = np.dot(data.T, data)
    return la.norm(tmp, 2)


def least_square_step(y, X, Z):
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


def prox_l1(w, alpha):
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


def fista_l1l2(beta, tau, mu, X, y, max_iter, tol, rng, random, positive,
               adaptive=False):
    """Fista algorithm for l1l2 regularization.

    We minimize
    (1/n) * norm(y - X w, 2)^2 + tau norm(w, 1) + mu norm(w, 2)^2
    """
    n_samples = y.shape[0]
    n_features = beta.shape[0]

    # XTY = np.dot(Xt, y)
    # XTX = np.dot(Xt, X)
    # if n_samples > n_features:
    #     XTY = np.dot(Xt, y)

    # First iteration with standard sigma
    lipschitz_constant = get_lipschitz(X)
    sigma = lipschitz_constant / n_samples + mu

    if sigma < np.finfo(float).eps:  # is zero...
        return beta, None, tol, 0

    # mu_s = 1 - mu / sigma
    mu_s = 1 - mu * n_samples / (lipschitz_constant + mu * n_samples)
    # tau_s = tau / (2.0 * sigma)
    tau_s = tau * n_samples / (2. * lipschitz_constant + mu * n_samples)
    # nsigma = n_samples * sigma
    gamma = 1. / (lipschitz_constant + mu * n_samples)

    # Starting conditions
    aux_beta = np.copy(beta)
    beta_next = np.empty(n_features)
    t = 1.

    for n_iter in xrange(max_iter):
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
        if adaptive:
            beta_diff = (aux_beta - beta_next)

            # Only if there is an increment of the solution
            # we can calculate the adaptive step-size
            if np.any(beta_diff):
                # grad_diff = np.dot(XTn, np.dot(X, beta_diff))
                # num = np.dot(beta_diff, grad_diff)
                tmp = np.dot(X, beta_diff)  # <-- adaptive-step-size drawback
                num = np.dot(tmp, tmp) / n_samples

                sigma = (num / np.dot(beta_diff, beta_diff))
                mu_s = 1 - mu / sigma
                tau_s = tau / (2. * sigma)
                nsigma = n_samples * sigma

                # Soft-Thresholding
                value = grad / nsigma + mu_s * aux_beta
                beta_next = prox_l1(value, tau_s)
                # np.maximum(np.abs(value) - tau_s, 0, beta_next)
                # beta_next *= np.sign(value)

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


def l1l2_regularization(
    X, y, max_iter=100000, l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None,
    precompute='auto', Xy=None, copy_X=True, coef_init=None,
    verbose=False, return_n_iter=False, positive=False,
        tol=1e-5, check_input=True, **params):
    if check_input:
        X = check_array(X, 'csc', dtype=[np.float64, np.float32],
                        order='F', copy=copy_X)
        y = check_array(y, 'csc', dtype=X.dtype.type, order='F', copy=False,
                        ensure_2d=False)
        if Xy is not None:
            # Xy should be a 1d contiguous array or a 2D C ordered array
            Xy = check_array(Xy, dtype=X.dtype.type, order='C', copy=False,
                             ensure_2d=False)

    n_samples, n_features = X.shape

    multi_output = False
    if y.ndim != 1:
        multi_output = True
        _, n_outputs = y.shape

    # MultiTaskElasticNet does not support sparse matrices
    from scipy import sparse
    if not multi_output and sparse.isspmatrix(X):
        if 'X_offset' in params:
            # As sparse matrices are not actually centered we need this
            # to be passed to the CD solver.
            X_sparse_scaling = params['X_offset'] / params['X_scale']
            X_sparse_scaling = np.asarray(X_sparse_scaling, dtype=X.dtype)
        else:
            X_sparse_scaling = np.zeros(n_features, dtype=X.dtype)

    # X should be normalized and fit already if function is called
    # from ElasticNet.fit
    if check_input:
        X, y, X_offset, y_offset, X_scale, precompute, Xy = \
            _pre_fit(X, y, Xy, precompute, normalize=False,
                     fit_intercept=False, copy=False)
    if alphas is None:
        # No need to normalize of fit_intercept: it has been done above
        alphas = _alpha_grid(X, y, Xy=Xy, l1_ratio=l1_ratio,
                             fit_intercept=False, eps=eps, n_alphas=n_alphas,
                             normalize=False, copy_X=False)
    else:
        alphas = np.sort(alphas)[::-1]  # make sure alphas are properly ordered

    n_alphas = len(alphas)
    tol = params.get('tol', 1e-4)
    max_iter = params.get('max_iter', 1000)
    dual_gaps = np.empty(n_alphas)
    n_iters = []

    rng = check_random_state(params.get('random_state', None))
    selection = params.get('selection', 'cyclic')
    if selection not in ['random', 'cyclic']:
        raise ValueError("selection should be either random or cyclic.")
    random = (selection == 'random')

    if not multi_output:
        coefs = np.empty((n_features, n_alphas), dtype=X.dtype)
    else:
        coefs = np.empty((n_outputs, n_features, n_alphas),
                         dtype=X.dtype)

    if coef_init is None:
        coef_ = np.asfortranarray(np.zeros(coefs.shape[:-1], dtype=X.dtype))
    else:
        coef_ = np.asfortranarray(coef_init, dtype=X.dtype)

    for i, alpha in enumerate(alphas):
        l1_reg = alpha * l1_ratio * 2  # * n_samples
        l2_reg = alpha * (1.0 - l1_ratio)  # * n_samples
        if not multi_output and sparse.isspmatrix(X):
            # model = cd_fast.sparse_enet_coordinate_descent(
            #     coef_, l1_reg, l2_reg, X.data, X.indices,
            #     X.indptr, y, X_sparse_scaling,
            #     max_iter, tol, rng, random, positive)
            raise NotImplementedError()
        elif multi_output:
            # model = cd_fast.enet_coordinate_descent_multi_task(
            #     coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random)
            raise NotImplementedError('Multi output not implemented')
        elif isinstance(precompute, np.ndarray):
            # We expect precompute to be already Fortran ordered when bypassing
            # checks
            if check_input:
                precompute = check_array(precompute, dtype=np.float64,
                                         order='C')
            # model = cd_fast.enet_coordinate_descent_gram(
            #     coef_, l1_reg, l2_reg, precompute, Xy, y, max_iter,
            #     tol, rng, random, positive)
            raise NotImplementedError()

        elif precompute is False:
            # model = cd_fast.enet_coordinate_descent(
            #     coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random,
            #     positive)
            model = fista_l1l2(
                coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random,
                positive)
        else:
            raise ValueError("Precompute should be one of True, False, "
                             "'auto' or array-like. Got %r" % precompute)
        coef_, dual_gap_, eps_, n_iter_ = model
        coefs[..., i] = coef_
        dual_gaps[i] = dual_gap_
        n_iters.append(n_iter_)
        if dual_gap_ > eps_:
            import warnings
            warnings.warn('Objective did not converge.' +
                          ' You might want' +
                          ' to increase the number of iterations.' +
                          ' Fitting data with very small alpha' +
                          ' may cause precision problems.',
                          ConvergenceWarning)

        if verbose:
            if verbose > 2:
                print(model)
            elif verbose > 1:
                print('Path: %03i out of %03i' % (i, n_alphas))
            else:
                import sys
                sys.stderr.write('.')

    if return_n_iter:
        return alphas, coefs, dual_gaps, n_iters
    return alphas, coefs, dual_gaps


class L1L2Classifier(LinearClassifierMixin, SelectorMixin, ElasticNet):
    # def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
    #              normalize=False, precompute=False, max_iter=1000,
    #              copy_X=True, tol=1e-4, warm_start=False, positive=False,
    #              random_state=None, selection='cyclic'):
    path = staticmethod(l1l2_regularization)

    def __init__(self, mu=.5, tau=1.0, use_gpu=False, threshold=1e-16,
                 alpha=None, l1_ratio=None, fit_intercept=True,
                 normalize=False, precompute=False, max_iter=10000,
                 copy_X=True, tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        self.mu = mu
        self.tau = tau
        self.use_gpu = use_gpu
        self.threshold = threshold  # threshold to select relevant feature

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.coef_ = None
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.intercept_ = 0.0
        self.random_state = random_state
        self.selection = selection

    def fit(self, X, y, check_input=True):
        """Fit model with fista.

        Parameters
        -----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data

        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        """
        if self.l1_ratio is not None and self.alpha is not None:
            # tau and mu are selected as enet
            if self.l1_ratio == 1:
                self.mu = 0
                self.tau = 2 * self.alpha
            elif self.l1_ratio == 0:
                self.mu = 2 * self.alpha
                self.tau = 0
            else:
                self.mu = 2 * self.alpha * (1 - self.l1_ratio)
                self.tau = 2 * self.alpha * self.l1_ratio
        else:
            self.l1_ratio = self.tau / (self.tau + 2 * self.mu)
            self.alpha = self.tau * .5 / self.l1_ratio

        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        y = self._label_binarizer.fit_transform(y)
        if self._label_binarizer.y_type_.startswith('multilabel'):
            raise ValueError(
                "%s doesn't support multi-label classification" % (
                    self.__class__.__name__))

        # self.coef_ = self.path(
        #     X, y, self.mu, self.tau, beta=None, kmax=self.max_iter,
        #     tolerance=self.tol, return_iterations=False, adaptive=False)
        super(L1L2Classifier, self).fit(X, y, check_input)

        if self.classes_.shape[0] > 2:
            ndim = self.classes_.shape[0]
        else:
            ndim = 1
        self.coef_ = self.coef_.reshape(ndim, -1)

        return self

    @property
    def classes_(self):
        return self._label_binarizer.classes_

    def _get_support_mask(self):
        check_is_fitted(self, "n_iter_")
        scores = _get_feature_importances(self)
        self.threshold_ = _calculate_threshold(self, scores, self.threshold)
        return scores >= self.threshold_
