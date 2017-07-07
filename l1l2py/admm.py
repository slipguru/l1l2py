import numpy as np
from scipy import linalg as la
import sys
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import sparse

from sklearn.base import RegressorMixin
from sklearn.linear_model.base import _preprocess_data
from sklearn.utils import check_array, check_X_y, deprecated
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import check_cv
from sklearn.externals.joblib import Parallel, delayed
from sklearn.externals import six
from sklearn.externals.six.moves import xrange
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import column_or_1d
from sklearn.exceptions import ConvergenceWarning

from sklearn.utils import check_array, check_X_y, deprecated
from sklearn.linear_model.base import LinearModel, _pre_fit
from sklearn.linear_model.coordinate_descent import _alpha_grid

from .data import center


def shrinkage(x, kappa):
    return np.maximum(0, x - kappa) - np.maximum(0, -x - kappa)


def enet_admm(X, y, z=None, rho=1.0, alpha=1.0, max_iter=1000, abs_tol=1e-6,
              rel_tol=1e-4, tau=0.5, mu=0.5):
    n, d = X.shape

    XTy = np.dot(X.T, y)

    x = np.zeros(d)
    z = np.zeros(d)
    u = np.zeros(d)

    L, U = factor(X, rho, mu)

    for k in xrange(max_iter):
        # x-update
        q = 2. / n * XTy + rho * (z - u)    # temporary value

        if n >= d:      # if skinny
            x = la.solve_triangular(U, la.solve_triangular(L, q, lower=True),
                                    lower=False)
        else:            # if fat
            tmp = la.solve_triangular(U, la.solve_triangular(L, np.dot(X, q),
                                      lower=True), lower=False)
            x = q / rho - np.dot(X.T, tmp) * (2. / (n * rho * rho))

        # z-update with relaxation
        zold = z
        x_hat = alpha * x + (1 - alpha) * zold
        z = shrinkage(x_hat + u, tau / rho)

        # u-update
        u += (x_hat - z)

        # Stopping
        r_norm = la.norm(x - z)
        s_norm = la.norm(-rho * (z - zold))

        eps_pri = np.sqrt(d) * abs_tol + rel_tol * max(la.norm(x), la.norm(-z))
        eps_dual = np.sqrt(d) * abs_tol + rel_tol * la.norm(rho * u)

        if (r_norm < eps_pri) and (s_norm < eps_dual):
            break

    return z, s_norm, eps_dual, k + 1


def enet_admm_path(X, y, fit_intercept=True, tau=0.5, mu=0.5,
                   rho=1.0, alpha=1.0,
                   max_iter=1000, abs_tol=1e-6, rel_tol=1e-4,
                   normalize=False,
                   copy_X=True, warm_start=False, positive=False,
                   random_state=None, selection='cyclic',
                   alphas=None, precompute='auto', Xy=None, coef_init=None,
                   verbose=False, return_n_iter=False,
                   check_input=True, **params):
    # We expect X and y to be already Fortran ordered when bypassing
    # checks
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
        # No need to normalize of fit_intercept: it has been done
        # above
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

    for i, mu in enumerate(alphas):
        l1_reg = tau
        l2_reg = mu
        if not multi_output and sparse.isspmatrix(X):
            # model = cd_fast.sparse_enet_coordinate_descent(
            #     coef_, l1_reg, l2_reg, X.data, X.indices,
            #     X.indptr, y, X_sparse_scaling,
            #     max_iter, tol, rng, random, positive)
            raise NotImplementedError()
        elif multi_output:
            raise NotImplementedError()
            # model = cd_fast.enet_coordinate_descent_multi_task(
            #     coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random)
        elif isinstance(precompute, np.ndarray):
            # We expect precompute to be already Fortran ordered when bypassing
            # checks
            if check_input:
                precompute = check_array(precompute, dtype=np.float64,
                                         order='C')
            # model = cd_fast.enet_coordinate_descent_gram(
            #     coef_, l1_reg, l2_reg, precompute, Xy, y, max_iter,
            #     tol, rng, random, positive)
            # raise NotImplementedError()
            model = enet_admm(
                X, y, coef_, rho=rho, alpha=alpha, max_iter=max_iter,
                abs_tol=abs_tol, rel_tol=rel_tol, tau=tau, mu=mu)
        elif precompute is False:
            model = enet_admm(
                X, y, coef_, rho=rho, alpha=alpha, max_iter=max_iter,
                abs_tol=abs_tol, rel_tol=rel_tol, tau=tau, mu=mu)
            # coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random,
            # positive)
        else:
            raise ValueError("Precompute should be one of True, False, "
                             "'auto' or array-like. Got %r" % precompute)
        coef_, dual_gap_, eps_, n_iter_ = model
        coefs[..., i] = coef_
        dual_gaps[i] = dual_gap_
        n_iters.append(n_iter_)
        if dual_gap_ > eps_:
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
                sys.stderr.write('.')

    if return_n_iter:
        return alphas, coefs, dual_gaps, n_iters
    return alphas, coefs, dual_gaps


class ElasticNet(LinearModel):
    def __init__(self, fit_intercept=True, tau=1, mu=0.5,
                 rho=1.0, alpha=1.0,
                 max_iter=1000, abs_tol=1e-6, rel_tol=1e-4,
                 normalize=False, precompute=False,
                 copy_X=True, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):

        self.tau = tau
        self.mu = mu
        self.rho = rho      # step size
        self.alpha = alpha  # over relaxation parameter
        self.max_iter = max_iter
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.intercept_ = 0.0
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.normalize = normalize
        self.precompute = precompute
        self.copy_X = copy_X
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

    def fit(self, X, y, check_input=True):
        if check_input:
            X, y = check_X_y(X, y, accept_sparse='csc',
                             order='F', dtype=[np.float64, np.float32],
                             copy=self.copy_X and self.fit_intercept,
                             multi_output=True, y_numeric=True)
            y = check_array(y, order='F', copy=False, dtype=X.dtype.type,
                            ensure_2d=False)

        X, y, X_offset, y_offset, X_scale, precompute, Xy = \
            _pre_fit(X, y, None, self.precompute, self.normalize,
                     self.fit_intercept, copy=False)

        # # Centering Data
        # if self.fit_intercept:
        #     X, Xmean = center(X, return_mean=True)
        #     y, ymean = center(y, return_mean=True)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        if Xy is not None and Xy.ndim == 1:
            Xy = Xy[:, np.newaxis]

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        if self.selection not in ['cyclic', 'random']:
            raise ValueError("selection should be either random or cyclic.")

        if not self.warm_start or self.coef_ is None:
            coef_ = np.zeros((n_targets, n_features), dtype=X.dtype,
                             order='F')
        else:
            coef_ = self.coef_
            if coef_.ndim == 1:
                coef_ = coef_[np.newaxis, :]

        dual_gaps_ = np.zeros(n_targets, dtype=X.dtype)
        self.n_iter_ = []

        for k in xrange(n_targets):
            if Xy is not None:
                this_Xy = Xy[:, k]
            else:
                this_Xy = None
            _, this_coef, this_dual_gap, this_iter = enet_admm_path(
                X, y[:, k], rho=self.rho, alpha=self.alpha,
                max_iter=self.max_iter, return_n_iter=True,
                abs_tol=self.abs_tol, rel_tol=self.rel_tol, tau=self.tau,
                mu=self.mu, alphas=[self.mu])
            coef_[k] = this_coef[:, 0]
            dual_gaps_[k] = this_dual_gap[0]
            self.n_iter_.append(this_iter[0])

        # # Fitting the intercept if required
        # if self.fit_intercept:
        #     self._intercept = ymean - np.dot(Xmean, self.coef_)
        # else:
        #     self._intercept = 0.0
        if n_targets == 1:
            self.n_iter_ = self.n_iter_[0]

        self.coef_, self.dual_gap_ = map(np.squeeze, [coef_, dual_gaps_])
        self._set_intercept(X_offset, y_offset, X_scale)

        self.coef_ = np.asarray(self.coef_, dtype=X.dtype)
        return self


class Lasso(ElasticNet):
    def __init__(self, fit_intercept=True, tau=0.5,
                 rho=1.0, alpha=1.0,
                 max_iter=1000, abs_tol=1e-6, rel_tol=1e-4):
        super(Lasso, self).__init__(
            fit_intercept=fit_intercept, tau=tau, mu=0.0, rho=rho, alpha=alpha,
            max_iter=max_iter, abs_tol=abs_tol, rel_tol=rel_tol)


def factor(X, rho, mu=0.0):
    n, d = X.shape

    if n >= d:
        L = la.cholesky((2. / n) * np.dot(X.T, X) + (2. * mu + rho) * np.eye(d), lower=True)
    else:
        L = la.cholesky(np.eye(n) + (2. / (rho * n)) * np.dot(X, X.T), lower=True)

    return L, L.T  # L, U
