"""Wrapper for l1l2 in case of classification."""

# This code is written by
#       Federico Tomasi <federico.tomasi@dibris.unige.it>
# Copyright (C) 2017 SlipGURU -
# Statistical Learning and Image Processing Genoa University Research Group
# Via Dodecaneso, 35 - 16146 Genova, ITALY.
#
# This file is part of L1L2Py.
#
# L1L2Py is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# L1L2Py is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with L1L2Py. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import six
from sklearn.base import BaseEstimator
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import column_or_1d

from l1l2py.linear_model import L1L2


class L1L2Classifier(LinearClassifierMixin, L1L2):
    """Classification with combined L1 and L2 priors as regularizer.

    Minimizes the objective function::
            1 / n_samples * ||y - Xw||^2_2
            + tau * ||w||_1
            + mu * ||w||^2_2

    using the FISTA method, Dealing with classification problems.

    Parameters
    ----------
    mu : float, optional, default 0.5
        Constant that multiplies the l1 norm.

    tau : float, optional, default 1
        Constant that multiplies the l2 norm.

    use_gpu : bool, optional, default False
        If True, use the implementation of FISTA using the GPU.
        Currently ignored.

    threshold : float, optional, default 1e-16
        Threshold to select relevant variables in the ``transform`` method.

    alpha : float, optional, default None
        Constant that multiplies the penalty terms. Defaults to None.
        This is for parallel with sklearn ElasticNet class.
        See the notes for the exact mathematical meaning of this
        parameter.``alpha = 0`` is equivalent to an ordinary least square,
        solved by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
        Given this, you should use the :class:`LinearRegression` object.

    l1_ratio : float, optional, default None
        This is for parallel with sklearn ElasticNet class.
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    fit_intercept : bool
        Whether the intercept should be estimated or not. If ``False``, the
        data is assumed to be already centered.

    normalize : boolean, optional, default False
        If ``True``, the regressors X will be normalized before regression.
        This parameter is ignored when ``fit_intercept`` is set to ``False``.
        When the regressors are normalized, note that this makes the
        hyperparameters learnt more robust and almost independent of the number
        of samples. The same property is not valid for standardized data.
        However, if you wish to standardize, please use
        :class:`preprocessing.StandardScaler` before calling ``fit`` on an
        estimator with ``normalize=False``.

    precompute : True | False | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. The Gram matrix can also be passed as argument.
        For sparse input this option is always ``True`` to preserve sparsity.

    max_iter : int, optional
        The maximum number of iterations

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    positive : bool, optional
        When set to ``True``, forces the coefficients to be positive.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    random_state : int, RandomState instance, or None (default)
        The seed of the pseudo random number generator that selects
        a random feature to update. Useful only when selection is set to
        'random'.

    Attributes
    ----------
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1) | \
            (n_targets, n_features)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float | array, shape (n_targets,)
        independent term in decision function.

    n_iter_ : array-like, shape (n_targets,)
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.
    """

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


class L1L2TwoStepClassifier(Pipeline):
    r"""L1L2 penalized linear classification with overshinking correction.

    Linear regression with combined L1 and L2 priors as regularizer,
    followed by a ridge classification on the selected variables.

    Minimizes the objective function::
            1 / n_samples * ||y - Xw||^2_2
            + tau * ||w||_1
            + mu * ||w||^2_2

    followed by the minimization of::
            1 / n_samples * ||y - X_tilde w_tilde||^2_2
            + lambda * ||w_tilde||^2_2


    in which `w_tilde` and `X_tilde`
    represent, respectively, the weights vector and the input matrix
    restricted to the variables selected by the l1l2 selection.

    Parameters
    ----------
    tau : float, optional, default 1
        Constant that multiplies the l1 norm.

    mu : float, optional, default 0.5
        Constant that multiplies the l2 norm.

    use_gpu : bool, optional, default False
        If True, use the implementation of FISTA using the GPU.
        Currently ignored.

    alpha : float, optional, default None
        Constant that multiplies the penalty terms. Defaults to None.
        This is for parallel with sklearn ElasticNet class.
        See the notes for the exact mathematical meaning of this
        parameter.``alpha = 0`` is equivalent to an ordinary least square,
        solved by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
        Given this, you should use the :class:`LinearRegression` object.

    l1_ratio : float, optional, default None
        This is for parallel with sklearn ElasticNet class.
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    fit_intercept : bool
        Whether the intercept should be estimated or not. If ``False``, the
        data is assumed to be already centered.

    normalize : boolean, optional, default False
        If ``True``, the regressors X will be normalized before regression.
        This parameter is ignored when ``fit_intercept`` is set to ``False``.
        When the regressors are normalized, note that this makes the
        hyperparameters learnt more robust and almost independent of the number
        of samples. The same property is not valid for standardized data.
        However, if you wish to standardize, please use
        :class:`preprocessing.StandardScaler` before calling ``fit`` on an
        estimator with ``normalize=False``.

    precompute : True | False | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. The Gram matrix can also be passed as argument.
        For sparse input this option is always ``True`` to preserve sparsity.

    max_iter : int, optional
        The maximum number of iterations

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    positive : bool, optional
        When set to ``True``, forces the coefficients to be positive.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    random_state : int, RandomState instance, or None (default)
        The seed of the pseudo random number generator that selects
        a random feature to update. Useful only when selection is set to
        'random'.

    Attributes
    ----------
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1) | \
            (n_targets, n_features)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float | array, shape (n_targets,)
        independent term in decision function.

    n_iter_ : array-like, shape (n_targets,)
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.
    """

    def __init__(self, mu=.5, tau=1.0, lamda=1, use_gpu=False, threshold=1e-16,
                 alpha=None, l1_ratio=None, fit_intercept=True,
                 normalize=False, precompute=False, max_iter=10000,
                 copy_X=True, tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        vs = L1L2(mu=mu, tau=tau, use_gpu=use_gpu, threshold=threshold,
                  alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept,
                  normalize=normalize, precompute=precompute,
                  max_iter=max_iter, copy_X=copy_X, tol=tol,
                  warm_start=warm_start, positive=positive,
                  random_state=random_state, selection=selection)
        mdl = RidgeClassifier(
            alpha=lamda, fit_intercept=fit_intercept,
            normalize=normalize, copy_X=copy_X, max_iter=max_iter,
            tol=tol, random_state=random_state)
        super(L1L2TwoStepClassifier, self).__init__(
            (('l1l2', vs), ('ridge', mdl)))

        self.mu = mu
        self.tau = tau
        self.lamda = lamda
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.use_gpu = use_gpu
        self.threshold = threshold

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

    def fit(self, X, y, **fit_params):
        """Fit Ridge regression model on top of L1L2 selected features.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data

        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values

        sample_weight : float or array-like of shape [n_samples]
            Sample weight

        Returns
        -------
        self : Returns self.
        """
        # print "l1l22step tau", self.tau, "mu", self.mu, 'lamda', self.lamda
        # print self.steps
        fit_params_ = {}
        # account for different names
        map_l1l2 = dict(check_input='check_input')
        for mapped, param in six.iteritems(map_l1l2):
            if fit_params.get(param, None) is not None:
                fit_params_['__'.join(('l1l2', mapped))] = fit_params[param]
        map_ridge = dict(sample_weight='sample_weight')
        for mapped, param in six.iteritems(map_ridge):
            if fit_params.get(param, None) is not None:
                fit_params_['__'.join(('ridge', mapped))] = fit_params[param]
        super(L1L2TwoStepClassifier, self).fit(X, y, **fit_params_)

        # self.coef_ contains a zero vector apart from coef_ selected by Ridge
        l1l2_coef_ = self.steps[0][1].coef_
        ridge_coef_ = self.steps[1][1].coef_
        selected_ = np.nonzero(l1l2_coef_)[0]
        coef_ = np.zeros_like(l1l2_coef_)
        coef_[selected_] = ridge_coef_
        self.coef_ = coef_

        return self

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self
        """
        # kwargs_ = {}
        # account for different names
        map_l1l2 = dict(
            mu='mu', tau='tau', use_gpu='use_gpu', threshold='threshold',
            alpha='alpha', l1_ratio='l1_ratio', fit_intercept='fit_intercept',
            normalize='normalize', precompute='precompute',
            max_iter='max_iter', copy_X='copy_X', tol='tol',
            warm_start='warm_start', positive='positive',
            random_state='random_state', selection='selection')
        for mapped, param in six.iteritems(map_l1l2):
            if kwargs.get(param, None) is not None:
                kwargs['__'.join(('l1l2', mapped))] = kwargs[param]
        map_ridge = dict(
            alpha='lamda', fit_intercept='fit_intercept',
            normalize='normalize', copy_X='copy_X', max_iter='max_iter',
            tol='tol', random_state='random_state')
        for mapped, param in six.iteritems(map_ridge):
            if kwargs.get(param, None) is not None:
                kwargs['__'.join(('ridge', mapped))] = kwargs[param]
        return super(L1L2TwoStepClassifier, self).set_params(**kwargs)


class L1L2StageOneClassifier(LinearClassifierMixin, BaseEstimator):
    """Stage I a la DeMol09.

    Parameters
    ----------
    taus : float, optional, default 1
        Constant that multiplies the l1 norm (step 1).

    mu : float, optional, default 0.5
        Constant that multiplies the l2 norm (step 1).

    lamdas : array-like of floats, optional
        Ridge regression (step 2) regularization constants.

    use_gpu : bool, optional, default False
        If True, use the implementation of FISTA using the GPU.
        Currently ignored.

    fit_intercept : bool
        Whether the intercept should be estimated or not. If ``False``, the
        data is assumed to be already centered.

    normalize : boolean, optional, default False
        If ``True``, the regressors X will be normalized before regression.
        This parameter is ignored when ``fit_intercept`` is set to ``False``.
        When the regressors are normalized, note that this makes the
        hyperparameters learnt more robust and almost independent of the number
        of samples. The same property is not valid for standardized data.
        However, if you wish to standardize, please use
        :class:`preprocessing.StandardScaler` before calling ``fit`` on an
        estimator with ``normalize=False``.

    precompute : True | False | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. The Gram matrix can also be passed as argument.
        For sparse input this option is always ``True`` to preserve sparsity.

    max_iter : int, optional
        The maximum number of iterations

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    positive : bool, optional
        When set to ``True``, forces the coefficients to be positive.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    random_state : int, RandomState instance, or None (default)
        The seed of the pseudo random number generator that selects
        a random feature to update. Useful only when selection is set to
        'random'.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this GridSearchCV instance after fitting.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : boolean, default=True
        If ``'False'``, the ``cv_results_`` attribute will not include training
        scores.


    Attributes
    ----------
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1) | \
            (n_targets, n_features)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float | array, shape (n_targets,)
        independent term in decision function.

    n_iter_ : array-like, shape (n_targets,)
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.
    """

    def __init__(self, taus=(0.1, 0.5, 1), mu=0.5, lamdas=(0.1, 1.0, 10.0),
                 use_gpu=False, threshold=1e-16,
                 fit_intercept=True,
                 normalize=False, precompute=False, max_iter=10000,
                 copy_X=True, tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic',
                 cv=None, scoring=None, n_jobs=1, iid=True, refit=True,
                 verbose=0, pre_dispatch='2*n_jobs', error_score='raise',
                 return_train_score=True):
        self.mu = mu
        self.taus = taus
        self.lamdas = lamdas
        self.use_gpu = use_gpu
        self.threshold = threshold
        self.cv = cv
        self.scoring = scoring
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
        self.n_jobs = n_jobs
        self.iid = iid
        self.refit = refit
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score

    def fit(self, X, y, sample_weight=None):
        """Fit Ridge regression model after searching for the best mu and tau.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data

        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values

        sample_weight : float or array-like of shape [n_samples]
            Sample weight

        Returns
        -------
        self : Returns self.
        """
        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        y = self._label_binarizer.fit_transform(y)
        if self._label_binarizer.y_type_.startswith('multilabel'):
            raise ValueError(
                "%s doesn't support multi-label classification" % (
                    self.__class__.__name__))
        else:
            y = column_or_1d(y, warn=False)

        param_grid = {'tau': self.taus, 'lamda': self.lamdas}
        fit_params = {'sample_weight': sample_weight}
        estimator = L1L2TwoStepClassifier(
            mu=self.mu, fit_intercept=self.fit_intercept,
            use_gpu=self.use_gpu, threshold=self.threshold,
            normalize=self.normalize, precompute=self.precompute,
            max_iter=self.max_iter,
            copy_X=self.copy_X, tol=self.tol, warm_start=self.warm_start,
            positive=self.positive,
            random_state=self.random_state, selection=self.selection)
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid, fit_params=fit_params, cv=self.cv,
            scoring=self.scoring, n_jobs=self.n_jobs, iid=self.iid,
            refit=self.refit, verbose=self.verbose,
            pre_dispatch=self.pre_dispatch, error_score=self.error_score,
            return_train_score=self.return_train_score)
        gs.fit(X, y)
        estimator = gs.best_estimator_
        self.tau_ = estimator.tau
        self.lamda_ = estimator.lamda
        self.coef_ = estimator.coef_
        self.intercept_ = estimator.intercept_
        self.best_estimator_ = estimator  # XXX DEBUG

        if self.classes_.shape[0] > 2:
            ndim = self.classes_.shape[0]
        else:
            ndim = 1
            self.coef_ = self.coef_.reshape(ndim, -1)

        return self

    @property
    def classes_(self):
        return self._label_binarizer.classes_


class L1L2StageTwoClassifier(LinearClassifierMixin, BaseEstimator):
    """Stage I and II a la DeMol09b.

    This implements the two main stages of l1l2 regularization with double
    optimization variable selection, as in [DeMol09b]_.

    Parameters
    ----------
    estimator : instance of L1L2StageOne
        It may not be fitted.
    mus : array-like
        List of `mu` parameter for the Stage II.
    """

    def __init__(self, estimator, mus=(0.5, 0.75, 1)):
        # super(L1L2TwoStepCV, self).__init__(
        #     alphas=lamdas,
        #     fit_intercept=fit_intercept, normalize=normalize, scoring=scoring,
        #     cv=cv, gcv_mode=gcv_mode,
        #     store_cv_values=store_cv_values)
        self.mus = mus
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        """Fit Ridge regression model after searching for the best mu and tau.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data

        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values

        sample_weight : float or array-like of shape [n_samples]
            Sample weight

        Returns
        -------
        self : Returns self.
        """
        if not hasattr(self.estimator, 'fit'):
            raise TypeError("%s is not an estimator instance." % (self.estimator))
        attributes = ('tau_', 'lamda_')
        if not all([hasattr(self.estimator, attr) for attr in attributes]):
            # not fitted
            self.estimator.fit(X, y, sample_weight=sample_weight)

        tau_ = self.estimator.tau_
        lamda_ = self.estimator.lamda_
        params = self.estimator.get_params()

        coef_ = []
        for mu in self.mus:
            estimator = L1L2TwoStepClassifier(
                mu=mu, tau=tau_, lamda=lamda_, use_gpu=params['use_gpu'],
                threshold=params['threshold'],
                fit_intercept=params['fit_intercept'],
                normalize=params['normalize'],
                precompute=params['precompute'], max_iter=params['max_iter'],
                copy_X=params['copy_X'], tol=params['tol'],
                warm_start=params['warm_start'],
                positive=params['positive'],
                random_state=params['random_state'],
                selection=params['selection'])
            estimator_coef_ = estimator.fit(X, y).coef_
            coef_.append(estimator_coef_.copy())

        self.coef_ = coef_

        return self
