"""l1l2py main functions.

In this module are implemented the two main stages of the l1l2 with
double optimization variable selection.

"""
## This code is written by Salvatore Masecchia <salvatore.masecchia@unige.it>
## and Annalisa Barla <annalisa.barla@unige.it>
## Copyright (C) 2010 SlipGURU -
## Statistical Learning and Image Processing Genoa University Research Group
## Via Dodecaneso, 35 - 16146 Genova, ITALY.
##
## This file is part of L1L2Py.
##
## L1L2Py is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## L1L2Py is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with L1L2Py. If not, see <http://www.gnu.org/licenses/>.

__all__ = ['model_selection', 'minimal_model', 'nested_models']

import numpy as np
import itertools as it

from l1l2py.algorithms import ridge_regression, l1l2_regularization, l1l2_path

def model_selection(data, labels, test_data, test_labels,
                    mu_range, tau_range, lambda_range,
                    cv_splits, cv_error_function, error_function,
                    data_normalizer=None, labels_normalizer=None,
                    sparse=False, regularized=True,
                    return_predictions=False):
    r"""Complete model selection procedure.

    It executes the two stages implemented in ``minimal_model`` and
    ``nested_models`` and returns their output wrapped in a dictionary.

    Note that the error function calculated in the *Stage I* may have more
    than one minimum.

    By default the less sparse but more regularized
    solution (minimum value of ``tau`` and maximum value of ``lambda``) is
    selected, **in the set of (tau, lambda) pairs with minimum error**, .

    The boolean parameters ``sparse`` and ``regularized`` allow to change this
    behaviour.

    .. note::

        See the functions documentation for details on each stage and the
        meaning of each parameter. The **Parameters** section
        describes only the ``sparse`` and ``regularized`` parameters.

    Parameters
    ----------
    sparse : bool, optional (default is `False`)
        If `True`, the function selects at STAGE I the sparsest solution with
        minimum cross validation error.
    regularized : bool, optional (default is `True`)
        If `True`, the function selects at STAGE I the most regularized solution
        with minimum cross validation error.

    Returns
    -------
    out : dict
        Output dictionary. According with the parameters the dictionary has
        the following keys:

        **kcv_err_ts** : (T, L) ndarray
            [STAGE I] Mean cross validation errors on the training set.
        **kcv_err_tr** : (T, L) ndarray
            [STAGE I] Mean cross validation errors on the training set.
        **tau_opt** : float
            Optimal value of tau selected in ``tau_range``.
        **lambda_opt** : float
            Optimal value of lambda selected in ``lambda_range``.
        **beta_list** :  list of M (S,1) ndarray
            [STAGE II] Models calculated for each value in ``mu_range``.
        **selected_list** : list of M (P,) ndarray of boolean
            [STAGE II] Selected variables for each model calculated.
        **err_ts_list** : list of M floats
            [STAGE II] List of Test errors evaluated for the all the models.
        **err_tr_list** : list of M floats
            [STAGE II] List of Training errors evaluated for the all the models.
        **prediction_ts_list** : list of M two-dimensional ndarray, optional
            [STAGE II] Prediction vectors for the models evaluated on the test
            set.
        **prediction_tr_list** : list of M two-dimensional ndarray, optional
            [STAGE II] Prediction vectors for the models evaluated on the
            training set.

    """

    # STAGE I
    stage1_out = minimal_model(data, labels, mu_range[0],
                               tau_range, lambda_range,
                               cv_splits, cv_error_function,
                               data_normalizer, labels_normalizer)
    out = dict(it.izip(('kcv_err_ts', 'kcv_err_tr'), stage1_out))

    # KCV MINIMUM SELECTION
    err_ts = out['kcv_err_ts']
    tau_opt_idxs, lambda_opt_idxs = np.where(err_ts == err_ts.min())
    tau_opt, lambda_opt = _minimum_selection(tau_opt_idxs, lambda_opt_idxs,
                                             sparse, regularized)
    out['tau_opt'] = tau_range[tau_opt]
    out['lambda_opt'] = lambda_range[lambda_opt]

    # STAGE II
    stage2_out = nested_models(data, labels,
                               test_data, test_labels,
                               mu_range, out['tau_opt'], out['lambda_opt'],
                               error_function,
                               data_normalizer, labels_normalizer,
                               return_predictions)

    keys = ['beta_list', 'selected_list', 'err_ts_list', 'err_tr_list']
    if return_predictions:
        keys.append('prediction_ts_list')
        keys.append('prediction_tr_list')

    out.update(it.izip(keys, stage2_out))

    return out

def _minimum_selection(tau_idxs, lambda_idxs, sparse=False, regularized=False):
    r"""Selection of the miminum error coordinates.

    Given two ranges of minimum errors coordinates selects
    the right pair according to the two parameters sparse, regularized.

    """

    from collections import defaultdict

    d = defaultdict(list)
    for t, l in it.izip(tau_idxs, lambda_idxs):
        d[t].append(l)

    tau_idx = max(d.keys()) if sparse else min(d.keys())
    lam_idx = max(d[tau_idx]) if regularized else min(d[tau_idx])

    return tau_idx, lam_idx

def minimal_model(data, labels, mu, tau_range, lambda_range,
                  cv_splits, error_function,
                  data_normalizer=None, labels_normalizer=None):
    r"""Minimal model selection.

    Given a supervised training set (``data`` and ``labels``), for a fixed
    value of ``mu`` (should be minimum), it finds the values in ``tau_range``
    and ``lambda_range`` minimizing the prediction error via cross validation
    (see error functions in the ``l1l2py.tools`` module).

    Cross validation splits must be provided (``cv_splits``) as a list
    of pairs containing traning-set and validation-set indexes
    (see cross validation tools in the ``l1l2py.tools`` module).

    Data and labels will be normalized on each split using the function
    ``data_normalizer`` and ``labels_normalizer``
    (see data normalization functions in the ``l1l2py.tools`` module).

    .. warning ::

        On each cross validation split the number of valid solutions (not void)
        may be different (on high values of ``tau``).
        The function calculates the optimum value of ``tau`` for which the model
        is not void on all cross validation splits.

        **This means than in extreme cases the output could be void.**

    Parameters
    ----------
    data : (N, P) ndarray
        Data matrix.
    labels : (N,)  or (N, 1) ndarray
        Labels vector.
    mu : float
        Minimum `l2` norm penalty (`l1l2` functional).
    tau_range : array_like of `T` floats
        `l1` norm penalties (`l1l2` functional).
    lambda_range : array_like `L` of floats
        `l2` norm penalties (`RLS` functional).
    cv_splits : array_like of tuples
        Each tuple contains two lists with the training set and testing set
        indexes.
    error_function : function object
        Cross validation error function.
    data_normalizer : function object, optional (default is `None`)
        Data normalization function.
    labels_normalizer : function object, optional (default is `None`)
        Labels normalization function.

    Returns
    -------
    err_ts : (< T, L) ndarray
        Matrix of average cross validation error on the training set.
        The first dimension depends on the number of valid ``tau`` values,
        **even zero**.
    err_tr : (< T, L) ndarray
        Matrix of average cross validation error on the training set.
        The first dimension depends on the number of valid ``tau`` values,
        **even zero**.

    Raises
    ------
    ValueError
        If the given range of ``tau`` values produces all void solutions with
        the given data splits.

    """

    err_ts = list()
    err_tr = list()
    max_tau_num = len(tau_range)

    for train_idxs, test_idxs in cv_splits:
        # First create a view and then normalize (eventually)
        data_tr, data_ts = data[train_idxs, :], data[test_idxs, :]
        if not data_normalizer is None:
            data_tr, data_ts = data_normalizer(data_tr, data_ts)

        labels_tr, labels_ts = labels[train_idxs, :], labels[test_idxs, :]
        if not labels_normalizer is None:
            labels_tr, labels_ts = labels_normalizer(labels_tr, labels_ts)

        # Builds a classifier for each value of tau
        beta_casc = l1l2_path(data_tr, labels_tr, mu, tau_range[:max_tau_num])

        if len(beta_casc) == 0:
            raise ValueError("the given range of 'tau' values produces all "
                             "void solutions with the given data splits")

        max_tau_num = min(max_tau_num, len(beta_casc))
        _err_ts = np.empty((max_tau_num, len(lambda_range)))
        _err_tr = np.empty_like(_err_ts)

        # For each sparse model builds a
        # rls classifier for each value of lambda
        for j, beta in it.izip(xrange(max_tau_num), beta_casc):
            selected = (beta.flat != 0)
            for k, lam in enumerate(lambda_range):
                beta = ridge_regression(data_tr[:, selected], labels_tr, lam)

                prediction = np.dot(data_ts[:, selected], beta)
                _err_ts[j, k] = error_function(labels_ts, prediction)

                prediction = np.dot(data_tr[:, selected], beta)
                _err_tr[j, k] = error_function(labels_tr, prediction)

        err_ts.append(_err_ts)
        err_tr.append(_err_tr)

    # cut columns and computes the mean
    err_ts = np.asarray([a[:max_tau_num] for a in err_ts]).mean(axis=0)
    err_tr = np.asarray([a[:max_tau_num] for a in err_tr]).mean(axis=0)

    return err_ts, err_tr

def nested_models(data, labels, test_data, test_labels,
                  mu_range, tau, lambda_, error_function,
                  data_normalizer=None, labels_normalizer=None,
                  return_predictions=False):
    r"""The function generates the models with the (almost) nested lists of
    selected variables.

    Given a training set (``data`` and ``labels``) and a test set
    (``test_data`` and ``test_labels``), for fixed values of ``tau``
    and ``lambda`` (should be the optimal values estimated at Stage I),
    it calculates one model for each increasing value in ``mu_range``.

    Data and labels will be normalized using the function ``data_normalizer``
    and ``labels_normalizer`` (see data normalization functions
    in the ``l1l2py.tools`` module).

    The function returns test and training errors using the
    ``error_function`` provided (see error functions
    in the ``l1l2py.tools`` module).

    Parameters
    ----------
    data : (N, P) ndarray
        Data matrix.
    labels : (N,)  or (N, 1) ndarray
        Labels vector.
    test_data : (T, P) ndarray
        Test set matrix.
    test_labels : (T,)  or (T, 1) ndarray
        Test set labels vector.
    mu_range : array_like of M floats
        `l2` norm penalties (`l1l2` functional).
    tau : float
        Optimal `l1` norm penalty (`l1l2` functional).
    lambda_: float
        Optimal `l2` norm penalty (`RLS` functional).
    error_function : function object
        Error function.
    data_normalizer : function object, optional (default is `None`)
        Data normalization function.
    labels_normalizer : function object, optional (default is `None`)
        Labels normalization function.

    Returns
    -------
    beta_list : list of M (S,1) ndarray
        Models calculated for each value in ``mu_range``.
    selected_list : list of M (P,) ndarray of boolean
        Selected feature for each models calculated.
    err_ts_list : list of M floats
        Test error for the models calculated.
    err_tr_list : list of M floats
        Training error for the models calculated.
    prediction_ts_list : list of M (T, 1) ndarray
        Prediction vector calculated for each value in ``mu_range`` on the
        test set.
    prediction_tr_list : list of M (N, 1) ndarray
        Prediction vector calculated for each value in ``mu_range`` on the
        training set.

    Raises
    ------
    ValueError
        If the given value of ``tau`` produces a void solution with the
        given data.

    """

    if not data_normalizer is None:
        data, test_data = data_normalizer(data, test_data)

    if not labels_normalizer is None:
        labels, test_labels = labels_normalizer(labels, test_labels)

    beta_list = list()
    selected_list = list()
    err_ts_list = list()
    err_tr_list = list()

    if return_predictions:
        prediction_ts_list = list()
        prediction_tr_list = list()

    for mu in mu_range:
        beta = l1l2_regularization(data, labels, mu, tau)
        selected = (beta.flat != 0)

        if not selected.any():
            raise ValueError("the given value of 'tau' produces a void "
                             "solution with the given data")

        beta = ridge_regression(data[:, selected], labels, lambda_)

        beta_list.append(beta)
        selected_list.append(selected)

        prediction_ts = np.dot(test_data[:, selected], beta)
        err_ts_list.append(error_function(test_labels, prediction_ts))

        prediction_tr = np.dot(data[:, selected], beta)
        err_tr_list.append(error_function(labels, prediction_tr))

        if return_predictions:
            prediction_ts_list.append(prediction_ts)
            prediction_tr_list.append(prediction_tr)

    if return_predictions:
        return (beta_list, selected_list, err_ts_list, err_tr_list,
                prediction_ts_list, prediction_tr_list)
    else:
        return beta_list, selected_list, err_ts_list, err_tr_list
