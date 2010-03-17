r"""Core functions.

In this module are implemented the two main stage of the model selection
framwework presented in [DeMol09]_:

* :func:`minimal_model` (`Stage I`)
* :func:`nested_lists`  (`Stage II`)

There is also a simple wrapper of the above functions which can be used to
perform sequentially the two stages:

* :func:`model_selection`

"""

__all__ = ['model_selection', 'minimal_model', 'nested_lists']

import numpy as np
import itertools as it

from algorithms import *
from tools import *

def model_selection(data, labels, test_data, test_labels,
                    mu_range, tau_range, lambda_range,
                    cv_splits, error_function,
                    data_normalizer=None, labels_normalizer=None,
                    returns_kcv_errors=False):
    r"""Complete model selection procedure.

    This functions implements the full model selection framework
    presented in [DeMol09]_.

    The model selection consist of the sequential execution of two stage,
    implemented in the functions :func:`minimal_model` and :func:`nested_lists`.
    See the function documentation for details on each stage and the meaning
    of each parameter.

    .. warning::
    
        This function is a simple wrapper wich calls the other two
        core function and returns a concatenation of the outputs of
        :func:`minimal_model` and :func:`nested_lists`.

    See Also
    --------
    minimal_model
    nested_lists  

    """

    stage1_out = minimal_model(data, labels, mu_range[0],
                               tau_range, lambda_range,
                               cv_splits, error_function,
                               data_normalizer, labels_normalizer,
                               returns_kcv_errors)
    tau_opt, lambda_opt = stage1_out[0:2]

    stage2_out = nested_lists(data, labels,
                              test_data, test_labels,
                              tau_opt, lambda_opt, mu_range,
                              error_function,
                              data_normalizer, labels_normalizer)

    return stage1_out + stage2_out

def minimal_model(data, labels, mu, tau_range, lambda_range,
                  cv_splits, error_function,
                  data_normalizer=None, labels_normalizer=None,
                  returns_kcv_errors=False):
    r"""Performs the minimal model selection.
    
    Given a supervised training set (``data`` and ``labels``), for the fixed
    value of ``mu`` (should be minimum), finds the values in ``tau_range``
    and ``lambda_range`` with minimum performance error via cross validation.
    
    Cross validation splits must be provided (``cv_splits``) as a list
    of pairs containing traning-set and validation-set indexes
    (see cross validation tools in :mod:`biolearning.tools`).
    
    Data and labels will be normalized on each split using the function
    ``data_normalizer`` and ``labels_normalizer``.
    
    .. warning ::
    
        On each cross validation split the number of nonempty model could be
        different (on high value of tau).
        
        The function calculates the optimum value of tau for wich the model is
        nonempty on every cross validation split.
    
    
    Parameters
    ----------
    data : (N, D) ndarray
        Data matrix.
    labels : (N,)  or (N, 1) ndarray
        Labels vector.
    mu : float
        :math:`\ell_2` norm penalty.
    tau_range : array_like of float
        :math:`\ell_1` norm penalties.
    lambda_range : array_like of float
        :math:`\ell_1` norm penalties.
    cv_splits : array_like of tuples
        Each tuple contains two lists with the training set and testing set
        indexes, like the output of the cross validation tools
        in :mod:`biolearning.tools`.
    error_function : function object
        A function like the error functions in :mod:`biolearning.tools`.
    data_normalizer : function object
        A function like the data normalization functions in
        :mod:`biolearning.tools`.
    labels_normalizer : function object
        A function like the data normalization functions in
        :mod:`biolearning.tools`.
    returns_kcv_errors : boolean
        If True returns the cross validation errors calculated.

    Returns
    -------
    tau_opt : float
        Optimal value of tau selected in ``tau_range``.        
    lambda_opt : float
        Optimal value of lambda selected in ``lambda_range``.
    err_ts : list, optional
        List of maximum ``len(cv_splits)`` cross validation error on the
        test set.
    err_tr : list, optional
        List of maximum ``len(cv_splits)`` cross validation error on the
        training set.

    See Also
    --------
    model_selection
    nested_lists
    biolearning.tools    
    
    """

    err_ts = list()
    err_tr = list()
    max_tau_num = len(tau_range)

    for train_idxs, test_idxs in cv_splits:
        # First create a view and then normalize (eventually)
        data_tr, data_ts = data[train_idxs,:], data[test_idxs,:]
        if not data_normalizer is None:
            data_tr, data_ts = data_normalizer(data_tr, data_ts)

        labels_tr, labels_ts = labels[train_idxs,:], labels[test_idxs,:]
        if not labels_normalizer is None:
            labels_tr, labels_ts = labels_normalizer(labels_tr, labels_ts)

        # Builds a classifier for each value of tau
        beta_casc = l1l2_path(data_tr, labels_tr, mu, tau_range[:max_tau_num])

        max_tau_num = min(max_tau_num, len(beta_casc))
        _err_ts = np.empty((max_tau_num, len(lambda_range)))
        _err_tr = np.empty_like(_err_ts)

        # For each sparse model builds a
        # rls classifier for each value of lambda
        for j, b in it.izip(xrange(max_tau_num), beta_casc):
            selected = (b.flat != 0)
            for k, lam in enumerate(lambda_range):
                beta = ridge_regression(data_tr[:,selected], labels_tr, lam)

                prediction = np.dot(data_ts[:,selected], beta)
                _err_ts[j, k] = error_function(labels_ts, prediction)

                prediction = np.dot(data_tr[:,selected], beta)
                _err_tr[j, k] = error_function(labels_tr, prediction)

        err_ts.append(_err_ts)
        err_tr.append(_err_tr)

    # cut columns and computes the mean
    err_ts = np.asarray([a[:max_tau_num] for a in err_ts]).mean(axis=0)
    err_tr = np.asarray([a[:max_tau_num] for a in err_tr]).mean(axis=0)

    tau_opt_idx, lambda_opt_idx = np.where(err_ts == err_ts.min())
    tau_opt = tau_range[tau_opt_idx[0]]
    lambda_opt = lambda_range[lambda_opt_idx[0]]

    if returns_kcv_errors:
        return tau_opt, lambda_opt, err_ts, err_tr
    else:
        return tau_opt, lambda_opt

def nested_lists(data, labels, test_data, test_labels,
                 tau, lambda_, mu_range, error_function=None,
                 data_normalizer=None, labels_normalizer=None):
    r"""TODO: add docstring
    
        Parameters
    ----------
    data : (N, D) ndarray
        Data matrix.
    labels : (N,)  or (N, 1) ndarray
        Labels vector.
    test_data : (N, D) ndarray
        Test set matrix.
    test_labels : (N,)  or (N, 1) ndarray
        Test set labels vector.
    mu_range : array_like of float
        :math:`\ell_2` norm penalties.
    tau : float
        :math:`\ell_1` norm penalty.
    lambda : float
        :math:`\ell_1` norm penalty.
        error_function : function object
    data_normalizer : function object
        A function like the data normalization functions in
        :mod:`biolearning.tools`.
    labels_normalizer : function object
        A function like the data normalization functions in
        :mod:`biolearning.tools`.

    Returns
    -------

    See Also
    --------
    model_selection
    minimal_model
    biolearning.tools
    
    """

    if not data_normalizer is None:
        data, test_data = data_normalizer(data, test_data)

    if not labels_normalizer is None:
        labels, test_labels = labels_normalizer(labels, test_labels)

    beta_list = list()
    selected_list = list()
    if error_function:
        err_tr_list = list()
        err_ts_list = list()

    for mu in mu_range:
        beta = l1l2_regularization(data, labels, mu, tau)
        selected = (beta.flat != 0)

        beta = ridge_regression(data[:,selected], labels, lambda_)

        if error_function:
            prediction = np.dot(data[:,selected], beta)
            err_tr_list.append(error_function(labels, prediction))

            prediction = np.dot(test_data[:,selected], beta)
            err_ts_list.append(error_function(test_labels, prediction))

        beta_list.append(beta)
        selected_list.append(selected)

    if error_function:
        return beta_list, selected_list, err_tr_list, err_ts_list
    else:
        return beta_list, selected_list
