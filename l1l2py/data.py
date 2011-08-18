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

import numpy as np

# Normalization ---------------------------------------------------------------
def center(matrix, optional_matrix=None, return_mean=False):
    r"""Center columns of a matrix setting each column to zero mean.

    The function returns the centered ``matrix`` given as input.
    Optionally centers an ``optional_matrix`` with respect to the mean value evaluated
    for ``matrix``.

    .. note::

        A one dimensional matrix is considered as a column vector.

    Parameters
    ----------
    matrix : (N,) or (N, P) ndarray
        Input matrix whose columns are to be centered.
    optional_matrix : (N,) or (N, P) ndarray, optional (default is `None`)
        Optional matrix whose columns are to be centered
        using mean of ``matrix``.
        It must have the same number of columns as ``matrix``.
    return_mean : bool, optional (default is `False`)
        If `True` returns mean of ``matrix``.

    Returns
    -------
    matrix_centered : (N,) or (N, P) ndarray
        Centered ``matrix``.
    optional_matrix_centered : (N,) or (N, P) ndarray, optional
        Centered ``optional_matrix`` with respect to ``matrix``
    mean : float or (P,) ndarray, optional
        Mean of ``matrix`` columns.

    Examples
    --------
    >>> X = numpy.array([[1, 2, 3], [4, 5, 6]])
    >>> l1l2py.tools.center(X)
    array([[-1.5, -1.5, -1.5],
           [ 1.5,  1.5,  1.5]])
    >>> l1l2py.tools.center(X, return_mean=True)
    (array([[-1.5, -1.5, -1.5],
           [ 1.5,  1.5,  1.5]]), array([ 2.5,  3.5,  4.5]))
    >>> x = numpy.array([[1, 2, 3]])             # 2-dimensional matrix
    >>> l1l2py.tools.center(x, return_mean=True)
    (array([[ 0.,  0.,  0.]]), array([ 1.,  2.,  3.]))
    >>> x = numpy.array([1, 2, 3])               # 1-dimensional matrix
    >>> l1l2py.tools.center(x, return_mean=True) # centered as a (3, 1) matrix
    (array([-1.,  0.,  1.]), 2.0)
    >>> l1l2py.tools.center(X, X[:,:2])
    Traceback (most recent call last):
        ...
    ValueError: shape mismatch: objects cannot be broadcast to a single shape

    """
    matrix = np.asanyarray(matrix)
    mean = matrix.mean(axis=0)
    
    if not optional_matrix is None:
        optional_matrix = np.asanyarray(optional_matrix)

    # Simple case
    if optional_matrix is None and return_mean is False:
        return matrix - mean

    if optional_matrix is None: # than return_mean is True
        return (matrix - mean, mean)

    if return_mean is False: # otherwise
        return (matrix - mean, optional_matrix - mean)

    # Full case
    return (matrix - mean, optional_matrix - mean, mean)

def standardize(matrix, optional_matrix=None, return_factors=False):
    r"""Standardize columns of a matrix setting each column with zero mean and
    unitary standard deviation.

    The function returns the standardized ``matrix`` given as input.
    Optionally it standardizes an ``optional_matrix`` with respect to the
    mean and standard deviation evaluated from ``matrix``.

    .. note::

        A one dimensional matrix is considered as a column vector.

    Parameters
    ----------
    matrix : (N,) or (N, P) ndarray
        Input matrix whose columns are to be standardized
        to mean `0` and standard deviation `1`.
    optional_matrix : (N,) or (N, P) ndarray, optional (default is `None`)
        Optional matrix whose columns are to be standardized
        using mean and standard deviation of ``matrix``.
        It must have same number of columns as ``matrix``.
    return_factors : bool, optional (default is `False`)
        If `True`, returns mean and standard deviation of ``matrix``.

    Returns
    -------
    matrix_standardized : (N,) or (N, P) ndarray
        Standardized ``matrix``.
    optional_matrix_standardized : (N,) or (N, P) ndarray, optional
        Standardized ``optional_matrix`` with respect to ``matrix``
    mean : float or (P,) ndarray, optional
        Mean of ``matrix`` columns.
    std : float or (P,) ndarray, optional
        Standard deviation of ``matrix`` columns.

    Raises
    ------
    ValueError
        If ``matrix`` has only one row.

    Examples
    --------
    >>> X = numpy.array([[1, 2, 3], [4, 5, 6]])
    >>> l1l2py.tools.standardize(X)
    array([[-0.70710678, -0.70710678, -0.70710678],
           [ 0.70710678,  0.70710678,  0.70710678]])
    >>> l1l2py.tools.standardize(X, return_factors=True)
    (array([[-0.70710678, -0.70710678, -0.70710678],
           [ 0.70710678,  0.70710678,  0.70710678]]), array([ 2.5,  3.5,  4.5]), array([ 2.12132034,  2.12132034,  2.12132034]))
    >>> x = numpy.array([[1, 2, 3]])                     # 1 row matrix
    >>> l1l2py.tools.standardize(x, return_factors=True)
    Traceback (most recent call last):
        ...
    ValueError: 'matrix' must have more than one row
    >>> x = numpy.array([1, 2, 3])                       # 1-dimensional matrix
    >>> l1l2py.tools.standardize(x, return_factors=True) # standardized as a (3, 1) matrix
    (array([-1.,  0.,  1.]), 2.0, 1.0)
    >>> l1l2py.tools.center(X, X[:,:2])
    Traceback (most recent call last):
        ...
    ValueError: shape mismatch: objects cannot be broadcast to a single shape

    """
    matrix = np.asanyarray(matrix)
    
    if matrix.ndim == 2 and matrix.shape[0] == 1:
        raise ValueError("'matrix' must have more than one row")

    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
    
    if not optional_matrix is None:
        optional_matrix = np.asanyarray(optional_matrix)

    # Simple case
    if optional_matrix is None and return_factors is False:
        return (matrix - mean)/std

    if optional_matrix is None: # than return_factors is True
        return (matrix - mean)/std, mean, std

    if return_factors is False: # otherwise
        return (matrix - mean)/std, (optional_matrix - mean)/std

    # Full case
    return (matrix - mean)/std, (optional_matrix - mean)/std, mean, std


def correlated_dataset(num_samples, num_variables,
                       groups_cardinality,
                       weights,
                       variables_stdev=1.0,
                       correlations_stdev=1e-2,
                       labels_stdev=1e-2):
    r"""Random supervised dataset generation with correlated variables.

    The function returns a supervised training set with ``num_samples``
    examples with ``num_variables`` variables.

    Parameters
    ----------
    num_samples : int
        Number of samples.
    num_variables : int
        Number of variables.
    groups_cardinality : tuple of int
        For each group of relevant variables indicates the group cardinality.
    weights : array_like of sum(groups) float
        True regression model.
    variables_stdev : float, optional (default is `1.0`)
        Standard deviation of the zero-mean Gaussian distribution generating
        variables column vectors.
    correlations_stdev : float, optional (default is `1e-2`)
        Standard deviation of the zero-mean Gaussian distribution generating
        errors between variables which belong to the same group
    labels_stdev : float, optional (default is `1e-2`)
        Standard deviation of the zero-mean Gaussian distribution generating
        regression errors.

    Returns
    -------
    X : (``num_samples``, ``num_variables``) ndarray
        Data matrix.
    Y : (``num_samples``, 1) ndarray
        Regression output.

    Notes
    -----
    The data will have ``len(groups)`` correlated groups of variables, where
    for each one the function generates a column vector :math:`\mathbf{x}` of
    ``num_samples`` values drawn from a zero-mean Gaussian distribution
    with standard deviation equal to ``variables_stdev``.

    For each variable of the group associated with the :math:`\mathbf{x}`
    vector, the function generates the  values as

    .. math:: \mathbf{x}^j = \mathbf{x} + \epsilon_x,

    where :math:`\epsilon_x` is additive noise drawn from a zero-mean Gaussian
    distribution with standard deviation equal to ``correlations_stdev``.

    The regression values will be generated as

    .. math::
        \mathbf{Y} = \mathbf{\tilde{X}}\boldsymbol{\tilde{\beta}} + \epsilon_y,

    where :math:`\boldsymbol{\tilde{\beta}}` is the ``weights`` parameter, a
    list of ``sum(groups)`` coefficients of the relevant variables,
    :math:`\mathbf{\tilde{X}}` is the submatrix containing only the column
    related to the relevant variables and :math:`\epsilon_y` is additive noise drawn
    from a zero-mean Gaussian distribution with standard deviation equal to
    ``labels_stdev``.

    At the end the function returns the matrices
    :math:`\mathbf{X}` and :math:`\mathbf{Y}` where

    .. math:: \mathbf{X} = [\mathbf{\tilde{X}}; \mathbf{X_N}]

    is the concatenation of the matrix :math:`\mathbf{\tilde{X}}` with the
    relevant variables with ``num_variables - sum(groups)`` noisy variables
    generated indipendently using values drawn from a zero-mean Gaussian
    distribution with standard deviation equal to ``variables_stdev``.

    Examples
    --------
    >>> X, Y = correlated_dataset(30, 40, (5, 5, 5), [3.0]*15)
    >>> X.shape
    (30, 40)
    >>> Y.shape
    (30, 1)

    """

    num_relevants = sum(groups_cardinality)
    num_noisy = num_variables - num_relevants

    X = np.empty((num_samples, 0))
    weights = np.asarray(weights)

    # For each group generates the correlated variables
    var_idx = 0
    for g in groups_cardinality:
        x = np.random.normal(scale=variables_stdev, size=(num_samples, 1))
        err_x = np.random.normal(scale=correlations_stdev, size=(num_samples, g))
        X = np.c_[X, x + err_x]
        var_idx += g

    # Generates the outcomes
    err_y = np.random.normal(scale=labels_stdev, size=num_samples)
    Y = np.dot(X, weights) + err_y

    # Add noisy variables
    unrelevant = np.random.normal(scale=variables_stdev,
                                  size=(num_samples, num_noisy))
    X = np.c_[X, unrelevant]

    n, d = X.shape
    assert d == num_variables
    assert n == num_samples

    return X, Y


def main():
    import sys

    num_samples = int(sys.argv[1])
    num_variables = int(sys.argv[2])

    if num_variables < 9:
        raise ValueError('needed at least 9 variables')

    print 'Generation of %d samples with %d variables...' % (num_samples,
                                                             num_variables),

    X, Y = correlated_dataset(num_samples, num_variables, (5, 5, 5), [1.0]*15)
    np.savetxt('data.txt', X)
    np.savetxt('labels.txt', Y)

    print 'done'



if __name__ == '__main__':
    main()
