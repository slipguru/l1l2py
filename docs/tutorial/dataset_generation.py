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

def correlated_dataset(num_samples, num_variables, groups, weights,
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
    groups : tuple of int
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

    num_relevants = sum(groups)
    num_noisy = num_variables - num_relevants

    X = np.empty((num_samples, num_variables))
    weights = np.asarray(weights).reshape(-1, 1)

    # For each group generates the correlated variables
    var_idx = 0
    for g in groups:
        x = np.random.normal(scale=variables_stdev, size=(num_samples,1))
        err_x = np.random.normal(scale=correlations_stdev, size=(num_samples, g))
        X[:, var_idx:var_idx+g] = x + err_x
        var_idx += g

    # Generates the outcomes
    err_y = np.random.normal(scale=labels_stdev, size=(num_samples,1))
    Y = np.dot(X[:,:num_relevants], weights) + err_y

    # Add noisy variables
    X[:, num_relevants:] = np.random.normal(scale=variables_stdev,
                                            size=(num_samples, num_noisy))

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
