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
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from nose.tools import *

from ..data import center, standardize, tau_max
from ..proximal import Lasso


def test_centering():
    X = [[1., 2., 3.],
         [3., 2., 1.]]
    
    Xcent, mean = center(X, return_mean=True)
    assert_array_almost_equal([2., 2., 2.], mean)
    assert_array_almost_equal([[-1., 0.,  1.],
                               [ 1., 0., -1.]], Xcent)
    
    assert_array_almost_equal(X - mean, Xcent)
    assert_array_almost_equal(Xcent + mean, X)

    Xcent2, mean2 = center(Xcent, return_mean=True)
    assert_array_almost_equal(Xcent, Xcent2)
    assert_array_almost_equal([0., 0., 0.], mean2)

    Xcent, Xcent2 = center(X, X)
    assert_array_almost_equal(Xcent, Xcent2)


def test_centering_outputs():
    y = [1., 2., 3.]
    
    assert_equals(np.ndarray, type(center(y)))
    assert_equals(2, len(center(y, y)))
    assert_equals(2, len(center(y, return_mean=True)))
    assert_equals(3, len(center(y, y, return_mean=True)))


def test_standardization():
    # Note: standardization includes matrix centering!
    X = [[1., 2.], [3., 4.]]
    
    Xstd, mean, std = standardize(X, return_factors=True)
    assert_array_almost_equal((X - mean)/std, Xstd)
    assert_array_almost_equal((Xstd * std) + mean, X)

    Xstd2, mean2, std2 = standardize(Xstd, return_factors=True)
    assert_array_almost_equal(Xstd, Xstd2)
    assert_array_almost_equal(np.zeros(Xstd.shape[1]), mean2)
    assert_array_almost_equal(np.ones(Xstd.shape[1]), std2)
    
    Xstd, Xstd2 = standardize(X, X)
    assert_array_almost_equal(Xstd, Xstd2)
    
    # One row matrix
    assert_raises(ValueError, standardize, np.array([[1, 2, 3]]))    


def test_standardization_outputs():
    X = [[1., 2.], [3., 4.]]
    
    assert_equals(np.ndarray, type(standardize(X)))
    assert_equals(2, len(standardize(X, X)))
    assert_equals(3, len(standardize(X, return_factors=True)))
    assert_equals(4, len(standardize(X, X, return_factors=True)))


def test_tau_max():
    X = [[1., 2.], [3., 4.]]
    y = [1., -1.]
    
    tmax = tau_max(X, y)
    assert_equals(2.0, tmax)

    lasso = Lasso(tau=tmax).train(X, y)
    assert_equals(0, len(lasso.beta.nonzero()[0]))


def test_correlated_dataset():
    pass
    #TODO