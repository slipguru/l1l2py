import os
import numpy as np
import scipy.io as sio
from mlabwrap import mlab

from nose.tools import *
import algorithms as alg

mlab.addpath('tests/matlab_code')
TOL = 1e-3

class TestAlgorithms(object):
    """
    Results generated with the original matlab code
    """

    def setup(self):     
        data = sio.loadmat('tests/toy_dataA.mat', struct_as_record=False)
        self.X = data['X']
        self.Y = data['Y']
        
    def test_data(self):
        assert_equals((30, 40), self.X.shape)
        assert_equals((30, 1), self.Y.shape)
              
    def test_rls(self):
        # case n >= d
        for penalty in np.linspace(0.0, 1.0, 5):
            expected = mlab.rls_algorithm(self.X, self.Y, penalty)
            value = alg.ridge_regression(self.X, self.Y, penalty)
            assert_true(np.allclose(expected, value, TOL))
        
        expected = alg.ridge_regression(self.X, self.Y, 0.0)
        value = alg.ridge_regression(self.X, self.Y)
        assert_true(np.allclose(expected, value, TOL))
        
        # case d > n
        X = self.X.T
        Y = self.X[0:1,:].T
        for penalty in np.linspace(0.0, 1.0, 5):
            expected = mlab.rls_algorithm(X, Y, penalty)
            value = alg.ridge_regression(X, Y, penalty)
            assert_true(np.allclose(expected, value, TOL))
            
        expected = alg.ridge_regression(X, Y, 0.0)
        value = alg.ridge_regression(X, Y)
        assert_true(np.allclose(expected, value, TOL))
        
    def test_soft_thresholding(self):
        b = alg.ridge_regression(self.X, self.Y, 0.1)
        expected = mlab.thresholding(b, 0.2)
        value = alg.soft_thresholding(b, 0.2)
        assert_true(np.allclose(expected, value))
        
        value = alg.soft_thresholding(b, 0.0)
        assert_true(np.allclose(b, value, TOL))

    def test_elastic_net(self):
        from itertools import product
        values = np.linspace(0.1, 1.0, 5)
        for mu, tau in product(values, values):
            exp_beta, exp_k = mlab.l1l2_algorithm(self.X, self.Y,
                                                  tau, mu, nout=2)
            beta, k = alg.elastic_net(self.X, self.Y, mu, tau)
    
            assert_true(np.allclose(exp_beta, beta, TOL))
            assert_true(np.allclose(exp_k, k))        
        
        