import os
import numpy as np
import scipy.io as sio
from mlabwrap import mlab

from nose.tools import *
import algorithms as alg

mlab.addpath('../REGRESSION_TOOLBOXES/L1L2_TOOLBOX/')
mlab.addpath('../REGRESSION_TOOLBOXES/MISCELLANEOUS/')
mlab.addpath('../REGRESSION_TOOLBOXES/RLS_TOOLBOX/')
TOL = 1e-3

class TestConfiguration(object):
    """
    Some result generated with the original matlab code
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
        for mu in np.linspace(0.1, 1.0, 5):
            for tau in np.linspace(0.1, 1.0, 5):
                exp_beta, exp_k = mlab.l1l2_algorithm(self.X, self.Y,
                                                      tau, mu, nout=2)
                beta, k = alg.elastic_net(self.X, self.Y, mu, tau)
        
                assert_true(np.allclose(exp_beta, beta, TOL))
                assert_true(np.allclose(exp_k, k))

    def test_classification_error(self):              
        labels = np.ones(100)
        predicted = labels.copy()
        for exp_error in (0.0, 0.5, 0.75, 1.0):
            index = exp_error*100
            predicted[0:index] = -1
            error = alg.prediction_error(labels, predicted, 'classification')
            assert_almost_equals(exp_error, error)
            
    def test_regression_error(self):
        beta = alg.ridge_regression(self.X, self.Y)
        predicted = np.dot(self.X, beta)
        
        error = alg.prediction_error(self.Y, predicted, 'regression')
        assert_almost_equals(0.0, error)

        predicted_mod = predicted.copy()        
        for num in np.arange(0, self.Y.size, 5):
            predicted_mod[0:num] = predicted[0:num] + 1.0
            exp_error = num / float(self.Y.size)
            error = alg.prediction_error(self.Y, predicted_mod, 'regression')
            assert_almost_equals(exp_error, error)
        
        
        