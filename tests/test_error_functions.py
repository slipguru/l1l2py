import numpy as np
import scipy.io as sio

from nose.tools import *

from biolearning.error_functions import *
import biolearning._algorithms as alg

from mlabwrap import mlab
mlab.addpath('tests/matlab_code')

TOL = 1e-3

class TestErrorFunctions(object):
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
                    
    def test_classification_error(self):              
        labels = np.ones(100)
        predicted = labels.copy()
        for exp_error in (0.0, 0.5, 0.75, 1.0):
            index = exp_error*100
            predicted[0:index] = -1
            error = classification_error(labels, predicted)
            assert_almost_equals(exp_error, error)
         
    def test_regression_error(self):
        beta = alg.ridge_regression(self.X, self.Y)
        predicted = np.dot(self.X, beta)
        
        error = regression_error(self.Y, predicted)
        assert_almost_equals(0.0, error)
        
        matlab_error = mlab.linear_test(self.X, self.Y, beta, 'regr')
        assert_almost_equals(matlab_error, error)
    
        predicted_mod = predicted.copy()        
        for num in np.arange(0, self.Y.size, 5):
            predicted_mod[0:num] = predicted[0:num] + 1.0
            exp_error = num / float(self.Y.size)
            
            error = regression_error(self.Y, predicted_mod)
            assert_almost_equals(exp_error, error)
            
    def test_balanced_classification_error(self):
        from biolearning.data_tools import center
        
        labels = np.ones((30, 1))
        for imbalance in np.linspace(0, 30, 4):
            labels[0:imbalance,:] = -1

            labels_centered = center(labels)[0]            
            TestErrorFunctions._test_balanced(self.X, labels)
            TestErrorFunctions._test_balanced(self.X, labels_centered)
    
    @staticmethod
    def _test_balanced(X, labels):
        beta = alg.ridge_regression(X, labels)
        exp_error = mlab.linear_test_weighted(X, labels, beta, 'class')
        
        predicted = np.dot(X, beta) 
        error = balanced_classification_error(labels, predicted)
        
        assert_almost_equals(exp_error, error)
  