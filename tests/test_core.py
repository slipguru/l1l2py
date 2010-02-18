import numpy as np
import scipy.io as sio

from nose.tools import *
from nose.plugins.attrib import attr

from biolearning._core import *
from test_algorithms import TestAlgorithms

from mlabwrap import mlab
mlab.addpath('tests/matlab_code')

TOL = 1e-3

class TestCore(object):
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
     
    def test_double_optimization(self):
        pass
            
    def test_minimal_model(self):
        from biolearning import tools
        
        tau_range = tools.linear_range(0.1, 1.0, 5)
        lambda_range = tools.linear_range(0.1, 1.0, 5)
        
        for mu in tools.linear_range(0.1, 1.0, 10):
            tau_opt_exp, lambda_opt_exp = mlab.l1l2_kcv(self.X, self.Y,
                                            tau_range, lambda_range, mu,
                                            5, 'regr', 0, 1, 1, nout=2)
                           
            sets = TestAlgorithms._get_matlab_splitting(self.Y, 5)    
            tau_opt, lambda_opt = minimal_model(self.X, self.Y, mu,
                                          tau_range, lambda_range, cv_sets=sets,
                                          error_function=tools.regression_error,
                                          data_normalizer=tools.standardize,
                                          labels_normalizer=tools.center)
                                                   
            assert_almost_equals(tau_opt_exp, tau_opt)
            assert_almost_equals(lambda_opt_exp, lambda_opt)
            
    def test_minimal_model_saturated(self):
        from biolearning import tools
        
        tau_range = [0.1, 1.0, 1e3, 1e4]
        lambda_range = tools.linear_range(0.1, 1.0, 5)
        
        for mu in tools.linear_range(0.1, 1.0, 10):
            tau_opt_exp, lambda_opt_exp = mlab.l1l2_kcv(self.X, self.Y,
                                            tau_range, lambda_range, mu,
                                            5, 'regr', 0, 1, 1, nout=2)
                           
            sets = TestAlgorithms._get_matlab_splitting(self.Y, 5)    
            tau_opt, lambda_opt = minimal_model(self.X, self.Y, mu,
                                          tau_range, lambda_range, cv_sets=sets,
                                          error_function=tools.regression_error,
                                          data_normalizer=tools.standardize,
                                          labels_normalizer=tools.center)
                                                   
            assert_almost_equals(tau_opt_exp, tau_opt)
            assert_almost_equals(lambda_opt_exp, lambda_opt)
    