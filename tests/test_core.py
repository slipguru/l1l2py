import numpy as np
import scipy.io as sio

from nose.tools import *
from nose.plugins.attrib import attr

from biolearning._core import *

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
     
    def test_model_selection(self):
        from biolearning import tools
        splits = tools.kfold_splits(self.Y, 2)
        tr_idx, ts_idx = splits[0]
        data, test_data = self.X[tr_idx, :], self.X[ts_idx, :]
        labels, test_labels = self.Y[tr_idx, :], self.Y[ts_idx, :]
        
        tau_range = tools.linear_range(0.1, 1.0, 5)
        lambda_range = tools.linear_range(0.1, 1.0, 6)
        mu_range = tools.linear_range(0.1, 1.0, 10)
        int_splits = tools.kfold_splits(labels, 3)
        
        out = model_selection(data, labels, test_data, test_labels,
                        mu_range, tau_range, lambda_range,
                        int_splits, tools.regression_error,
                        data_normalizer=tools.standardize,
                        labels_normalizer=tools.center,
                        returns_kcv_errors=False)
        assert_equals(6, len(out))
        
        out = model_selection(data, labels, test_data, test_labels,
                        mu_range, tau_range, lambda_range,
                        int_splits, tools.regression_error,
                        data_normalizer=tools.standardize,
                        labels_normalizer=tools.center,
                        returns_kcv_errors=True)
        assert_equals(8, len(out))
        
        (tau_opt, lambda_opt,
         kcv_err_ts, kcv_err_tr,
         beta_list, selected_list, err_tr_list, err_ts_list) = out
        
        assert_equals((len(tau_range), len(lambda_range)), kcv_err_ts.shape)
        assert_equals(kcv_err_tr.shape, kcv_err_ts.shape)
        
        assert_equals(len(mu_range), len(beta_list))
        assert_equals(len(mu_range), len(selected_list))
        assert_equals(len(mu_range), len(err_ts_list))
        assert_equals(len(mu_range), len(err_tr_list))
            
    def test_minimal_model(self):
        from biolearning import tools
        
        tau_range = tools.linear_range(0.1, 1.0, 5)
        lambda_range = tools.linear_range(0.1, 1.0, 5)
        
        for mu in tools.linear_range(0.1, 1.0, 10):
            tau_opt_exp, lambda_opt_exp = mlab.l1l2_kcv(self.X, self.Y,
                                            tau_range, lambda_range, mu,
                                            5, 'regr', 0, 1, 1, nout=2)

            from test_algorithms import TestAlgorithms                           
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

            from test_algorithms import TestAlgorithms                           
            sets = TestAlgorithms._get_matlab_splitting(self.Y, 5)    
            tau_opt, lambda_opt = minimal_model(self.X, self.Y, mu,
                                          tau_range, lambda_range, cv_sets=sets,
                                          error_function=tools.regression_error,
                                          data_normalizer=tools.standardize,
                                          labels_normalizer=tools.center)
                                                   
            assert_almost_equals(tau_opt_exp, tau_opt)
            assert_almost_equals(lambda_opt_exp, lambda_opt)
    