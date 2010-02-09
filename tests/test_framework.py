import os
import numpy as np
import scipy.io as sio
from mlabwrap import mlab

from nose.tools import *
import framework as fw
import tools

mlab.addpath('tests/matlab_code')
TOL = 1e-3

class TestFramework(object):
    """
    Results generated with the original matlab code
    """
    
    def setup(self):
        data = sio.loadmat('tests/toy_dataA.mat', struct_as_record=False)
        self.X = data['X']
        self.Y = data['Y']
        self.tau_range = tools.linear_range(0.1, 1.0, 5)
        self.lambda_range = tools.linear_range(0.1, 1.0, 5)
        self.mu = 0.1
        
    def test_data(self):
        assert_equals((30, 40), self.X.shape)
        assert_equals((30, 1), self.Y.shape)
       
    def test_select_features(self):
        for mu in tools.linear_range(0.1, 1.0, 10):
            tau_opt_exp, lambda_opt_exp = \
                                mlab.l1l2_kcv(self.X, self.Y, self.tau_range,
                                              self.lambda_range, mu,
                                             #K, type, split, mean, col
                                              5, 'regr', 0,    1,    1, nout=2)
                           
            sets = TestFramework._get_matlab_splitting(self.Y, 5)    
            tau_opt, lambda_opt = fw.select_features(self.X, self.Y, mu,
                                    self.tau_range, self.lambda_range,
                                    cv_sets=sets,
                                    error_function=tools.regression_error,
                                    data_normalizer=tools.standardize,
                                    labels_normalizer=tools.center)
                                                   
            assert_almost_equals(tau_opt_exp, tau_opt)
            assert_almost_equals(lambda_opt_exp, lambda_opt)
          
    @staticmethod
    def _get_matlab_splitting(labels, K):
        sets2 = mlab.splitting(labels, 5, 0)
        sets2 = mlab.double(mlab.cell2mat(sets2)); #each column = Test Set
        sets2 -=1 #matlab starts from 1
        sets = list()
        indexes = np.arange(labels.size)
        for ts in sets2.T: #loop on columns
            ts = np.array(ts, dtype=np.int)
            tr = np.array(list(set(indexes) - set(ts)))
            sets.append((tr, ts))
            
        return sets
          
       
    #def test_geometric_ranges(self):
    #    geom_params = np.array([0.1, 1.0, 10]).reshape((3, 1))       
    #    exp_geom = mlab.range_values(geom_params)
    #    geom = tools.geometric_range(0.1, 1.0, 10)
    #    assert_true(np.allclose(exp_geom, geom))
    
    