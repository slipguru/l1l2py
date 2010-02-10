import os
import numpy as np
import scipy.io as sio
from mlabwrap import mlab

from nose.tools import *
import framework as fw
import tools

mlab.addpath('tests/matlab_code')

class TestFramework(object):
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
       
    def test_select_features(self):
        tau_range = tools.linear_range(0.1, 1.0, 5)
        lambda_range = tools.linear_range(0.1, 1.0, 5)
        
        for mu in tools.linear_range(0.1, 1.0, 10):
            tau_opt_exp, lambda_opt_exp = mlab.l1l2_kcv(self.X, self.Y,
                                            tau_range, lambda_range, mu,
                                            5, 'regr', 0, 1, 1, nout=2)
                           
            sets = TestFramework._get_matlab_splitting(self.Y, 5)    
            tau_opt, lambda_opt = fw.select_features(self.X, self.Y, mu,
                                    tau_range, lambda_range, cv_sets=sets,
                                    error_function=tools.regression_error,
                                    data_normalizer=tools.standardize,
                                    labels_normalizer=tools.center)
                                                   
            assert_almost_equals(tau_opt_exp, tau_opt)
            assert_almost_equals(lambda_opt_exp, lambda_opt)
          
    @staticmethod
    def _get_matlab_splitting(labels, K):
        mlab_sets = mlab.splitting(labels, 5, 0)
        mlab_sets = mlab.double(mlab.cell2mat(mlab_sets)).T #row = set
        mlab_sets -= 1 #matlab starts from 1
    
        indexes = np.arange(labels.size)
        sets = list()
        for ts in mlab_sets:
            ts = np.array(ts, dtype=np.int)
            tr = np.array(list(set(indexes) - set(ts)))
            sets.append((tr, ts))
            
        return sets
    
    def test_build_classifier(self):
        sets = tools.kfold_splits(self.Y, 2)
        train_idx, test_idx = sets[0]
        
        Xtr, Xts = self.X[train_idx,:], self.X[test_idx,:]
        Ytr, Yts = self.Y[train_idx,:], self.Y[test_idx,:]
        mu_range = tools.linear_range(0.1, 1.0, 5)
        
        from itertools import product
        values = np.linspace(0.1, 10.0, 5)
        for tau, lam in product(values, values):
            beta_opt_exp = mlab.step2(Xtr, Ytr, Xts, Yts, tau, lam,
                                      mu_range, 'regr', 1, 1)
            
            beta_opt = fw.build_classifier(Xtr, Ytr, Xts, Yts,
                                           tau, lam, mu_range,
                                           error_function=tools.regression_error,
                                           data_normalizer=tools.standardize,
                                           labels_normalizer=tools.center)
            
            for i in xrange(mlab.length(beta_opt_exp)):
                exp = mlab.cell_element(beta_opt_exp, i+1)
                assert_true(np.allclose(exp, beta_opt[i]))
             
    