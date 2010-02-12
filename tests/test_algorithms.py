import numpy as np
import scipy.io as sio

from nose.tools import *
from nose.plugins.attrib import attr

from biolearning._algorithms import *

from mlabwrap import mlab
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
            value = ridge_regression(self.X, self.Y, penalty)
            assert_true(np.allclose(expected, value, TOL))
        
        expected = ridge_regression(self.X, self.Y, 0.0)
        value = ridge_regression(self.X, self.Y)
        assert_true(np.allclose(expected, value, TOL))
        
        # case d > n
        X = self.X.T
        Y = self.X[0:1,:].T
        for penalty in np.linspace(0.0, 1.0, 5):
            expected = mlab.rls_algorithm(X, Y, penalty)
            value = ridge_regression(X, Y, penalty)
            assert_true(np.allclose(expected, value, TOL))
            
        expected = ridge_regression(X, Y, 0.0)
        value = ridge_regression(X, Y)
        assert_true(np.allclose(expected, value, TOL))
        
    def test_soft_thresholding(self):
        from biolearning._algorithms import _soft_thresholding
        
        b = ridge_regression(self.X, self.Y, 0.1)
        expected = mlab.thresholding(b, 0.2)
        value = _soft_thresholding(b, 0.2)
        assert_true(np.allclose(expected, value))
        
        value = _soft_thresholding(b, 0.0)
        assert_true(np.allclose(b, value, TOL))

    def test_elastic_net(self):
        from itertools import product
        values = np.linspace(0.1, 1.0, 5)
        for mu, tau in product(values, values):
            exp_beta, exp_k = mlab.l1l2_algorithm(self.X, self.Y,
                                                  tau, mu, nout=2)
            beta, k = elastic_net(self.X, self.Y, mu, tau)
       
            assert_true(np.allclose(exp_beta, beta, TOL))
            assert_true(np.allclose(exp_k, k))
    
    @attr('slow')    
    def test_elastic_net_slow(self):
        from itertools import product
        values = np.linspace(0.0, 2.0, 10)
        for mu, tau in product(values, values):
            exp_beta, exp_k = mlab.l1l2_algorithm(self.X, self.Y,
                                                  tau, mu, nout=2)
            beta, k = elastic_net(self.X, self.Y, mu, tau)
       
            assert_true(np.allclose(exp_beta, beta, TOL))
            assert_true(np.allclose(exp_k, k))
            
    def test_regpath(self):
        values = np.linspace(0.1, 1.0, 5)
        beta_path = elastic_net_regpath(self.X, self.Y,
                                            0.1, values, kmax=np.inf)
        selected = (beta_path != 0)
        
        exp_selected = mlab.l1l2_regpath(self.X, self.Y,
                                         values, 0.1, kmax=np.inf)
        exp_selected = mlab.double(mlab.cell2mat(exp_selected)); # need because
        exp_selected = np.split(exp_selected, values.size)       # return a cell
        
        for b, s in zip(selected, exp_selected):
            # note: s contains 0s and 1s, b contains True and False values
            assert_true(np.all(b == s.squeeze()))
            
    def test_kcv_models_selection(self):
        from biolearning import data_tools as tools
        from biolearning import error_functions as err
        
        tau_range = tools.linear_range(0.1, 1.0, 5)
        lambda_range = tools.linear_range(0.1, 1.0, 5)
        
        for mu in tools.linear_range(0.1, 1.0, 10):
            tau_opt_exp, lambda_opt_exp = mlab.l1l2_kcv(self.X, self.Y,
                                            tau_range, lambda_range, mu,
                                            5, 'regr', 0, 1, 1, nout=2)
                           
            sets = TestAlgorithms._get_matlab_splitting(self.Y, 5)    
            tau_opt, lambda_opt = kcv_model_selection(self.X, self.Y, mu,
                                          tau_range, lambda_range, cv_sets=sets,
                                          error_function=err.regression_error,
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
            
    def test_reverse_enumerate(self):
        from biolearning._algorithms import _reverse_enumerate
        
        iterable = np.array((2, 3, 4, 5, 6))
        rev_enumerate = ((4, 6), (3, 5), (2, 4), (1, 3), (0, 2))
        for p1, p2 in zip(rev_enumerate, _reverse_enumerate(iterable)):
            assert_equal(p1, p2)
        
        