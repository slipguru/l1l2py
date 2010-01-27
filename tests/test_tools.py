import os
import numpy as np
import scipy.io as sio
from mlabwrap import mlab

from nose.tools import *
import tools

mlab.addpath('tests/matlab_code')
TOL = 1e-3

class TestTools(object):
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
             
    def test_geometric_ranges(self):
        geom_params = np.array([0.1, 1.0, 10]).reshape((3, 1))       
        exp_geom = mlab.range_values(geom_params)
        geom = tools.geometric_range(0.1, 1.0, 10)
        assert_true(np.allclose(exp_geom, geom))
    
    def test_linear_ranges(self):    
        lin_params = np.array([0.0, 0.1, 1.0]).reshape((1, 3))
        exp_lin = mlab.range_values(lin_params)
        lin = tools.linear_range(0.0, 1.0, 11)              
        assert_true(np.allclose(exp_lin, lin))
        
    def test_centering(self):
        Xexp, Yexp, Xexp2, Yexp2, mean_exp = mlab.normalization(self.X, self.Y,
                                                                True, False,
                                                                self.X, self.Y,
                                                                nout=5)
        Ycent, mean = tools.center(self.Y)
        assert_true(np.allclose(Yexp, Ycent, TOL))
        assert_true(np.allclose(Yexp, Yexp2))
        assert_almost_equal(mean_exp, mean)
        
        Ycent, Ycent2, mean = tools.center(self.Y, self.Y)
        assert_true(np.allclose(Ycent, Ycent2))
        
    def test_standardization(self):
        Xexp, Yexp, Xexp2, Yexp2, mean_exp = mlab.normalization(self.X, self.Y,
                                                                True, True,
                                                                self.X, self.Y,
                                                                nout=5)
        # Note: standardization includes matrix centering!
        Xstd = tools.standardize(self.X)
        
        assert_true(np.allclose(Xexp, Xstd, TOL))
        assert_true(np.allclose(Xexp, Xexp2))
        