import numpy as np
import scipy.io as sio

from nose.tools import *

from biolearning.data_tools import *

from mlabwrap import mlab
mlab.addpath('tests/matlab_code')

TOL = 1e-3

class TestDataTools(object):
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
        geom = geometric_range(0.1, 1.0, 10)
        assert_true(np.allclose(exp_geom, geom))
    
    def test_linear_ranges(self):    
        lin_params = np.array([0.0, 0.1, 1.0]).reshape((1, 3))
        exp_lin = mlab.range_values(lin_params)
        lin = linear_range(0.0, 1.0, 11)              
        assert_true(np.allclose(exp_lin, lin))
        
    def test_centering(self):
        Xexp, Yexp, Xexp2, Yexp2, mean_exp = mlab.normalization(self.X, self.Y,
                                                                True, False,
                                                                self.X, self.Y,
                                                                nout=5)
        Ycent, mean = center(self.Y, return_mean=True)
        assert_true(np.allclose(Yexp, Ycent, TOL))
        assert_true(np.allclose(Yexp, Yexp2))
        assert_almost_equal(mean_exp, mean)
        
        Ycent, Ycent2 = center(self.Y, self.Y)
        assert_true(np.allclose(Ycent, Ycent2))
        
    def test_centering_outputs(self):
        assert_equals(np.ndarray, type(center(self.Y)))
        assert_equals(2, len(center(self.Y, self.Y)))
        assert_equals(2, len(center(self.Y, return_mean=True)))
        assert_equals(3, len(center(self.Y, self.Y, return_mean=True)))
        
    def test_standardization(self):
        Xexp, Yexp, Xexp2, Yexp2, mean_exp = mlab.normalization(self.X, self.Y,
                                                                True, True,
                                                                self.X, self.Y,
                                                                nout=5)
        # Note: standardization includes matrix centering!
        Xstd, mean, std = standardize(self.X, return_factors=True)
        
        assert_true(np.allclose(Xexp, Xstd, TOL))
        assert_true(np.allclose(Xexp, Xexp2))
        
    def test_standardization_outputs(self):
        assert_equals(np.ndarray, type(standardize(self.X)))
        assert_equals(2, len(standardize(self.X, self.X)))
        assert_equals(3, len(standardize(self.X, return_factors=True)))
        assert_equals(4, len(standardize(self.X, self.X, return_factors=True)))
        