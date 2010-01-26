import os
import numpy as np
import scipy.io as sio
from mlabwrap import mlab

from nose.tools import *
import tools

mlab.addpath('tests/matlab_code')

class TestTools(object):
    """
    Results generated with the original matlab code
    """
             
    def test_geometric_ranges(self):
        geom_params = np.array([0.1, 1.0, 10]).reshape((3, 1))       
        exp_geom = mlab.range_values(geom_params)
        geom = tools.parameter_range('geometric', 0.1, 1.0, 10)
        assert_true(np.allclose(exp_geom, geom))
    
    def test_linear_ranges(self):    
        lin_params = np.array([0.0, 0.1, 1.0]).reshape((1, 3))
        exp_lin = mlab.range_values(lin_params)
        lin = tools.parameter_range('linear', 0.0, 1.0, 11)              
        assert_true(np.allclose(exp_lin, lin))
        
        