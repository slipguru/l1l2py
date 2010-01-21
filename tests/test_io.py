import os
import numpy as np
import scipy.io as sio

from nose.tools import *
from io import Configuration

class TestConfiguration(object):

    def setup(self):     
        self.conf = Configuration('tests/example.cfg')
            
    def test_path(self):
        assert_equals(os.path.abspath('tests/example.cfg'), self.conf.path)
        
    def test_experiment(self):
        assert_equals('regression', self.conf.experiment_type)
        assert_equals(('classification', 'regression'),
                            self.conf.experiment_types)
        
    def test_expressions(self):
        expressions = self.conf.expressions
        expected = sio.loadmat('tests/toy_dataA.mat',
                               struct_as_record=False)['X']
    
        assert_equals(np.ndarray, type(expressions))
        assert_equals(expected.shape, expressions.shape)
        assert_true(np.allclose(expected, expressions))
        assert_true(os.path.abspath('tests/toy_dataA.mat'),
                    self.conf.expressions_path)
        
    def test_labels(self):
        labels = self.conf.labels
        expected = sio.loadmat('tests/toy_dataA.mat',
                               struct_as_record=False)['Y']
        
        assert_equals(np.ndarray, type(labels))
        assert_equals(expected.shape, labels.shape)
        assert_true(np.allclose(expected, labels))
        assert_true(os.path.abspath('tests/toy_dataA.mat'),
                    self.conf.labels_path)
        
    def test_data_types(self):
        assert_equals(('matlab', 'csv'), self.conf.data_types)
        
    def test_range_type(self):
        assert_equals('linear', self.conf.tau_range_type)
        assert_equals('geometric', self.conf.lambda_range_type)
        assert_equals('linear', self.conf.mu_range_type)
        
        assert_equals(('linear', 'geometric'), self.conf.range_types)
        
    def test_range_values(self):
        import tools
        expected_l = np.linspace(0.1, 0.5, 20)
        expected_g = tools.geometric_range(0.1, 0.5, 20)
               
        assert_true(np.allclose(expected_l, self.conf.tau_range))
        assert_true(np.allclose(expected_g, self.conf.lambda_range))
        assert_true(np.allclose(expected_l, self.conf.mu_range))
        
    def test_raw_options(self):
        assert_equals(dict, type(self.conf.raw_options))
        assert_true('expressions_path' in self.conf.raw_options)
        assert_true('parameters_tau-min' in self.conf.raw_options)
        
    def test_kcv_values(self):
        assert_equals(3, self.conf.external_k)
        assert_equals(2, self.conf.internal_k)
        
    def test_split_index(self):
        assert_equals(1, self.conf.split_index)
                