import os
import numpy as np
import scipy.io as sio
from mlabwrap import mlab

from nose.tools import *
import tools
import algorithms as alg

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
        Xstd, mean, std = tools.standardize(self.X)
        
        assert_true(np.allclose(Xexp, Xstd, TOL))
        assert_true(np.allclose(Xexp, Xexp2))
        
    def test_kfold_splits(self):
        splits = tools.kfold_splits(self.Y, 2)
        assert_equal(2, len(splits))
        TestTools._test_splits(self.Y.size, splits)
        
    def test_stratified_kfold_splits(self):
        labels = np.ones(100)
        negative = np.arange(0, labels.size, 2)
        labels[negative] = -1
        assert_equal(int(labels.size/2.0), labels[labels > 0].size)
        assert_equal(int(labels.size/2.0), labels[labels < 0].size)
        
        splits = tools.stratified_kfold_splits(labels, 2)
        assert_equal(2, len(splits))
        
        TestTools._test_splits(labels.size, splits)
        TestTools._test_balancing(labels, splits)        
        
    def test_stratification(self):
        labels = np.ones(100)
        negative = np.arange(0, int(labels.size/2), 2) # only 25% of negative
        labels[negative] = -1
        assert_equal(100.0 - int(labels.size/4.0), labels[labels > 0].size)
        assert_equal(int(labels.size/4.0),  labels[labels < 0].size)
        
        splits = tools.stratified_kfold_splits(labels, 2)
        assert_equal(2, len(splits))

        TestTools._test_splits(labels.size, splits)
        TestTools._test_balancing(labels, splits)
        
    def test_rseed(self):
        splits1 = tools.kfold_splits(self.Y, 2, 10)
        splits2 = tools.kfold_splits(self.Y, 2, 10)
        splits3 = tools.kfold_splits(self.Y, 2, 1)
        assert_equal(splits1, splits2)
        assert_not_equal(splits1, splits3)
        assert_not_equal(splits2, splits3)
        
        labels = np.ones(100)
        negative = np.arange(0, labels.size, 2)
        labels[negative] = -1
        splits1 = tools.stratified_kfold_splits(labels, 2, 10)
        splits2 = tools.stratified_kfold_splits(labels, 2, 10)
        splits3 = tools.stratified_kfold_splits(labels, 2, 1)
        assert_equal(splits1, splits2)
        assert_not_equal(splits1, splits3)
        assert_not_equal(splits2, splits3)
               
    @staticmethod
    def _test_splits(labels_size, splits):
        test1, test2 = splits[0][0], splits[1][0]
        train1, train2 = splits[0][1], splits[1][1]
        #assert_equal(test1, train2)
        #assert_equal(test2, train1)
        
        test_idxs = sorted(test1 + test2)
        train_idxs = sorted(train1 + train2)
        #assert_equal(range(labels_size), test_idxs)
        #assert_equal(range(labels_size), train_idxs)
        
    @staticmethod
    def _test_balancing(labels, splits):
        positive = labels[labels > 0].size / float(labels.size)
        negative = labels[labels < 0].size / float(labels.size)
        
        # Test sets
        test1, test2 = labels[splits[0][0]], labels[splits[1][0]]
        test1_p = test1[test1 > 0].size/float(test1.size) #% of labels
        test1_n = test1[test1 < 0].size/float(test1.size)
        test2_p = test2[test2 > 0].size/float(test2.size)
        test2_n = test2[test2 < 0].size/float(test2.size)
        assert_almost_equal(positive, test1_p, 1)
        assert_almost_equal(positive, test2_p, 1)
        assert_almost_equal(negative, test1_n, 1)
        assert_almost_equal(negative, test2_n, 1)

        # Training sets
        train1, train2 = labels[splits[0][1]], labels[splits[1][1]]   
        train1_p = train1[train1 > 0].size/float(train1.size) #% of labels
        train1_n = train1[train1 < 0].size/float(train1.size)
        train2_p = train2[train2 > 0].size/float(train2.size)
        train2_n = train2[train2 < 0].size/float(train2.size)
        assert_almost_equal(positive, train1_p, 1)
        assert_almost_equal(positive, train2_p, 1)
        assert_almost_equal(negative, train1_n, 1)
        assert_almost_equal(negative, train2_n, 1)

    def test_classification_error(self):              
        labels = np.ones(100)
        predicted = labels.copy()
        for exp_error in (0.0, 0.5, 0.75, 1.0):
            index = exp_error*100
            predicted[0:index] = -1
            error = tools.classification_error(labels, predicted)
            assert_almost_equals(exp_error, error)
         
    def test_regression_error(self):
        beta = alg.ridge_regression(self.X, self.Y)
        predicted = np.dot(self.X, beta)
        
        error = tools.regression_error(self.Y, predicted)
        assert_almost_equals(0.0, error)
        
        matlab_error = mlab.linear_test(self.X, self.Y, beta, 'regr')
        assert_almost_equals(matlab_error, error)

        predicted_mod = predicted.copy()        
        for num in np.arange(0, self.Y.size, 5):
            predicted_mod[0:num] = predicted[0:num] + 1.0
            exp_error = num / float(self.Y.size)
            
            error = tools.regression_error(self.Y, predicted_mod)
            assert_almost_equals(exp_error, error)
            
    def test_reverse_enumerate(self):
        iterable = np.array((2, 3, 4, 5, 6))
        rev_enumerate = ((4, 6), (3, 5), (2, 4), (1, 3), (0, 2))
        for p1, p2 in zip(rev_enumerate, tools.reverse_enumerate(iterable)):
            assert_equal(p1, p2)