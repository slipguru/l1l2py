## This code is written by Salvatore Masecchia <salvatore.masecchia@unige.it>
## and Annalisa Barla <annalisa.barla@unige.it>
## Copyright (C) 2010 SlipGURU -
## Statistical Learning and Image Processing Genoa University Research Group
## Via Dodecaneso, 35 - 16146 Genova, ITALY.
##
## This file is part of L1L2Py.
##
## L1L2Py is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## L1L2Py is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with L1L2Py. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from nose.tools import *
from l1l2py.tools import *
from l1l2py.tests import _TEST_DATA_PATH

class TestKCVTools(object):

    def setup(self):
        data = np.loadtxt(_TEST_DATA_PATH)
        self.X = data[:,:-1]
        self.Y = data[:,-1]

    def test_data(self):
        assert_equals((30, 40), self.X.shape)
        assert_equals((30, ), self.Y.shape)

    def test_exceptions(self):
        assert_raises(ValueError, kfold_splits, self.Y, 0)
        assert_raises(ValueError, kfold_splits, self.Y, len(self.Y) + 1)

        # Not two class
        assert_raises(ValueError, stratified_kfold_splits, self.Y, 2)

        labels = np.ones(100)
        labels[::2] = -1

        # k, out of range
        assert_raises(ValueError, stratified_kfold_splits, labels, 1)
        assert_raises(ValueError, stratified_kfold_splits, labels, 51)

        # More negatives (75 vs 25)
        labels[:50] = -1
        assert_raises(ValueError, stratified_kfold_splits, labels, 26)

    def test_kfold_splits(self):
        for k in xrange(2, self.X.shape[0]):
            splits = kfold_splits(self.Y, k)

            assert_equal(k, len(splits))
            TestKCVTools._test_splits(self.Y.size, splits)

    def test_stratified_kfold_splits(self):
        labels = np.ones(100)
        negative = np.arange(0, labels.size, 2)
        labels[negative] = -1
        assert_equal(int(labels.size/2.0), labels[labels > 0].size)
        assert_equal(int(labels.size/2.0), labels[labels < 0].size)

        splits = stratified_kfold_splits(labels, 2)
        assert_equal(2, len(splits))

        TestKCVTools._test_splits(labels.size, splits)
        TestKCVTools._test_balancing(labels, splits)

    def test_stratification(self):
        labels = np.ones(100)
        negative = np.arange(0, int(labels.size/2), 2) # only 25% of negative
        labels[negative] = -1
        assert_equal(100.0 - int(labels.size/4.0), labels[labels > 0].size)
        assert_equal(int(labels.size/4.0),  labels[labels < 0].size)

        splits = stratified_kfold_splits(labels, 2)
        assert_equal(2, len(splits))

        TestKCVTools._test_splits(labels.size, splits)
        TestKCVTools._test_balancing(labels, splits)

    def test_rseed(self):
        splits1 = kfold_splits(self.Y, 2, 10)
        splits2 = kfold_splits(self.Y, 2, 10)
        splits3 = kfold_splits(self.Y, 2, 1)
        assert_equal(splits1, splits2)
        assert_not_equal(splits1, splits3)
        assert_not_equal(splits2, splits3)

        labels = np.ones(100)
        negative = np.arange(0, labels.size, 2)
        labels[negative] = -1
        splits1 = stratified_kfold_splits(labels, 2, 10)
        splits2 = stratified_kfold_splits(labels, 2, 10)
        splits3 = stratified_kfold_splits(labels, 2, 1)
        assert_equal(splits1, splits2)
        assert_not_equal(splits1, splits3)
        assert_not_equal(splits2, splits3)
        
    def test_rseed_client_code(self):
        import random
        
        labels = np.ones(10)
        labels[0:5] = -1
        
        for split_func in (kfold_splits, stratified_kfold_splits):
            random.seed(1)
            indexes_1 = [random.random() for i in range(10)]
            split_func(labels, 2)
            indexes_2 = [random.random() for i in range(10)]
            
            assert_not_equal(indexes_1, indexes_2)        
            
            split_func(labels, 2)
            indexes_3 = [random.random() for i in range(10)]
            
            assert_not_equal(indexes_1, indexes_2)
            assert_not_equal(indexes_1, indexes_3)
            assert_not_equal(indexes_2, indexes_3)        

    @staticmethod
    def _test_splits(labels_size, splits):

        cum_train = list()
        cum_test = list()

        for train, test in splits:
            assert_equal(labels_size, len(train + test))
            assert_true(set(test).isdisjoint(set(train)))

            cum_train.extend(train)
            cum_test.extend(test)

        assert_equal(set(cum_train), set(cum_test))

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
