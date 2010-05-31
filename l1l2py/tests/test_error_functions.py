import numpy as np

from nose.tools import *

from l1l2py.tools import *
import l1l2py.algorithms as alg

import os
data_path = os.path.join(os.path.dirname(__file__), 'data.txt')

class TestErrorFunctions(object):
    """
    Results generated with the original matlab code
    """

    def setup(self):
        data = np.loadtxt(data_path)
        self.X = data[:,:-1]
        self.Y = data[:,-1]

    def test_data(self):
        assert_equals((30, 40), self.X.shape)
        assert_equals((30, ), self.Y.shape)

    def test_classification_error(self):
        labels = np.ones(100)
        predictions = labels.copy()
        for exp_error in (0.0, 0.5, 0.75, 1.0):
            index = exp_error*100
            predictions[0:index] = -1
            error = classification_error(labels, predictions)
            assert_almost_equals(exp_error, error)

    def test_regression_error(self):
        beta = alg.ridge_regression(self.X, self.Y)
        predictions = np.dot(self.X, beta)

        error = regression_error(self.Y, predictions)
        assert_almost_equals(0.0, error)

        predictions_mod = predictions.copy()
        for num in np.arange(0, self.Y.size, 5):
            predictions_mod[0:num] = predictions[0:num] + 1.0
            exp_error = num / float(self.Y.size)

            error = regression_error(self.Y, predictions_mod)
            assert_almost_equals(exp_error, error)

    def test_balanced_classification_error(self):
        labels = np.ones(100)
        predictions = np.ones(100)

        for imbalance in np.linspace(10, 90, 9):
            labels[:imbalance] = -1
            exp_error = (imbalance * abs(-1 - labels.mean()))/ 100.0

            error = balanced_classification_error(labels, predictions)

            assert_almost_equals(exp_error, error)

    def test_balance_weights(self):
        labels = [1, 1, -1, -1, -1]
        predictions = [-1, -1, 1, 1, 1] # all errors
        default_weights = np.abs(center(np.asarray(labels)))

        exp_error = balanced_classification_error(labels, predictions)
        error = balanced_classification_error(labels, predictions, default_weights)
        assert_equals(exp_error, error)

        null_weights = np.ones_like(labels)
        exp_error = classification_error(labels, predictions)
        error = balanced_classification_error(labels, predictions, null_weights)
        assert_equals(exp_error, error)

        # Balanced classes
        labels = [1, 1, 1, -1, -1, -1]
        predictions = [-1, -1, -1, 1, 1, 1] # all errors
        exp_error = classification_error(labels, predictions)
        error = balanced_classification_error(labels, predictions)
        assert_equals(exp_error, error)

    def test_input_shape_errors(self):
        from itertools import product

        values = [1, 1, -1, -1, -1]
        values_array = np.asarray(values)
        values_array_2d = values_array.reshape(-1, 1)
        values_array_2dT = values_array_2d.T

        list_of_values = [values, values_array,
                          values_array_2d, values_array_2dT]

        for l, p in product(list_of_values, list_of_values):
            assert_equal(0.0, regression_error(l, p))
            assert_equal(0.0, classification_error(l, p))
            assert_equal(0.0, balanced_classification_error(l, p))

        print regression_error(np.asarray([[1, 1, 1],[1, 1, 1]]),
                               np.asarray([[1, 1, 1],[1, 1, 1]]))
