import numpy as np
import scipy.io as sio

from nose.tools import *

from l1l2py.tools import *
import l1l2py.algorithms as alg

class TestErrorFunctions(object):
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

