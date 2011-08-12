import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from nose.tools import *

from ..estimators import Ridge
from ..data import center
from ..metrics import regression_error, classification_error, \
                      balanced_classification_error

def test_classification_error():
    """Test classification error."""
    labels = np.ones(100)
    predictions = labels.copy()
    for exp_error in (0.0, 0.5, 0.75, 1.0):
        index = exp_error*100
        predictions[0:index] = -1
        error = classification_error(labels, predictions)
        assert_almost_equal(exp_error, error)

def test_loo_error():
    """Test Leave-One-Out classification error."""
    labels = np.ones(1)

    for predictions, exp_error in (([0], 1.0), ([1], 0.0)):
        error = classification_error(labels, predictions)
        assert_almost_equal(exp_error, error)
        
        error = balanced_classification_error(labels, predictions)
        assert_almost_equal(0.0, error) # Always zero in LOO!
        
        error = regression_error(labels, predictions)
        assert_almost_equal(exp_error, error)


def test_regression_error():
    """Test regression error."""
    random_state = np.random.RandomState(0)
    X = random_state.randn(10, 100)
    y = random_state.randn(10)
    
    ridge = Ridge(mu=0.0).fit(X, y)    
    predictions = ridge.predict(X)
    error = regression_error(y, predictions)
    assert_almost_equals(0.0, error)

    predictions_mod = predictions.copy()
    for num in np.arange(0, len(y), 5):
        predictions_mod[0:num] = predictions[0:num] + 1.0
        exp_error = num / float(len(y))

        error = regression_error(y, predictions_mod)
        assert_almost_equals(exp_error, error)

def test_balanced_classification_error():
    """Test balanced classification error."""
    labels = np.ones(100)
    predictions = np.ones(100)

    for imbalance in np.linspace(10, 90, 9):
        labels[:imbalance] = -1
        exp_error = (imbalance * abs(-1 - labels.mean()))/ 100.0

        error = balanced_classification_error(labels, predictions)

        assert_almost_equals(exp_error, error)

def test_balance_weights():
    """Test balanced classification error with custom weights."""
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


def test_input_shape_errors():
    """Test inputs error functions."""
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

