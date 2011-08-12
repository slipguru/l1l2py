import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from nose.tools import *

from ..cross_val import KFold, StratifiedKFold

def test_kfold_splits():
    """Test KFold splits generation."""
    random_state = np.random.RandomState(0)
    X = random_state.randn(10, 100)
    y = random_state.randn(10)
    
    for k in xrange(2, X.shape[0]):
        splits = list(KFold(len(y), k))

        assert_equal(k, len(splits))
        _test_splits(len(y), splits)
        
def test_stratified_kfold_splits():
    """Test StratifiedKFold splits generation."""
    labels = np.ones(100)
    negative = np.arange(0, labels.size, 2)
    labels[negative] = -1
    assert_equal(int(labels.size/2.0), labels[labels > 0].size)
    assert_equal(int(labels.size/2.0), labels[labels < 0].size)

    splits = list(StratifiedKFold(labels, 2))
    assert_equal(2, len(splits))

    _test_splits(labels.size, splits)
    _test_balancing(labels, splits)
    
def test_stratification():
    """Test StratifiedKFold stratification."""
    labels = np.ones(100)
    negative = np.arange(0, int(labels.size/2), 2) # only 25% of negative
    labels[negative] = -1
    assert_equal(100.0 - int(labels.size/4.0), labels[labels > 0].size)
    assert_equal(int(labels.size/4.0),  labels[labels < 0].size)

    splits = list(StratifiedKFold(labels, 2))
    assert_equal(2, len(splits))

    _test_splits(labels.size, splits)
    _test_balancing(labels, splits)
        
def test_rseed():
    """Test folds random generation."""
    random_state = np.random.RandomState(0)
    X = random_state.randn(10, 100)
    y = random_state.randn(10)
    
    splits1 = list(KFold(len(y), 2, random_state=10))
    splits2 = list(KFold(len(y), 2, random_state=10))
    splits3 = list(KFold(len(y), 2, random_state=1))
    assert_equal(splits1, splits2)
    assert_not_equal(splits1, splits3)
    assert_not_equal(splits2, splits3)

    labels = np.ones(100)
    negative = np.arange(0, labels.size, 2)
    labels[negative] = -1
    splits1 = list(StratifiedKFold(labels, 2, random_state=10))
    splits2 = list(StratifiedKFold(labels, 2, random_state=10))
    splits3 = list(StratifiedKFold(labels, 2, random_state=1))
    assert_equal(splits1, splits2)
    assert_not_equal(splits1, splits3)
    assert_not_equal(splits2, splits3)
    
def test_random_state_client_code():
    """Test random state from client code."""
    labels = np.ones(10)
    labels[0:5] = -1
    
    # Generate random numbers before and after splits generation
    random_state = np.random.RandomState(10)
    indexes_1a = [random_state.randint(100) for i in range(10)]
    list(KFold(len(labels), 2)) # internal random state
    indexes_2a = [random_state.randint(100) for i in range(10)]
    assert_not_equal(indexes_1a, indexes_2a)
    
    # Generate random numbers before and after splits generation
    # using client random_state
    random_state = np.random.RandomState(10)
    indexes_1b = [random_state.randint(100) for i in range(10)]
    list(KFold(len(labels), 2, random_state)) # client random state
    indexes_2b = [random_state.randint(100) for i in range(10)]
    assert_not_equal(indexes_1b, indexes_2b)
    
    assert_equal(indexes_1a, indexes_1b)
    assert_not_equal(indexes_2a, indexes_2b)
    
def test_exceptions():
    """Test KFold exceptions."""
    random_state = np.random.RandomState(0)
    X = random_state.randn(10, 100)
    y = random_state.randn(10)
    
    assert_raises(ValueError, list, KFold(y, 0))    
    assert_raises(ValueError, list, KFold(y, len(y) + 1))

    # Not two class
    assert_raises(ValueError, list, StratifiedKFold(y, 2))

    labels = np.ones(100)
    labels[::2] = -1

    # k, out of range
    assert_raises(ValueError, list, StratifiedKFold(labels, 1))
    assert_raises(ValueError, list, StratifiedKFold(labels, 51))

    # More negatives (75 vs 25)
    labels[:50] = -1
    assert_raises(ValueError, list, StratifiedKFold(labels, 26))
    
        
def _test_splits(labels_size, splits):
    cum_train = list()
    cum_test = list()

    for train, test in splits:
        assert_equal(labels_size, len(train + test))
        assert_true(set(test).isdisjoint(set(train)))

        cum_train.extend(train)
        cum_test.extend(test)

    assert_equal(set(cum_train), set(cum_test))
    
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