import numpy as np
import scipy.io as sio

from nose.tools import *
from nose.plugins.attrib import attr

from biolearning.algorithms import *

class TestAlgorithms(object):
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

    def test_rls(self):
        # case n >= d
        for penalty in np.linspace(0.0, 1.0, 5):
            value = ridge_regression(self.X, self.Y, penalty)
            assert_equal(value.shape, (self.X.shape[1], 1))

        expected = ridge_regression(self.X, self.Y, 0.0)
        value = ridge_regression(self.X, self.Y)
        assert_true(np.allclose(expected, value))

        # case d > n
        X = self.X.T
        Y = self.X[0:1,:].T
        for penalty in np.linspace(0.0, 1.0, 5):
            value = ridge_regression(X, Y, penalty)
            assert_equal(value.shape, (X.shape[1], 1))

        expected = ridge_regression(X, Y, 0.0)
        value = ridge_regression(X, Y)
        assert_true(np.allclose(expected, value))

    def test_l1l2_regularization(self):
        from itertools import product
        n, m = self.X.shape

        values = np.linspace(0.1, 1.0, 5)
        for mu, tau in product(values, values):
            beta, k1 = l1l2_regularization(self.X, self.Y, mu, tau,
                                           returns_iterations=True)
            assert_equal(beta.shape, (self.X.shape[1], 1))

            beta, k2 = l1l2_regularization(self.X, self.Y, mu, tau,
                                           tolerance=1e-3,
                                           returns_iterations=True)
            assert_true(k2 < k1)

            beta, k3 = l1l2_regularization(self.X, self.Y, mu, tau,
                                           tolerance=1e-3, kmax=10,
                                           returns_iterations=True)
            assert_true(k3 < k2)
            assert_true(k3 == 10)

            beta1, k1 = l1l2_regularization(self.X, self.Y, mu, tau,
                                            returns_iterations=True)
            beta2, k2 = l1l2_regularization(self.X, self.Y, mu, tau,
                                            beta=beta,
                                            returns_iterations=True)
            assert_true(k2 < k1)

    def test_l1l2_path(self):
        values = np.linspace(0.1, 1.0, 5)
        beta_path = l1l2_path(self.X, self.Y, 0.1, values)

        assert_true(len(beta_path) <= len(values))
        for i in xrange(1, len(beta_path)):

            b = beta_path[i]
            b_prev = beta_path[i-1]

            selected_prev = len(b_prev[b_prev != 0.0])
            selected = len(b[b != 0.0])
            assert_true(selected <= selected_prev)

    def test_l1l2_path_saturation(self):
        values = [0.1, 1e1, 1e3, 1e4]
        beta_path = l1l2_path(self.X, self.Y, 0.1, values)
        assert_equals(len(beta_path), 2)

        for i in xrange(2):
            b = beta_path[i]
            selected = len(b[b != 0.0])

            assert_true(selected <= len(b))
