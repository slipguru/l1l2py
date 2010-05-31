import numpy as np

from nose.tools import *
from nose.plugins.attrib import attr

from l1l2py.algorithms import *
from l1l2py.algorithms import _soft_thresholding

import os
data_path = os.path.join(os.path.dirname(__file__), 'data.txt')

class TestAlgorithms(object):
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
                                           return_iterations=True)
            assert_equal(beta.shape, (self.X.shape[1], 1))

            beta, k2 = l1l2_regularization(self.X, self.Y, mu, tau,
                                           tolerance=1e-3,
                                           return_iterations=True)
            assert_true(k2 <= k1)

            beta, k3 = l1l2_regularization(self.X, self.Y, mu, tau,
                                           tolerance=1e-3, kmax=100,
                                           return_iterations=True)
            assert_true(k3 <= k2)
            assert_true(k3 == 100)

            beta1, k1 = l1l2_regularization(self.X, self.Y, mu, tau,
                                            return_iterations=True)
            beta2, k2 = l1l2_regularization(self.X, self.Y, mu, tau,
                                            beta=beta1.squeeze(),
                                            return_iterations=True)
            assert_true(k2 <= k1)

            beta1, k1 = l1l2_regularization(self.X, self.Y, mu, tau,
                                            return_iterations=True)
            beta2, k2 = l1l2_regularization(self.X, self.Y.squeeze(), mu, tau,
                                            return_iterations=True)
            assert_equal(k1, k2)
            assert_true(beta1.shape, beta2.shape)
            assert_true(np.allclose(beta1, beta2))

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

    def test_l1_bound(self):
        tau_max = l1_bound(self.X, self.Y)

        beta = l1l2_regularization(self.X, self.Y, 0.0, tau_max)
        assert_equals(0, len(beta.nonzero()[0]))

        beta = l1l2_regularization(self.X, self.Y, 0.0, tau_max-1e-5)
        assert_equals(1, len(beta.nonzero()[0]))

    def test_soft_thresholding(self):
        beta = ridge_regression(self.X, self.Y)
        for th in np.linspace(0.0, 10.0, 50):
            out = _soft_thresholding(beta, th)

            # Verbose and slow version
            out_exp = beta - (np.sign(beta) * th/2.0)
            out_exp[np.abs(beta) <= th/2.0] = 0.0

            assert_true(np.allclose(out, out_exp))
