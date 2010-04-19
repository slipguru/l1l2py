import numpy as np
import scipy.io as sio

from nose.tools import *

from biolearning.tools import *

class TestDataTools(object):
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
        exp_geom = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
        geom = geometric_range(1e-3, 1e3, 7)

        assert_true(np.allclose(exp_geom, geom))

    def test_linear_ranges(self):
        exp_lin = [ 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        lin = linear_range(0.0, 1.0, 11)

        assert_true(np.allclose(exp_lin, lin))

    def test_centering(self):
        Ycent, mean = center(self.Y, return_mean=True)
        assert_true(np.allclose(self.Y - mean, Ycent))
        assert_true(np.allclose(Ycent + mean, self.Y))

        Ycent2, mean2 = center(Ycent, return_mean=True)
        assert_true(np.allclose(Ycent, Ycent2))
        assert_true(np.allclose(np.zeros(len(Ycent)), mean2))

        Ycent, Ycent2 = center(self.Y, self.Y)
        assert_true(np.allclose(Ycent, Ycent2))

    def test_centering_outputs(self):
        assert_equals(np.ndarray, type(center(self.Y)))
        assert_equals(2, len(center(self.Y, self.Y)))
        assert_equals(2, len(center(self.Y, return_mean=True)))
        assert_equals(3, len(center(self.Y, self.Y, return_mean=True)))

    def test_standardization(self):
        # Note: standardization includes matrix centering!
        Xstd, mean, std = standardize(self.X, return_factors=True)
        assert_true(np.allclose((self.X - mean)/std, Xstd))
        assert_true(np.allclose((Xstd * std) + mean, self.X))

        Xstd2, mean2, std2 = standardize(Xstd, return_factors=True)
        assert_true(np.allclose(Xstd, Xstd2))
        assert_true(np.allclose(np.zeros(Xstd.shape[1]), mean2))
        assert_true(np.allclose(np.ones(Xstd.shape[1]), std2))

        Xstd, Xstd2 = standardize(self.X, self.X)
        assert_true(np.allclose(Xstd, Xstd2))

    def test_standardization_outputs(self):
        assert_equals(np.ndarray, type(standardize(self.X)))
        assert_equals(2, len(standardize(self.X, self.X)))
        assert_equals(3, len(standardize(self.X, return_factors=True)))
        assert_equals(4, len(standardize(self.X, self.X, return_factors=True)))
