import numpy as np
import scipy.io as sio

from nose.tools import *
from nose.plugins.attrib import attr

from l1l2py._core import *

class TestCore(object):
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

    def test_minimum_selection(self):
        from l1l2py._core import _minimum_selection
        range1 = (0, 0, 1, 1)
        range2 = (0, 1, 0, 1)

        # default: NOT sparse and NOT regularized
        out = _minimum_selection(range1, range2)
        assert_equals((0, 0), out)

        # sparse and NOT regularized
        out = _minimum_selection(range1, range2, sparse=True)
        assert_equals((1, 0), out)

        # NOT sparse and regularized
        out = _minimum_selection(range1, range2, regularized=True)
        assert_equals((0, 1), out)

        # sparse and regularized
        out = _minimum_selection(range1, range2, sparse=True, regularized=True)
        assert_equals((1, 1), out)

    def test_model_selection(self):
        from l1l2py import tools
        splits = tools.kfold_splits(self.Y, 2)
        tr_idx, ts_idx = splits[0]
        data, test_data = self.X[tr_idx, :], self.X[ts_idx, :]
        labels, test_labels = self.Y[tr_idx, :], self.Y[ts_idx, :]

        tau_range = tools.linear_range(0.1, 1.0, 5)
        lambda_range = tools.linear_range(0.1, 1.0, 6)
        mu_range = tools.linear_range(0.1, 1.0, 10)
        int_splits = tools.kfold_splits(labels, 3)

        out = model_selection(data, labels, test_data, test_labels,
                              mu_range, tau_range, lambda_range,
                              int_splits,
                              tools.regression_error,
                              tools.regression_error,
                              data_normalizer=tools.standardize,
                              labels_normalizer=tools.center)
        assert_equals(8, len(out))

        assert_equals((len(tau_range), len(lambda_range)), out['kcv_err_ts'].shape)
        assert_equals(out['kcv_err_ts'].shape, out['kcv_err_ts'].shape)

        assert_equals(len(mu_range), len(out['beta_list']))
        assert_equals(len(mu_range), len(out['selected_list']))
        assert_equals(len(mu_range), len(out['err_ts_list']))
        assert_equals(len(mu_range), len(out['err_tr_list']))

        # Predictions
        out = model_selection(data, labels, test_data, test_labels,
                      mu_range, tau_range, lambda_range,
                      int_splits,
                      tools.regression_error,
                      tools.regression_error,
                      data_normalizer=tools.standardize,
                      labels_normalizer=tools.center,
                      return_predictions=True)
        assert_equals(10, len(out))

        assert_equals(len(mu_range), len(out['prediction_ts_list']))
        assert_equals(len(mu_range), len(out['prediction_tr_list']))

        for p in out['prediction_ts_list']:
            assert_equals(len(test_labels), len(p))
        for p in out['prediction_tr_list']:
            assert_equals(len(labels), len(p))

    def test_minimal_model(self):
        from l1l2py import tools
        splits = tools.kfold_splits(self.Y, 2)

        tau_range = tools.linear_range(0.1, 1.0, 5)
        lambda_range = tools.linear_range(0.1, 1.0, 5)

        for mu in tools.linear_range(0.1, 1.0, 10):
            out = minimal_model(self.X, self.Y, mu, tau_range, lambda_range,
                                splits, error_function=tools.regression_error,
                                data_normalizer=tools.standardize,
                                labels_normalizer=tools.center)
            assert_equals(2, len(out))

            out = minimal_model(self.X, self.Y, mu, tau_range, lambda_range,
                                splits, error_function=tools.regression_error,
                                data_normalizer=tools.standardize,
                                labels_normalizer=tools.center)
            assert_equals(2, len(out))

            (kcv_err_ts, kcv_err_tr) = out

            assert_equals((len(tau_range), len(lambda_range)), kcv_err_ts.shape)
            assert_equals(kcv_err_tr.shape, kcv_err_ts.shape)

    def test_minimal_model_saturated(self):
        from l1l2py import tools
        splits = tools.kfold_splits(self.Y, 2)

        tau_range = [0.1, 1e3, 1e4]
        lambda_range = tools.linear_range(0.1, 1.0, 5)

        for mu in tools.linear_range(0.1, 1.0, 10):
            out = minimal_model(self.X, self.Y, mu, tau_range, lambda_range,
                                splits, error_function=tools.regression_error,
                                data_normalizer=tools.standardize,
                                labels_normalizer=tools.center)

            (kcv_err_ts, kcv_err_tr) = out

            assert_equals((1, len(lambda_range)), kcv_err_ts.shape)
            assert_equals(kcv_err_tr.shape, kcv_err_ts.shape)

    def test_nested_models(self):
        from l1l2py import tools
        splits = tools.kfold_splits(self.Y, 2)
        tr_idx, ts_idx = splits[0]
        data, test_data = self.X[tr_idx, :], self.X[ts_idx, :]
        labels, test_labels = self.Y[tr_idx, :], self.Y[ts_idx, :]

        tau_opt, lambda_opt = (0.1, 0.1)

        mu_range = tools.linear_range(0.1, 1.0, 10)
        out = nested_models(data, labels, test_data, test_labels,
                            mu_range, tau_opt, lambda_opt,
                            error_function=tools.regression_error,
                            data_normalizer=tools.standardize,
                            labels_normalizer=tools.center)
        assert_equals(4, len(out))

        (beta_list, selected_list, err_tr_list, err_ts_list) = out

        assert_equals(len(mu_range), len(beta_list))
        assert_equals(len(mu_range), len(selected_list))
        assert_equals(len(mu_range), len(err_ts_list))
        assert_equals(len(mu_range), len(err_tr_list))

        for b, s in zip(beta_list, selected_list):
            assert_true(len(b) == len(s[s]))

        for i in xrange(1, len(mu_range)):
            s_prev = selected_list[i-1]
            s = selected_list[i]

            assert_true(len(s_prev[s_prev]) <= len(s[s]))

    def test_nested_models_predictions(self):
        from l1l2py import tools
        splits = tools.kfold_splits(self.Y, 2)
        tr_idx, ts_idx = splits[0]
        data, test_data = self.X[tr_idx, :], self.X[ts_idx, :]
        labels, test_labels = self.Y[tr_idx, :], self.Y[ts_idx, :]

        tau_opt, lambda_opt = (0.1, 0.1)
        mu_range = tools.linear_range(0.1, 1.0, 10)
        out = nested_models(data, labels, test_data, test_labels,
                            mu_range, tau_opt, lambda_opt,
                            error_function=tools.regression_error,
                            data_normalizer=tools.standardize,
                            labels_normalizer=tools.center,
                            return_predictions=True)
        assert_equals(6, len(out))

    def test_nested_model_void(self):
        from l1l2py import tools
        data, test_data = np.vsplit(self.X, 2)
        labels, test_labels = np.vsplit(self.Y, 2)

        tau_opt, lambda_opt = (50.0, 0.1)
        mu_range = tools.linear_range(0.1, 1.0, 10)

        assert_raises(ValueError, nested_models,
                                  data, labels, test_data, test_labels,
                                  mu_range, tau_opt, lambda_opt,
                                  error_function=tools.regression_error,
                                  data_normalizer=tools.standardize,
                                  labels_normalizer=tools.center)
