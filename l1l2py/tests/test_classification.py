"""Testing for linear_model.py."""

# This code is written by
#       Federico Tomasi <federico.tomasi@dibris.unige.it>
# Copyright (C) 2017 SlipGURU -
# Statistical Learning and Image Processing Genoa University Research Group
# Via Dodecaneso, 35 - 16146 Genova, ITALY.
#
# This file is part of L1L2Py.
#
# L1L2Py is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# L1L2Py is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with L1L2Py. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from nose.tools import assert_equals, assert_raises, assert_true

from l1l2py.classification import L1L2Classifier
from l1l2py.classification import L1L2StageOneClassifier
from l1l2py.classification import L1L2StageTwoClassifier
from l1l2py.tests import _TEST_DATA_PATH

class TestClassification(object):

    def setup(self):
        data = np.loadtxt(_TEST_DATA_PATH)
        self.X = data[:, :-1]
        self.Y = np.sign(data[:, -1])

    def test_data(self):
        assert_equals((30, 40), self.X.shape)
        assert_equals((30, ), self.Y.shape)

    # def test_l1l2(self):
    #     coef_ = L1L2Classifier(mu=.5, tau=1.0).fit(self.X, self.Y).coef_
    #
    #     true_coef = np.array([
    #         2.54916703,  2.57154267,  2.56869228,  2.56931109,  2.55051137,
    #         2.49967827,  2.49216303,  2.49392693,  2.48734147,  2.4912053 ,
    #         2.39791057,  2.4035761 ,  2.40152707,  2.37603518,  2.42354404,
    #         0.        , -0.        , -0.        , -0.        ,  0.0862872 ,
    #         0.03008878, -0.1937291 ,  0.42276159, -0.        , -0.0768753 ,
    #        -0.33118946, -0.37828447, -0.74693446,  0.        , -0.        ,
    #         0.        , -0.49657344, -0.37692851, -0.        ,  0.        ,
    #        -0.        ,  0.        , -0.        , -0.        ,  0.32717314])
    #
    #     assert_true(np.allclose(true_coef, coef_))
    #
    #     coef_ = L1L2(mu=0, tau=1.0).fit(self.X, self.Y).coef_
    #
    #     true_coef = np.array([
    #          0.        ,  10.93683418,   3.46579585,   0.        ,
    #          0.        ,   7.13565392,   0.        ,   7.37737905,
    #          0.        ,   0.        ,   0.        ,   0.        ,
    #          0.        ,   0.        ,  14.36854962,  -0.        ,
    #         -0.        ,  -0.        ,  -0.        ,   0.        ,
    #          0.        ,  -0.        ,   0.        ,  -0.        ,
    #         -0.        ,  -0.        ,  -0.02985752,  -0.        ,
    #         -0.        ,  -0.        ,   0.        ,  -0.        ,
    #         -0.        ,  -0.        ,   0.        ,  -0.        ,
    #         -0.        ,  -0.        ,  -0.        ,   0.        ])
    #
    #     assert_true(np.allclose(true_coef, coef_))

    def test_conversion_params(self):
        coef_0 = L1L2Classifier(mu=0.5, tau=1.0).fit(self.X, self.Y).coef_
        coef_1 = L1L2Classifier(l1_ratio=0.5, alpha=1.0).fit(self.X, self.Y).coef_
        assert_true(np.allclose(coef_0, coef_1))

        coef_0 = L1L2Classifier(mu=1, tau=0).fit(self.X, self.Y).coef_
        coef_1 = L1L2Classifier(l1_ratio=0, alpha=1.0).fit(self.X, self.Y).coef_
        assert_true(np.allclose(coef_0, coef_1))

        coef_0 = L1L2Classifier(mu=0, tau=1).fit(self.X, self.Y).coef_
        coef_1 = L1L2Classifier(l1_ratio=1, alpha=0.5).fit(self.X, self.Y).coef_
        assert_true(np.allclose(coef_0, coef_1))

    def test_stage_two(self):
        mdl = L1L2StageTwoClassifier(None)
        assert_raises(TypeError, mdl.fit, None, None)

        coefs = L1L2StageTwoClassifier(
            L1L2StageOneClassifier(error_score=-1), mus=(.1, 1, 10, 100)
        ).fit(self.X, self.Y, sample_weight=1., check_input=True).coef_
        for i in range(1, len(coefs)):
            assert_true(np.sum(coefs[i - 1] != 0) <= np.sum(coefs[i] != 0))
