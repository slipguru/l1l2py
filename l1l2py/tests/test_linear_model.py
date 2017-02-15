# This code is written by Salvatore Masecchia <salvatore.masecchia@unige.it>
# and Annalisa Barla <annalisa.barla@unige.it>
# Copyright (C) 2010 SlipGURU -
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
# from nose.plugins.attrib import attr

from l1l2py.linear_model import L1L2
from l1l2py.tests import _TEST_DATA_PATH

class TestLinearModel(object):

    def setup(self):
        data = np.loadtxt(_TEST_DATA_PATH)
        self.X = data[:, :-1]
        self.Y = data[:, -1]

    def test_data(self):
        assert_equals((30, 40), self.X.shape)
        assert_equals((30, ), self.Y.shape)

    def test_l1l2(self):
        coef_ = L1L2(mu=.5, tau=1.0).fit(self.X, self.Y).coef_

        true_coef = np.array(
            [ 2.54847672,  2.5712637 ,  2.56824943,  2.56879183,  2.55001544,
              2.49877257,  2.49117123,  2.49292074,  2.48615372,  2.49009366,
              2.39617356,  2.4019562 ,  2.39986878,  2.37417205,  2.42213084,
              0.        , -0.        , -0.        , -0.        ,  0.06651892,
              0.0229522 , -0.1862063 ,  0.40750036, -0.        , -0.06511224,
              -0.32570505, -0.37182631, -0.74583084,  0.        , -0.       ,
              0.        , -0.48883576, -0.36477394, -0.        ,  0.        ,
              -0.        ,  0.        , -0.        , -0.        ,  0.31458035])
        assert_true(np.allclose(true_coef, coef_))

        coef_ = L1L2(mu=0, tau=1.0).fit(self.X, self.Y).coef_

        true_coef = np.array(
            [  0.        ,  10.93683418,   3.46579585,   0.        ,
             0.        ,   7.13565392,   0.        ,   7.37737905,
             0.        ,   0.        ,   0.        ,   0.        ,
             0.        ,   0.        ,  14.36854962,  -0.        ,
            -0.        ,  -0.        ,  -0.        ,   0.        ,
             0.        ,  -0.        ,   0.        ,  -0.        ,
            -0.        ,  -0.        ,  -0.02985752,  -0.        ,
            -0.        ,  -0.        ,   0.        ,  -0.        ,
            -0.        ,  -0.        ,   0.        ,  -0.        ,
            -0.        ,  -0.        ,  -0.        ,   0.        ])
        assert_true(np.allclose(true_coef, coef_))

    def test_conversion_params(self):
        coef_0 = L1L2(mu=0.5, tau=1.0).fit(self.X, self.Y).coef_
        coef_1 = L1L2(l1_ratio=0.5, alpha=1.0).fit(self.X, self.Y).coef_
        assert_true(np.allclose(coef_0, coef_1))

        coef_0 = L1L2(mu=1, tau=0).fit(self.X, self.Y).coef_
        coef_1 = L1L2(l1_ratio=0, alpha=1.0).fit(self.X, self.Y).coef_
        assert_true(np.allclose(coef_0, coef_1))

        coef_0 = L1L2(mu=0, tau=1).fit(self.X, self.Y).coef_
        coef_1 = L1L2(l1_ratio=1, alpha=0.5).fit(self.X, self.Y).coef_
        assert_true(np.allclose(coef_0, coef_1))
