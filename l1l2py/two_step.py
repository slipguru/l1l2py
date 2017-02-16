"""Wrapper for l1l2 two step."""

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

from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier
from sklearn.utils.validation import check_is_fitted

from l1l2py.classification import L1L2Classifier


class L1L2TwoSteps(Pipeline):
    def __init__(self, mu=.5, tau=1, lamda=1):
        """INIT DOC."""
        vs = L1L2Classifier(mu=mu, tau=tau)
        clf = RidgeClassifier(alpha=lamda)
        super(L1L2TwoSteps, self).__init__(
            (('l1l2_vs', vs), ('ridge_clf', clf)))

        self.mu = mu
        self.tau = tau
        self.lamda = lamda

    @property
    def coef_(self):
        check_is_fitted(self.steps[1][1], "coef_")
        return self.steps[1][1].coef_
