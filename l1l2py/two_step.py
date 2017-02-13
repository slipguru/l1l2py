"""Wrapper for l1l2 two step."""
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
