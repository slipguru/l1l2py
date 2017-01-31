"""Wrapper for l1l2 two step."""
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier

from l1l2py.classification import L1L2Classifier


class L1L2TwoSteps(Pipeline):
    # def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
    #              normalize=False, precompute=False, max_iter=1000,
    #              copy_X=True, tol=1e-4, warm_start=False, positive=False,
    #              random_state=None, selection='cyclic'):
    def __init__(self, mu=.5, tau=1, lamda=1):
        """INIT DOC."""
        vs = L1L2Classifier(mu=mu, tau=tau)
        clf = RidgeClassifier(alpha=lamda)
        super(L1L2TwoSteps, self).__init__(
            (('l1l2_vs', vs), ('ridge_clf', clf)))

        self.mu = mu
        self.tau = tau
        self.lamda = lamda
        # self.fit_intercept = fit_intercept
        # self.use_gpu = use_gpu
        # self.max_iter = max_iter
        # self.tol = tol
        # # self.path = l1l2_regularization
        # self.coef_ = None
        # self.alpha = alpha
        # self.l1_ratio = l1_ratio
        #
        # if l1_ratio is not None and alpha is not None:
        #     # tau and mu are selected as enet
        #     if l1_ratio == 1:
        #         self.mu = 0
        #         self.tau = 2 * alpha
        #     elif l1_ratio == 0:
        #         self.mu = 2 * alpha
        #         self.tau = 0
        #     else:
        #         self.mu = 2 * alpha * (1 - l1_ratio)
        #         self.tau = 2 * alpha * l1_ratio
        # else:
        #     self.l1_ratio = self.tau / (self.tau + self.mu)
        #     self.alpha = self.tau * .5 / self.l1_ratio
