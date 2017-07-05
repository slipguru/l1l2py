import numpy as np

from .base import AbstractLinearModel

# TODO!!
class DoubleStepEstimator(AbstractLinearModel):
    def __init__(self, selector, estimator, threshold=1e-10):
        self.selector = selector
        self.estimator = estimator
        self.threshold = threshold
        self.fit_intercept = estimator.fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.selected_ = None
        # Intercept depends on 'estimator' params
    
    def fit(self, X, y):
        X = np.asanyarray(X)
        y = np.asanyarray(y)
        
        # Selection        
        self.selector.fit(X, y)
        self.selected_ = (np.abs(self.selector.coef_) >= self.threshold)
        
        # Final Estimation
        self.estimator.fit(X[:, self.selected_], y)
        
        # Coefficients
        self.coef_ = np.zeros_like(self.selector.coef_)
        self.coef_[self.selected_] = self.estimator.coef_
        self.intercept_ = self.estimator.intercept_
        
        return self
    
class DoubleStepEstimatorCV(AbstractLinearModel):
    pass

        
        
        
