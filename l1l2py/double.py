import numpy as np

from .estimators import LinearModel

class DoubleStepEstimator(LinearModel):
    def __init__(self, selector, regressor, threshold=1e-10):
        self.selector = selector
        self.regressor = regressor
        self.threshold = threshold
        self.coef_ = None
        self.intercept_ = None
        self.selected_ = None
    
    def fit(self, X, y):
        # Selection        
        self.selector.fit(X, y)
        self.selected_ = (np.abs(self.selector.coef_) >= self.threshold)
        
        # Final Estimation
        self.regressor.fit(X[:, self.selected_], y)
        
        # Coefficients
        self.coef_ = np.zeros_like(self.selector.coef_)
        self.coef_[self.selected_] = self.regressor.coef_
        self.intercept_ = self.regressor.intercept_
        
        return self

        
        
        
