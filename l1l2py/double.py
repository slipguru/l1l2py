import numpy as np

from .estimators import LinearModel

class DoubleStepEstimator(LinearModel):
    def __init__(self, selector, regressor):
        self.selector = selector
        self.regressor = regressor
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        # Selection        
        self.selector.fit(X, y)
        X = self.selector.transform(X)
        
        # Final Estimation
        self.regressor.fit(X, y)
        
        # Coefficients
        self.coef_ = np.zeros_like(self.selector.coef_)
        self.coef_[self.selector.selected_] = self.regressor.coef_
        self.intercept_ = self.regressor.intercept_

        
        
        
