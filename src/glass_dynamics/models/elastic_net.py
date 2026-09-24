"""
Elastic Net Regression Module.

Implements Elastic Net regression, combining L1 (Lasso) and L2 (Ridge) penalties.
This model is ideal for high-dimensional, collinear datasets where feature selection 
(sparsity) is desired alongside regularization, identifying a subset of the most 
important physical descriptors.
"""

from sklearn.linear_model import ElasticNet
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class CustomElasticNet(BaseEstimator, RegressorMixin):
    """
    Custom wrapper for Elastic Net Regression.
    Maintains compatibility with scikit-learn pipelines and exposes 
    coefficients for physical interpretability analysis.
    """
    
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, max_iter=10000, random_state=42):
        """
        Parameters
        ----------
        alpha : float, default=1.0
            Constant that multiplies the penalty terms.
        l1_ratio : float, default=0.5
            The ElasticNet mixing parameter (0 <= l1_ratio <= 1).
            l1_ratio=0 corresponds to Ridge, l1_ratio=1 to Lasso.
        fit_intercept : bool, default=True
            Whether to calculate the intercept for this model.
        max_iter : int, default=10000
            The maximum number of iterations.
        random_state : int, default=42
            Seed for reproducible results.
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.coef_ = None
        self.intercept_ = None
        self.model_ = None

    def fit(self, X, y):
        """
        Fits the Elastic Net model.
        """
        self.model_ = ElasticNet(
            alpha=self.alpha, 
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.model_.fit(X, y)
        
        # Expose coefficients identically to CustomRidgeRegression
        self.coef_ = self.model_.coef_
        self.intercept_ = self.model_.intercept_
        
        return self

    def predict(self, X):
        """
        Predicts targets using the fitted model.
        """
        return self.model_.predict(X)