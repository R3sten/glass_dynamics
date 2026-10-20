"""
Principal Component Regression (PCR) Module.

Combines Principal Component Analysis (PCA) for dimensionality reduction 
with Ordinary Least Squares (OLS) regression. This completely eliminates 
multicollinearity by transforming features into orthogonal components before regression.
"""

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class CustomPCR(BaseEstimator, RegressorMixin):
    """
    Custom implementation of Principal Component Regression.
    Projects the data onto a lower-dimensional orthogonal space and performs regression.
    It automatically projects the resulting coefficients back to the original feature 
    space for interpretability.
    """
    
    def __init__(self, n_components=10, fit_intercept=True):
        """
        Parameters
        ----------
        n_components : int or float, default=10
            Number of principal components to keep. If float between 0 and 1, 
            it represents the variance explained threshold.
        fit_intercept : bool, default=True
            Whether to calculate the intercept for this model.
        """
        self.n_components = n_components
        self.fit_intercept = fit_intercept
        
        self.pca_ = None
        self.regressor_ = None
        self.coef_ = None
        self.intercept_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X, y):
        """
        Fits the PCA and the linear regressor.
        """
        # 1. Dimensionality Reduction
        self.pca_ = PCA(n_components=self.n_components)
        X_pca = self.pca_.fit_transform(X)
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        
        # 2. Linear Regression on orthogonal components
        self.regressor_ = LinearRegression(fit_intercept=self.fit_intercept)
        self.regressor_.fit(X_pca, y)
        
        
        # Salviamo i pesi originali grezzi (affetti dallo scaling di sklearn)
        self.pc_weights_raw_ = self.regressor_.coef_

        # 2. Estrazione dei pesi grezzi. 
        # NON usiamo ravel(), così manteniamo la forma (n_targets, n_components)
        self.pc_weights_raw_ = self.regressor_.coef_

        # 3. Pesi normalizzati (w_tilde = w_raw * sqrt(lambda))
        selected_lambdas = self.pca_.explained_variance_[:self.n_components]
        # NumPy moltiplica automaticamente ogni riga della matrice per i 14 lambda (broadcasting)
        self.pc_weights_ = self.pc_weights_raw_ * np.sqrt(selected_lambdas)

        # 4. Retro-proiezione sulle feature originali
        # Usiamo lo slicing [:self.n_components] invece di selected_indices_
        selected_components = self.pca_.components_[:self.n_components, :]
        
        # Gestione sicura del prodotto scalare (multi-target vs single-target)
        if self.pc_weights_raw_.ndim > 1:
            # Multi-target: (3, 14) @ (14, 560) -> (3, 560)
            self.coef_ = self.pc_weights_raw_ @ selected_components
        else:
            # Single-target: (14, 560).T @ (14,) -> (560,)
            self.coef_ = selected_components.T @ self.pc_weights_raw_

        if self.fit_intercept:
            self.intercept_ = self.regressor_.intercept_
            
        return self

    def predict(self, X):
        """
        Transforms new data via PCA and predicts targets.
        """
        X_pca = self.pca_.transform(X)
        return self.regressor_.predict(X_pca)