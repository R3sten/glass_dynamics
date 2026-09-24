"""
Supervised Principal Component Regression (SPCR) Module.

Extracts components and selects the top P subspace based on absolute 
Pearson correlation with the target variable. Includes a variance threshold 
guardrail to prevent catastrophic overfitting on near-zero eigenvalue components.
"""

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import pearsonr
import numpy as np
from glass_dynamics.core.logger import logger

class CustomSupervisedPCR(BaseEstimator, RegressorMixin):
    """
    Target-specific Supervised Principal Component Regression with 
    variance-based noise filtering.
    """
    
    def __init__(self, n_components=10, variance_threshold=1e-4, fit_intercept=True):
        """
        Parameters
        ----------
        n_components : int, default=10
            Number of correlated principal components to keep.
        variance_threshold : float, default=1e-4
            Minimum explained variance ratio required for a PC to be considered.
            Prevents OLS weight explosion on noise components.
        fit_intercept : bool, default=True
            Whether to calculate the intercept for this model.
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.fit_intercept = fit_intercept
        
        self.pca_ = None
        self.regressor_ = None
        self.coef_ = None
        self.intercept_ = None
        self.selected_pc_indices_ = None
        self.correlations_ = None

    def fit(self, X, y):
        """
        Fits the Supervised PCA and the linear regressor.
        """
        y_1d = np.asarray(y).ravel()
        
        # 1. Unsupervised Extraction
        self.pca_ = PCA()
        X_pca_full = self.pca_.fit_transform(X)
        variances = self.pca_.explained_variance_ratio_
        
        # 2. Supervised Evaluation with Variance Guardrail
        correlations = []
        valid_indices = []
        
        for i in range(X_pca_full.shape[1]):
            # Guardrail: Only consider components with meaningful structural variance
            if variances[i] > self.variance_threshold:
                pc_col = X_pca_full[:, i]
                if np.std(pc_col) < 1e-12:
                    corr = 0.0
                else:
                    corr = abs(pearsonr(pc_col, y_1d)[0])
                    
                correlations.append(corr)
                valid_indices.append(i)
                
        self.correlations_ = np.array(correlations)
        valid_indices = np.array(valid_indices)
        
        # 3. Supervised Selection
        # If n_components is larger than valid components, cap it
        n_to_select = min(self.n_components, len(valid_indices))
        
        # Sort valid components by correlation (descending)
        sorted_relative_indices = np.argsort(self.correlations_)[::-1][:n_to_select]
        
        # Map back to the original PCA indices
        self.selected_pc_indices_ = valid_indices[sorted_relative_indices]
        
        X_pca_selected = X_pca_full[:, self.selected_pc_indices_]
        
        # 4. Linear Regression
        self.regressor_ = LinearRegression(fit_intercept=self.fit_intercept)
        self.regressor_.fit(X_pca_selected, y_1d)
        
        # 5. Project coefficients back to original feature space
        self.coef_ = self.regressor_.coef_ @ self.pca_.components_[self.selected_pc_indices_, :]
        self.intercept_ = self.regressor_.intercept_
        
        return self

    def predict(self, X):
        X_pca_full = self.pca_.transform(X)
        X_pca_selected = X_pca_full[:, self.selected_pc_indices_]
        return self.regressor_.predict(X_pca_selected)