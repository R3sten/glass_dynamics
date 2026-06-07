"""
Custom Ridge Regression Model.

This module implements a mathematically explicit Ridge Regression from scratch,
strictly following the objective function used in physics and statistical literature:
    Loss = (1/N) * ||Y - XW||^2 + alpha * ||W||^2

It natively supports multi-target regression, computes exact effective 
degrees of freedom, and estimates standard errors for both coefficients and predictions.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
from sklearn.base import BaseEstimator
from glass_dynamics.core.logger import logger

class CustomRidgeRegression(BaseEstimator):
    """
    A transparent, mathematically explicit implementation of Ridge Regression.
    Inherits from BaseEstimator to be fully compatible with scikit-learn utilities
    
    Attributes
    ----------
    alpha : float
        The regularization parameter as defined in standard literature.
    fit_intercept : bool
        Whether to center the data and calculate an intercept.
    coef_ : np.ndarray
        The fitted weights matrix of shape (n_features, n_targets).
    intercept_ : np.ndarray
        The fitted intercept vector of shape (n_targets,).
    coef_se_ : np.ndarray
        The standard errors of the coefficients (n_features, n_targets).
    sigma_squared_ : np.ndarray
        The estimated variance of the residuals for each target (n_targets,).
    condition_number_ : float
        The condition number of the regularized covariance matrix.
    effective_df_ : float
        The effective degrees of freedom of the model.
    """

    def __init__(self, alpha: float = 0.1, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        
        # Initializing model artifacts
        self.coef_ = None
        self.intercept_ = None
        self.coef_se_ = None
        self.sigma_squared_ = None
        self.condition_number_ = None
        self.effective_df_ = None

    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        Y: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> 'CustomRidgeRegression':
        """
        Fit the Ridge regression model and compute statistical uncertainties.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Training feature matrix of shape (N, P).
        Y : Union[pd.DataFrame, pd.Series, np.ndarray]
            Training target matrix of shape (N, T) for multi-target, or (N,) for single.

        Returns
        -------
        self : CustomRidgeRegression
            The fitted model instance.
        """
        # Ensure numpy arrays for matrix operations
        X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        Y_mat = Y.to_numpy() if isinstance(Y, (pd.DataFrame, pd.Series)) else np.asarray(Y)
        
        if Y_mat.ndim == 1:
            Y_mat = Y_mat.reshape(-1, 1)

        N, P = X_mat.shape
        _, T = Y_mat.shape

        # Data centering
        if self.fit_intercept:
            self.X_mean_ = np.mean(X_mat, axis=0)
            self.Y_mean_ = np.mean(Y_mat, axis=0)
            X_c = X_mat - self.X_mean_
            Y_c = Y_mat - self.Y_mean_
        else:
            self.X_mean_ = np.zeros(P)
            self.Y_mean_ = np.zeros(T)
            X_c = X_mat
            Y_c = Y_mat

        # 1. Compute empirical covariance matrix (C = X^T * X / N)
        # and cross-covariance (XY = X^T * Y / N)
        C = (X_c.T @ X_c) / N
        XY = (X_c.T @ Y_c) / N

        # 2. Construct the Ridge matrix: (C + alpha * I)
        I = np.eye(P)
        ridge_matrix = C + self.alpha * I

        # Store condition number for multicollinearity diagnostics
        self.condition_number_ = np.linalg.cond(ridge_matrix)

        # 3. Solve for weights: W = (C + alpha * I)^(-1) * XY
        self.coef_ = np.linalg.solve(ridge_matrix, XY)

        # 4. Compute intercept
        if self.fit_intercept:
            self.intercept_ = self.Y_mean_ - (self.X_mean_ @ self.coef_)
        else:
            self.intercept_ = np.zeros(T)

        # --- STATISTICAL INFERENCE & ERRORS ---
        
        # Compute inverse of Ridge matrix for inference calculations
        ridge_inv = np.linalg.inv(ridge_matrix)
        
        # Calculate Effective Degrees of Freedom (EDF)
        # EDF = trace( X * (X^T X + N*alpha*I)^-1 * X^T ) = trace( (C + alpha I)^-1 * C )
        self.effective_df_ = np.trace(ridge_inv @ C)
        
        # Calculate residuals and estimate intrinsic noise variance (sigma^2)
        Y_pred = X_c @ self.coef_
        residuals = Y_c - Y_pred
        
        # Unbiased estimator for variance using EDF
        self.sigma_squared_ = np.sum(residuals**2, axis=0) / (N - self.effective_df_)

        # Covariance matrix of the coefficients (unscaled by noise)
        # Var(W) = (sigma^2 / N) * [ (C + alpha I)^-1 * C * (C + alpha I)^-1 ]
        self._cov_unscaled = (ridge_inv @ C @ ridge_inv) / N

        # Compute Standard Errors for each coefficient and each target
        # shape will be (P, T)
        self.coef_se_ = np.zeros((P, T))
        for t in range(T):
            variance_W = self.sigma_squared_[t] * self._cov_unscaled
            self.coef_se_[:, t] = np.sqrt(np.diag(variance_W))

        return self

    def predict(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict targets for new data, optionally returning prediction uncertainties.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            New feature matrix of shape (M, P).
        return_std : bool, default=False
            If True, returns the standard deviation of the prediction (confidence intervals).

        Returns
        -------
        Y_pred : np.ndarray
            Predicted values of shape (M, T).
        Y_std : np.ndarray, optional
            Standard errors of the predictions of shape (M, T), returned if return_std=True.
        """
        X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        M, _ = X_mat.shape
        T = self.coef_.shape[1]

        # Standard linear prediction
        Y_pred = (X_mat @ self.coef_) + self.intercept_

        if not return_std:
            return Y_pred

        # Compute prediction uncertainties
        # Var(Y_pred) = Var(noise) + x^T * Var(W) * x
        Y_std = np.zeros((M, T))
        
        # x^T * unscaled_cov * x for all samples simultaneously
        # np.sum( (X @ Cov) * X, axis=1 ) computes the diagonal of X @ Cov @ X^T efficiently
        x_cov_x = np.sum((X_mat @ self._cov_unscaled) * X_mat, axis=1)

        for t in range(T):
            # Total variance = inherent noise + model uncertainty
            pred_var = self.sigma_squared_[t] + (self.sigma_squared_[t] * x_cov_x)
            Y_std[:, t] = np.sqrt(pred_var)

        return Y_pred, Y_std