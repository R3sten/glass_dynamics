"""
Model Performance Analysis Module.

This module provides routines to evaluate the Custom Ridge Regression model 
across a range of alpha parameters, specifically tracking the Pearson 
correlation coefficient (R) and the coefficient of determination (R^2) 
for both training and testing datasets.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from typing import Dict
from glass_dynamics.models.ridge import CustomRidgeRegression
from glass_dynamics.core.logger import logger

def compute_alpha_performance_metrics(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    alphas: np.ndarray
) -> Dict[str, pd.DataFrame]:
    """
    Compute R and R^2 metrics for training and test sets across a spectrum of alphas.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.DataFrame
        Training target matrix containing MYEGA parameters.
    X_test : pd.DataFrame
        Testing feature matrix.
    y_test : pd.DataFrame
        Testing target matrix.
    alphas : np.ndarray
        Array of regularization strengths (alpha) to evaluate.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary mapping each target name to a pandas DataFrame containing
        the computed metrics ('train_R', 'test_R', 'train_R2', 'test_R2') 
        indexed by alpha.
    """
    target_names = y_train.columns.tolist()
    n_alphas = len(alphas)
    
    logger.info(f"Computing R and R^2 metrics for {len(target_names)} targets.")
    
    # Initialize a dictionary to hold the results for each target
    results_data = {
        target: {
            "train_R": np.zeros(n_alphas),
            "test_R": np.zeros(n_alphas),
            "train_R2": np.zeros(n_alphas),
            "test_R2": np.zeros(n_alphas)
        } for target in target_names
    }
    
    for i, alpha in enumerate(alphas):
        # Instantiate and fit our transparent, math-exact Ridge model
        model = CustomRidgeRegression(alpha=alpha, fit_intercept=True)
        model.fit(X_train, y_train)
        
        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        for j, target in enumerate(target_names):
            y_t_true = y_train.iloc[:, j].to_numpy()
            y_t_pred = y_train_pred[:, j]
            
            y_v_true = y_test.iloc[:, j].to_numpy()
            y_v_pred = y_test_pred[:, j]
            
            # Calculate R^2
            results_data[target]["train_R2"][i] = r2_score(y_t_true, y_t_pred)
            results_data[target]["test_R2"][i] = r2_score(y_v_true, y_v_pred)
            
            # Calculate Pearson R
            # pearsonr returns a tuple: (statistic, p-value), we only need [0]
            results_data[target]["train_R"][i] = pearsonr(y_t_true, y_t_pred)[0]
            results_data[target]["test_R"][i] = pearsonr(y_v_true, y_v_pred)[0]

    # Convert dictionaries to pandas DataFrames for easy plotting
    metrics_dfs = {}
    for target in target_names:
        df = pd.DataFrame(results_data[target], index=alphas)
        df.index.name = "alpha"
        metrics_dfs[target] = df
        
    logger.info("Performance metrics computation complete.")
    return metrics_dfs