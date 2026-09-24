"""
Elastic Net Grid Performance Module.

This module evaluates the Custom Elastic Net model across a 2D grid of 
parameters (alpha and l1_ratio). It tracks Pearson correlation (R) and 
Coefficient of Determination (R^2) for both training and testing sets,
returning 2D matrices suitable for 3D surface plotting. Includes safe 
handling of constant predictions caused by extreme L1 sparsity.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from typing import Dict
from glass_dynamics.models.elastic_net import CustomElasticNet
from glass_dynamics.core.logger import logger

def compute_elastic_net_grid_metrics(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    alphas: np.ndarray,
    l1_ratios: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute R and R^2 metrics across a 2D grid of alphas and l1_ratios.
    """
    target_names = y_train.columns.tolist()
    n_alphas = len(alphas)
    n_l1 = len(l1_ratios)
    
    logger.info(f"Computing 2D grid metrics ({n_alphas}x{n_l1} models) for {len(target_names)} targets.")
    
    results_data = {
        target: {
            "train_R": np.zeros((n_alphas, n_l1)),
            "test_R": np.zeros((n_alphas, n_l1)),
            "train_R2": np.zeros((n_alphas, n_l1)),
            "test_R2": np.zeros((n_alphas, n_l1))
        } for target in target_names
    }
    
    for i, alpha in enumerate(alphas):
        for j, l1_ratio in enumerate(l1_ratios):
            
            model = CustomElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            for k, target in enumerate(target_names):
                y_t_true = y_train.iloc[:, k].to_numpy()
                y_t_pred = y_train_pred[:, k]
                
                y_v_true = y_test.iloc[:, k].to_numpy()
                y_v_pred = y_test_pred[:, k]
                
                # R^2 manages constant predictions automatically (returns 0.0 or negative)
                results_data[target]["train_R2"][i, j] = r2_score(y_t_true, y_t_pred)
                results_data[target]["test_R2"][i, j] = r2_score(y_v_true, y_v_pred)
                
                # --- SAFE PEARSON CALCULATION ---
                # If the model predicts a constant value (std dev is ~0), R is technically 0.
                if np.std(y_t_pred) < 1e-12:
                    results_data[target]["train_R"][i, j] = 0.0
                else:
                    results_data[target]["train_R"][i, j] = pearsonr(y_t_true, y_t_pred)[0]
                    
                if np.std(y_v_pred) < 1e-12:
                    results_data[target]["test_R"][i, j] = 0.0
                else:
                    results_data[target]["test_R"][i, j] = pearsonr(y_v_true, y_v_pred)[0]

    logger.info("2D Performance metrics computation complete.")
    return results_data