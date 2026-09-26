"""
Elastic Net Cross-Validation Hyperparameter Heatmap Module.

Evaluates predictive performance across a 2D grid of alpha and l1_ratio values
using K-Fold Cross Validation strictly on the Training Set to avoid data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.base import clone
from typing import Any, Dict
from glass_dynamics.core.logger import logger

def compute_myega_viscosity(T: np.ndarray, log_inf: np.ndarray, Tg: np.ndarray, m: np.ndarray) -> np.ndarray:
    T = np.maximum(T, 1e-9)
    term1 = (12.0 - log_inf) * (Tg / T)
    term2 = np.exp((m / (12.0 - log_inf) - 1.0) * (Tg / T - 1.0))
    return log_inf + term1 * term2

def compute_elasticnet_cv_heatmaps(
    base_model: Any,
    alpha_array: np.ndarray,
    l1_ratio_array: np.ndarray,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    viscosity_df: pd.DataFrame,
    y_scaler: Any = None,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Computes 2D matrices of Cross-Validated R^2 and RMSE for global viscosity and targets.
    """
    target_cols = y_train.columns.tolist()
    n_a, n_l1 = len(alpha_array), len(l1_ratio_array)
    
    logger.info(f"Computing {n_splits}-Fold CV Heatmaps: {n_a} alphas x {n_l1} l1_ratios ({n_a * n_l1 * n_splits} fits).")
    
    tg_col = next(c for c in target_cols if 'Tg' in c)
    m_col = next(c for c in target_cols if 'm' in c)
    inf_col = next(c for c in target_cols if 'inf' in c.lower())
    
    results_dict = {
        "Global_Viscosity": {"R2": np.zeros((n_a, n_l1)), "RMSE": np.zeros((n_a, n_l1))},
        tg_col: {"R2": np.zeros((n_a, n_l1)), "RMSE": np.zeros((n_a, n_l1))},
        m_col: {"R2": np.zeros((n_a, n_l1)), "RMSE": np.zeros((n_a, n_l1))},
        inf_col: {"R2": np.zeros((n_a, n_l1)), "RMSE": np.zeros((n_a, n_l1))}
    }
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for i, alpha in enumerate(alpha_array):
        for j, l1_ratio in enumerate(l1_ratio_array):
            
            # Temporary storage for the K folds
            fold_metrics = {k: {"R2": [], "RMSE": []} for k in target_cols + ["Global_Viscosity"]}
            
            for train_idx, val_idx in kf.split(X_train):
                # Isolate the current CV folds
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model = clone(base_model)
                model.set_params(alpha=alpha, l1_ratio=l1_ratio)
                model.fit(X_tr, y_tr)
                
                y_val_pred_scaled = model.predict(X_val)
                    
                if y_scaler is not None:
                    y_val_pred_phys = y_scaler.inverse_transform(y_val_pred_scaled)
                    y_val_true_phys = y_scaler.inverse_transform(y_val)
                else:
                    y_val_pred_phys = y_val_pred_scaled
                    y_val_true_phys = y_val.to_numpy()
                    
                val_preds_df = pd.DataFrame(y_val_pred_phys, columns=target_cols, index=y_val.index)
                
                # 1. Parameter Metrics
                for k, col in enumerate(target_cols):
                    fold_metrics[col]["R2"].append(r2_score(y_val_true_phys[:, k], y_val_pred_phys[:, k]))
                    fold_metrics[col]["RMSE"].append(np.sqrt(mean_squared_error(y_val_true_phys[:, k], y_val_pred_phys[:, k])))
                
                # 2. Global Viscosity Metrics
                val_visc_df = viscosity_df[viscosity_df['ID'].isin(y_val.index)].copy()
                merged_df = val_visc_df.join(val_preds_df, on='ID', how='inner')
                
                preds_eta = compute_myega_viscosity(
                    T=merged_df['T'].values, log_inf=merged_df[inf_col].values,
                    Tg=merged_df[tg_col].values, m=merged_df[m_col].values
                )
                true_eta = merged_df['log_visc'].values
                
                valid_mask = ~np.isnan(preds_eta)
                if valid_mask.sum() > 0:
                    fold_metrics["Global_Viscosity"]["R2"].append(r2_score(true_eta[valid_mask], preds_eta[valid_mask]))
                    fold_metrics["Global_Viscosity"]["RMSE"].append(np.sqrt(mean_squared_error(true_eta[valid_mask], preds_eta[valid_mask])))
            
            # Average the metrics across all K folds (using nanmean for safety)
            for key in results_dict.keys():
                results_dict[key]["R2"][i, j] = np.nanmean(fold_metrics[key]["R2"])
                results_dict[key]["RMSE"][i, j] = np.nanmean(fold_metrics[key]["RMSE"])
                
    final_dfs = {}
    for key in results_dict.keys():
        final_dfs[key] = {
            "R2": pd.DataFrame(results_dict[key]["R2"], index=alpha_array, columns=l1_ratio_array),
            "RMSE": pd.DataFrame(results_dict[key]["RMSE"], index=alpha_array, columns=l1_ratio_array)
        }
        
    return final_dfs