"""
Global Viscosity Dimensionality Analysis Module.

Evaluates the global physical predictive performance (MYEGA Viscosity) 
across varying numbers of Principal Components (P) for both Train and Test sets. 
Compatible with both standard multi-target models (PCR) and target-specific models (SPCR).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import clone
from typing import Any
from glass_dynamics.core.logger import logger

def compute_myega_viscosity(T: np.ndarray, log_inf: np.ndarray, Tg: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Vectorized MYEGA computation for rapid grid search."""
    T = np.maximum(T, 1e-9)
    term1 = (12.0 - log_inf) * (Tg / T)
    term2 = np.exp((m / (12.0 - log_inf) - 1.0) * (Tg / T - 1.0))
    return log_inf + term1 * term2

def compute_global_viscosity_p_grid(
    base_model: Any,
    p_array: np.ndarray,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    viscosity_df: pd.DataFrame,
    y_scaler: Any = None
) -> pd.DataFrame:
    """
    Computes global viscosity R^2 and RMSE across varying numbers of components.
    """
    model_name = base_model.__class__.__name__
    target_cols = y_train.columns.tolist()
    n_p = len(p_array)
    
    logger.info(f"Computing train and test global viscosity metrics across {n_p} configurations for {model_name}.")
    
    tg_col = next(c for c in target_cols if 'Tg' in c)
    m_col = next(c for c in target_cols if 'm' in c)
    inf_col = next(c for c in target_cols if 'inf' in c.lower())
    
    results = []
    
    for p in p_array:
        model = clone(base_model)
        model.set_params(n_components=int(p))
        
        y_train_pred_scaled = np.zeros_like(y_train.to_numpy())
        y_test_pred_scaled = np.zeros_like(y_test.to_numpy())
        
        # 1. Intelligent fitting (SPCR vs PCR)
        if model_name == "CustomSupervisedPCR":
            for i, target in enumerate(target_cols):
                target_model = clone(model)
                target_model.fit(X_train, y_train[target])
                y_train_pred_scaled[:, i] = target_model.predict(X_train)
                y_test_pred_scaled[:, i] = target_model.predict(X_test)
        else:
            model.fit(X_train, y_train)
            y_train_pred_scaled = model.predict(X_train)
            y_test_pred_scaled = model.predict(X_test)
            
        # 2. Transform to physical units
        if y_scaler is not None:
            y_train_pred_phys = y_scaler.inverse_transform(y_train_pred_scaled)
            y_test_pred_phys = y_scaler.inverse_transform(y_test_pred_scaled)
        else:
            y_train_pred_phys = y_train_pred_scaled
            y_test_pred_phys = y_test_pred_scaled
            
        train_preds_df = pd.DataFrame(y_train_pred_phys, columns=target_cols, index=y_train.index)
        test_preds_df = pd.DataFrame(y_test_pred_phys, columns=target_cols, index=y_test.index)
        
        # 3. Helper function to compute metrics for a given split
        def get_metrics(preds_df, original_indices):
            split_visc_df = viscosity_df[viscosity_df['ID'].isin(original_indices)].copy()
            merged_df = split_visc_df.join(preds_df, on='ID', how='inner')
            
            preds_eta = compute_myega_viscosity(
                T=merged_df['T'].values,
                log_inf=merged_df[inf_col].values,
                Tg=merged_df[tg_col].values,
                m=merged_df[m_col].values
            )
            true_eta = merged_df['log_visc'].values
            
            valid_mask = ~np.isnan(preds_eta)
            if valid_mask.sum() > 0:
                r2 = r2_score(true_eta[valid_mask], preds_eta[valid_mask])
                rmse = np.sqrt(mean_squared_error(true_eta[valid_mask], preds_eta[valid_mask]))
                return r2, rmse
            return 0.0, np.nan
            
        # 4. Compute metrics for both sets
        train_r2, train_rmse = get_metrics(train_preds_df, y_train.index)
        test_r2, test_rmse = get_metrics(test_preds_df, y_test.index)
            
        results.append({
            'n_components': int(p),
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'Train_RMSE': train_rmse,
            'Test_RMSE': test_rmse
        })
        
    return pd.DataFrame(results).set_index('n_components')