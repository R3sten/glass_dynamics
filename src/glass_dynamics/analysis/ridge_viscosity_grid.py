"""
Ridge Global Viscosity Grid Search Module.

Evaluates the global physical predictive performance (MYEGA Viscosity) 
of a Ridge Regression model across a logarithmic grid of alpha penalties.
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

def compute_ridge_viscosity_alpha_grid(
    base_model: Any,
    alpha_array: np.ndarray,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    viscosity_df: pd.DataFrame,
    y_scaler: Any = None
) -> pd.DataFrame:
    """
    Computes global viscosity R^2 and RMSE across varying alpha penalties for Ridge.
    """
    target_cols = y_train.columns.tolist()
    n_a = len(alpha_array)
    
    logger.info(f"Computing train and test global viscosity metrics across {n_a} alpha values.")
    
    tg_col = next(c for c in target_cols if 'Tg' in c)
    m_col = next(c for c in target_cols if 'm' in c)
    inf_col = next(c for c in target_cols if 'inf' in c.lower())
    
    results = []
    
    for alpha in alpha_array:
        model = clone(base_model)
        # Assuming the model has an 'alpha' parameter standard to scikit-learn
        model.set_params(alpha=alpha)
        
        # Train and Predict
        model.fit(X_train, y_train)
        y_train_pred_scaled = model.predict(X_train)
        y_test_pred_scaled = model.predict(X_test)
            
        # Transform to physical units
        if y_scaler is not None:
            y_train_pred_phys = y_scaler.inverse_transform(y_train_pred_scaled)
            y_test_pred_phys = y_scaler.inverse_transform(y_test_pred_scaled)
        else:
            y_train_pred_phys = y_train_pred_scaled
            y_test_pred_phys = y_test_pred_scaled
            
        train_preds_df = pd.DataFrame(y_train_pred_phys, columns=target_cols, index=y_train.index)
        test_preds_df = pd.DataFrame(y_test_pred_phys, columns=target_cols, index=y_test.index)
        
        # Helper function to compute metrics
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
            
        train_r2, train_rmse = get_metrics(train_preds_df, y_train.index)
        test_r2, test_rmse = get_metrics(test_preds_df, y_test.index)
            
        results.append({
            'alpha': float(alpha),
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'Train_RMSE': train_rmse,
            'Test_RMSE': test_rmse
        })
        
    return pd.DataFrame(results).set_index('alpha')