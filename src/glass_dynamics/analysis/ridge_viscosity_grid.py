"""
Ridge Global Viscosity Grid Search Module.

Evaluates the global physical predictive performance (MYEGA Viscosity) 
of a Ridge Regression model across a logarithmic grid of alpha penalties.
"""

import inspect
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.base import clone
from typing import Any, Optional, Tuple
from glass_dynamics.core.logger import logger

def compute_myega_viscosity(
    T: np.ndarray, 
    log_inf: np.ndarray, 
    Tg: np.ndarray, 
    m: np.ndarray
) -> np.ndarray:
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
    y_scaler: Any = None,
    sample_weight: pd.Series = None
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
    
    # Extract aligned training weights if provided
    w_train = sample_weight.loc[X_train.index].to_numpy() if sample_weight is not None else None
    
    for alpha in alpha_array:
        model = clone(base_model)
        # Assuming the model has an 'alpha' parameter standard to scikit-learn
        model.set_params(alpha=alpha)
        
        # Train and Predict
        model.fit(X_train, y_train, sample_weight=w_train)
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


def compute_ridge_viscosity_alpha_grid_cross_validation(
    base_model: Any,
    alpha_array: np.ndarray,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    viscosity_df: pd.DataFrame,
    y_scaler: Optional[Any] = None,
    sample_weight: Optional[pd.Series] = None,
    n_splits: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Computes global viscosity R^2 and RMSE across varying alpha penalties for Ridge
    using K-Fold Cross-Validation strictly on the training set.

    Parameters
    ----------
    base_model : Any
        Unfitted Ridge estimator (e.g., CustomRidgeRegression).
    alpha_array : np.ndarray
        1D array of regularization penalties (alpha) to evaluate.
    X_train : pd.DataFrame
        Training feature matrix indexed by Glass ID.
    y_train : pd.DataFrame
        Training target matrix (scaled or unscaled) indexed by Glass ID.
    viscosity_df : pd.DataFrame
        Raw experimental dataset containing 'ID', 'T', and 'log_visc'.
    y_scaler : Any, optional
        Fitted scaler used to inverse-transform predicted targets back to physical units.
    sample_weight : pd.Series, optional
        Optional sample weights indexed by Glass ID for Weighted Ridge.
    n_splits : int, default=5
        Number of folds for K-Fold Cross-Validation.
    random_state : int, default=42
        Random seed for reproducible fold splitting.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by 'alpha' containing mean training and validation (CV)
        R^2 and RMSE metrics across the K folds.
    """
    target_cols = y_train.columns.tolist()
    n_a = len(alpha_array)
    
    logger.info(
        f"Computing {n_splits}-Fold CV global viscosity metrics across "
        f"{n_a} alpha values ({n_a * n_splits} total fits)."
    )
    
    tg_col = next(c for c in target_cols if 'Tg' in c)
    m_col = next(c for c in target_cols if 'm' in c)
    inf_col = next(c for c in target_cols if 'inf' in c.lower())
    
    # Pre-filter viscosity DataFrame to training IDs once for fast fold lookups
    train_visc_df = viscosity_df[viscosity_df['ID'].isin(X_train.index)].copy()
    
    # Align sample weights with X_train if provided
    if sample_weight is not None:
        w_all = sample_weight.loc[X_train.index].to_numpy(dtype=float)
    else:
        w_all = None

    supports_weights = "sample_weight" in inspect.signature(base_model.fit).parameters
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def get_fold_metrics(preds_df: pd.DataFrame) -> Tuple[float, float]:
        """Helper to evaluate MYEGA R^2 and RMSE on a specific fold."""
        merged_df = train_visc_df.join(preds_df, on='ID', how='inner')
        
        preds_eta = compute_myega_viscosity(
            T=merged_df['T'].to_numpy(),
            log_inf=merged_df[inf_col].to_numpy(),
            Tg=merged_df[tg_col].to_numpy(),
            m=merged_df[m_col].to_numpy()
        )
        true_eta = merged_df['log_visc'].to_numpy()
        
        valid_mask = ~np.isnan(preds_eta)
        if valid_mask.sum() > 0:
            r2 = r2_score(true_eta[valid_mask], preds_eta[valid_mask])
            rmse = np.sqrt(mean_squared_error(true_eta[valid_mask], preds_eta[valid_mask]))
            return r2, rmse
        return np.nan, np.nan

    results = []
    
    for alpha in alpha_array:
        fold_train_r2, fold_val_r2 = [], []
        fold_train_rmse, fold_val_rmse = [], []

        for train_idx, val_idx in kf.split(X_train):
            # 1. Isolate current training and validation folds
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = clone(base_model)
            model.set_params(alpha=alpha)
            
            # 2. Fit on the K-1 training folds
            if w_all is not None and supports_weights:
                w_tr = w_all[train_idx]
                model.fit(X_tr, y_tr, sample_weight=w_tr)
            else:
                model.fit(X_tr, y_tr)
                
            # 3. Predict on training fold and validation fold
            y_tr_pred_scaled = model.predict(X_tr)
            y_val_pred_scaled = model.predict(X_val)
            
            # 4. Transform back to physical units
            if y_scaler is not None:
                y_tr_pred_phys = y_scaler.inverse_transform(y_tr_pred_scaled)
                y_val_pred_phys = y_scaler.inverse_transform(y_val_pred_scaled)
            else:
                y_tr_pred_phys = y_tr_pred_scaled
                y_val_pred_phys = y_val_pred_scaled
                
            tr_preds_df = pd.DataFrame(y_tr_pred_phys, columns=target_cols, index=y_tr.index)
            val_preds_df = pd.DataFrame(y_val_pred_phys, columns=target_cols, index=y_val.index)
            
            # 5. Compute physical MYEGA metrics for the current fold
            tr_r2, tr_rmse = get_fold_metrics(tr_preds_df)
            val_r2, val_rmse = get_fold_metrics(val_preds_df)
            
            fold_train_r2.append(tr_r2)
            fold_train_rmse.append(tr_rmse)
            fold_val_r2.append(val_r2)
            fold_val_rmse.append(val_rmse)
            
        # Average across all K folds
        results.append({
            'alpha': float(alpha),
            'Train_R2': float(np.nanmean(fold_train_r2)),
            'CV_R2': float(np.nanmean(fold_val_r2)),
            'Train_RMSE': float(np.nanmean(fold_train_rmse)),
            'CV_RMSE': float(np.nanmean(fold_val_rmse))
        })
        
    return pd.DataFrame(results).set_index('alpha')