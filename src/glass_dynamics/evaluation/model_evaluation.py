"""
Model Evaluation and Physical Validation Module.

This module evaluates the performance of the trained regression models 
(including Ridge, Elastic Net, and SPCR) on unseen test data. It computes 
metrics for individual targets in their scaled (training) space. Then, it 
validates the global physical performance by comparing predicted viscosities 
against both the exact experimental values and the "ideal" viscosities 
derived from the true MYEGA fitted parameters.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.base import clone
from scipy.stats import pearsonr
from typing import Optional, Dict, Any
from glass_dynamics.core.logger import logger

def myega_equation(T: np.ndarray, log_inf: np.ndarray, Tg: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Calculates log10(Viscosity) using the physical MYEGA equation.
    
    Parameters
    ----------
    T : np.ndarray or float
        Temperature (K).
    log_inf : np.ndarray or float
        High-temperature viscosity limit.
    Tg : np.ndarray or float
        Glass transition temperature.
    m : np.ndarray or float
        Fragility index.
        
    Returns
    -------
    np.ndarray
        Predicted log10(Viscosity).
    """
    T = np.maximum(T, 1e-9) # Prevent division by zero
    term1 = (12.0 - log_inf) * (Tg / T)
    term2 = np.exp((m / (12.0 - log_inf) - 1.0) * (Tg / T - 1.0))
    return log_inf + term1 * term2


def evaluate_final_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    viscosity_df: pd.DataFrame,
    y_scaler: Optional[Any] = None,
    target_cols: list = ['Tg', 'm','log_inf']
) -> Dict[str, Any]:
    """
    Train and evaluate the model on individual parameters (scaled space) 
    and globally on physical viscosity using MYEGA.

    Parameters
    ----------
    model : Any
        The un-fitted regression model (e.g., CustomRidgeRegression, CustomSupervisedPCR).
    X_train, y_train : pd.DataFrame
        Training data required to fit the model properly.
    X_test, y_test : pd.DataFrame
        Testing data. y_test must be in the scaled space used for training.
    viscosity_df : pd.DataFrame
        The raw dataset containing experimental Viscosity and Temperature measurements.
        Must contain columns 'ID', 'T', and 'log_visc'.
    y_scaler : Any, optional
        A fitted scaler used to inverse_transform targets back to physical units.
    target_cols : list, default=['Tg', 'm','log_inf']
        The names of the target columns.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all computed metrics and prediction arrays.
    """
    model_name = model.__class__.__name__
    logger.info(f"Training and Evaluating {model_name} on the unseen Test Set ({len(X_test)} glasses)...")
    
    results = {"parameters": {}, "viscosity_vs_experimental": {}, "viscosity_vs_fitted": {}}
    
    # Storage for predictions
    y_pred_scaled = np.zeros_like(y_test.to_numpy())
    
    # --- 1. INTELLIGENT TRAINING & PREDICTION ---
    if model_name == "CustomSupervisedPCR":
        # SPCR requires a specific optimal subspace for each target
        for i, target in enumerate(target_cols):
            target_model = clone(model)
            target_model.fit(X_train, y_train[target])
            y_pred_scaled[:, i] = target_model.predict(X_test)
    else:
        # Standard models (Ridge, ElasticNet, PCR) can handle multi-target natively
        cloned_model = clone(model)
        cloned_model.fit(X_train, y_train)
        y_pred_scaled = cloned_model.predict(X_test)

    # --- 2. Parameter Evaluation (In Scaled Space) ---
    y_true_scaled = y_test.to_numpy()
    
    predictions_scaled_df = pd.DataFrame(y_pred_scaled, columns=target_cols, index=y_test.index)
    truths_scaled_df = pd.DataFrame(y_true_scaled, columns=target_cols, index=y_test.index)

    print("\n" + "=" * 50)
    print("      TEST SET PERFORMANCE (SCALED PARAMETERS)")
    print("=" * 50)

    for col in target_cols:
        y_true_col = truths_scaled_df[col].to_numpy()
        y_pred_col = predictions_scaled_df[col].to_numpy()

        r, _ = pearsonr(y_true_col, y_pred_col)
        r2 = r2_score(y_true_col, y_pred_col)
        rmse = np.sqrt(mean_squared_error(y_true_col, y_pred_col))
        mae = mean_absolute_error(y_true_col, y_pred_col)
        medae = median_absolute_error(y_true_col, y_pred_col)

        results["parameters"][col] = {"R": r, "R2": r2, "RMSE": rmse, "MAE": mae, "MedAE": medae}

        print(f"Target: {col}")
        print(f"  Pearson : {r:.3f}")
        print(f"  R^2     : {r2:.3f}")
        print(f"  RMSE    : {rmse:.3f}")
        print(f"  MAE     : {mae:.3f}")
        print(f"  MedAE   : {medae:.3f}")
        print("-" * 50)

    # --- 3. Transform back to physical units for MYEGA calculations ---
    if y_scaler is not None:
        y_pred_physical = y_scaler.inverse_transform(y_pred_scaled)
        y_true_physical = y_scaler.inverse_transform(y_true_scaled)
    else:
        y_pred_physical = y_pred_scaled
        y_true_physical = y_true_scaled
        
    predictions_df = pd.DataFrame(y_pred_physical, columns=target_cols, index=y_test.index)
    truths_df = pd.DataFrame(y_true_physical, columns=target_cols, index=y_test.index)

    # --- 4. Global Viscosity Evaluation ---
    logger.info("Computing global viscosity performance via MYEGA equation...")
    
    true_viscosities = []  # Experimental log_visc
    inter_viscosities = [] # MYEGA log_visc using true fitted parameters
    pred_viscosities = []  # MYEGA log_visc using predicted parameters

    # Dynamically find the appropriate column names based on substring matches
    tg_col = next(c for c in target_cols if 'Tg' in c)
    m_col = next(c for c in target_cols if 'm' in c)
    log_inf_col = next(c for c in target_cols if 'inf' in c.lower())

    for idx in y_test.index:
        Tg_pred = predictions_df.loc[idx, tg_col]
        m_pred = predictions_df.loc[idx, m_col]
        log_inf_pred = predictions_df.loc[idx, log_inf_col]
        
        Tg_true = truths_df.loc[idx, tg_col]
        m_true = truths_df.loc[idx, m_col]
        log_inf_true = truths_df.loc[idx, log_inf_col]

        glass_df = viscosity_df[viscosity_df['ID'] == idx]

        for _, row in glass_df.iterrows():
            T_val = row['T']
            true_visc = row['log_visc']
            
            eta_pred = myega_equation(T_val, log_inf_pred, Tg_pred, m_pred)
            eta_inter = myega_equation(T_val, log_inf_true, Tg_true, m_true)
            
            if not np.isnan(eta_pred):
                pred_viscosities.append(eta_pred)
                inter_viscosities.append(eta_inter)
                true_viscosities.append(true_visc)

    pred_viscosities = np.array(pred_viscosities)
    inter_viscosities = np.array(inter_viscosities)
    true_viscosities = np.array(true_viscosities)
    
    def evaluate_and_print(true_array, pred_array, title, dict_key):
        r, _ = pearsonr(true_array, pred_array)
        r2 = r2_score(true_array, pred_array)
        rmse = np.sqrt(mean_squared_error(true_array, pred_array))
        mae = mean_absolute_error(true_array, pred_array)
        medae = median_absolute_error(true_array, pred_array)
        
        results[dict_key] = {"R": r, "R2": r2, "RMSE": rmse, "MAE": mae, "MedAE": medae}
        
        print(f"      {title}")
        print("-" * 50)
        print(f"  Pearson : {r:.3f}")
        print(f"  R^2     : {r2:.3f}")
        print(f"  RMSE    : {rmse:.3f}")
        print(f"  MAE     : {mae:.3f}")
        print(f"  MedAE   : {medae:.3f}")
        print("=" * 50)

    print("\n" + "=" * 50)
    
    evaluate_and_print(
        inter_viscosities, pred_viscosities, 
        title="VISCOSITY: PREDICTED vs TRUE FITTED (MYEGA)", 
        dict_key="viscosity_vs_fitted"
    )

    evaluate_and_print(
        true_viscosities, pred_viscosities, 
        title="VISCOSITY: PREDICTED vs EXPERIMENTAL (GROUND TRUTH)", 
        dict_key="viscosity_vs_experimental"
    )
    
    # Save the arrays for downstream plotting
    results["viscosity_arrays"] = {
        "true_experimental": true_viscosities,
        "true_fitted_myega": inter_viscosities,
        "predicted_myega": pred_viscosities
    }

    return results