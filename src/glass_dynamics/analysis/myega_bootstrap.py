"""
MYEGA Bootstrap Uncertainty Module.

Uses an Ensemble of Ridge Regression models trained on bootstrap samples of the 
training data to estimate the true predictive uncertainty of the MYEGA parameters.
This approach inherently preserves the physical covariance between Tg and fragility (m),
producing stable and realistic confidence intervals.
"""

import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.base import clone
from typing import Dict, Tuple, List, Any
from glass_dynamics.core.logger import logger

def compute_myega_curve(T: np.ndarray, log_inf: float, Tg: float, m: float) -> np.ndarray:
    """Computes a single MYEGA viscosity curve."""
    T = np.maximum(T, 1e-9)
    term1 = (12.0 - log_inf) * (Tg / T)
    term2 = np.exp((m / (12.0 - log_inf) - 1.0) * (Tg / T - 1.0))
    return log_inf + term1 * term2

def generate_bootstrap_predictions(
    base_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_scaler: Any = None,
    n_estimators: int = 100,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[Any, pd.DataFrame]]:
    """
    Trains an ensemble of models on bootstrap samples to generate distributions of predictions.
    Includes a bulletproof switch for Target-Specific models like CustomSupervisedPCR.
    """
    logger.info(f"Training ensemble of {n_estimators} bootstrap models...")
    
    np.random.seed(random_state)
    target_cols = y_train.columns
    test_ids = X_test.index
    
    # CONTROLLO INFALLIBILE: Verifica se la parola "Supervised" è nel nome del tipo.
    # Questo elude qualsiasi problema di import o di rinominazione del modulo.
    is_target_specific = "Supervised" in type(base_model).__name__
    
    # Storage for predictions: shape (n_estimators, n_test_samples, n_targets)
    all_raw_preds = np.zeros((n_estimators, len(X_test), len(target_cols)))
    
    for i in range(n_estimators):
        # 1. Create a bootstrap sample (sample with replacement)
        X_boot, y_boot = resample(X_train, y_train, random_state=random_state + i)
        
        # 2. INTELLIGENT FITTING
        if is_target_specific:
            # Addestramento indipendente per Tg, m, log_eta_inf
            for j, target in enumerate(target_cols):
                target_model = clone(base_model)
                # Estraiamo rigorosamente una singola colonna e la forziamo a vettore 1D
                y_boot_1d = y_boot[target].to_numpy().ravel()
                target_model.fit(X_boot, y_boot_1d)
                all_raw_preds[i, :, j] = target_model.predict(X_test)
        else:
            # Modelli standard (Ridge, ElasticNet)
            model = clone(base_model)
            model.fit(X_boot, y_boot)
            all_raw_preds[i, :, :] = model.predict(X_test)

    # 3. Convert all predictions back to physical units (if a scaler was used)
    flat_preds = all_raw_preds.reshape(-1, len(target_cols))
    if y_scaler is not None:
        flat_preds_physical = y_scaler.inverse_transform(flat_preds)
    else:
        flat_preds_physical = flat_preds
        
    all_physical_preds = flat_preds_physical.reshape(n_estimators, len(X_test), len(target_cols))
    
    # 4. Calculate the Mean Prediction for the final model output
    mean_preds = np.mean(all_physical_preds, axis=0)
    mean_df = pd.DataFrame(mean_preds, index=test_ids, columns=target_cols)
    
    # 5. Organize the bootstrap distributions per Glass ID
    distributions_dict = {}
    for j, glass_id in enumerate(test_ids):
        glass_preds = all_physical_preds[:, j, :]
        distributions_dict[glass_id] = pd.DataFrame(glass_preds, columns=target_cols)

    return mean_df, distributions_dict

def extract_myega_confidence_bands(
    T_array: np.ndarray,
    glass_distributions_df: pd.DataFrame,
    confidence: float = 95.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the confidence bands by generating all N MYEGA curves and extracting percentiles.

    Parameters
    ----------
    T_array : np.ndarray
        Temperatures to evaluate the curve on.
    glass_distributions_df : pd.DataFrame
        The N bootstrap predictions for a single glass (log_inf, Tg, m).
    confidence : float, default=95.0
        The confidence interval percentage.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (mean_curve, lower_band, upper_band)
    """
    # Dynamically find columns
    tg_col = next(c for c in glass_distributions_df.columns if 'Tg' in c)
    m_col = next(c for c in glass_distributions_df.columns if 'm' in c)
    inf_col = next(c for c in glass_distributions_df.columns if 'inf' in c.lower())

    n_models = len(glass_distributions_df)
    all_curves = np.zeros((n_models, len(T_array)))

    # Generate the MYEGA curve for each of the N models in the ensemble
    for i in range(n_models):
        row = glass_distributions_df.iloc[i]
        all_curves[i, :] = compute_myega_curve(T_array, row[inf_col], row[tg_col], row[m_col])

    # Extract percentiles point-by-point
    lower_perc = (100.0 - confidence) / 2.0
    upper_perc = 100.0 - lower_perc
    
    # np.percentile processes the columns (temperatures) across all rows (models)
    lower_band = np.percentile(all_curves, lower_perc, axis=0)
    median_curve = np.percentile(all_curves, 50.0, axis=0)
    upper_band = np.percentile(all_curves, upper_perc, axis=0)

    return median_curve, lower_band, upper_band