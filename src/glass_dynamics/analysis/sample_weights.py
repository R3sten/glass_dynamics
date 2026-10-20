"""
MYEGA Fit Quality Sample Weights Module.

Computes inverse-MSE weights for each glass liquid based on the quality of its 
experimental MYEGA curve fit. Noisy or bifurcated experimental curves receive 
lower weights to prevent them from corrupting linear regression parameters.
"""

import numpy as np
import pandas as pd
from typing import Any
from glass_dynamics.core.logger import logger

def compute_myega_viscosity(T: np.ndarray, log_inf: np.ndarray, Tg: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Vectorized MYEGA viscosity equation."""
    T = np.maximum(T, 1e-9)
    term1 = (12.0 - log_inf) * (Tg / T)
    term2 = np.exp((m / (12.0 - log_inf) - 1.0) * (Tg / T - 1.0))
    return log_inf + term1 * term2

def compute_myega_fit_weights(
    viscosity_df: pd.DataFrame,
    y_df: pd.DataFrame,
    y_scaler: Any = None,
    epsilon: float = 1e-2,
    normalize: bool = True
) -> pd.Series:
    """
    Computes sample weights inversely proportional to the initial MYEGA fit MSE:
    p_i = 1 / (MSE_MYEGA,i + epsilon)

    Parameters
    ----------
    viscosity_df : pd.DataFrame
        Raw experimental dataset containing 'ID', 'T', and 'log_visc'.
    y_df : pd.DataFrame
        Target DataFrame (scaled or unscaled) indexed by Glass ID, containing 
        the reference MYEGA parameters (Tg, m, log_eta_inf).
    y_scaler : Any, optional
        Scaler used on y_df. If provided, y_df is inverse-transformed to physical units.
    epsilon : float, default=1e-2
        Damping factor to prevent division by zero or extreme weight explosion 
        for liquids with near-zero fit error.
    normalize : bool, default=True
        If True, normalizes weights so that their mean equals 1.0. This preserves 
        the exact scale of the Ridge penalty alpha.

    Returns
    -------
    pd.Series
        Sample weights indexed by Glass ID, aligned with y_df.index.
    """
    logger.info(f"Computing MYEGA quality weights for {len(y_df)} liquids (epsilon={epsilon}).")
    
    target_cols = y_df.columns.tolist()
    tg_col = next(c for c in target_cols if 'Tg' in c)
    m_col = next(c for c in target_cols if 'm' in c)
    inf_col = next(c for c in target_cols if 'inf' in c.lower())

    # Convert reference targets to physical units if scaled
    if y_scaler is not None:
        y_phys = pd.DataFrame(
            y_scaler.inverse_transform(y_df),
            index=y_df.index,
            columns=target_cols
        )
    else:
        y_phys = y_df.copy()

    # Join experimental viscosity points with their reference MYEGA parameters
    valid_visc = viscosity_df[viscosity_df['ID'].isin(y_phys.index)].copy()
    merged = valid_visc.join(y_phys, on='ID', how='inner')

    # Compute expected MYEGA viscosity from the reference parameters
    merged['pred_visc'] = compute_myega_viscosity(
        T=merged['T'].to_numpy(),
        log_inf=merged[inf_col].to_numpy(),
        Tg=merged[tg_col].to_numpy(),
        m=merged[m_col].to_numpy()
    )
    merged['sq_error'] = (merged['log_visc'] - merged['pred_visc']) ** 2

    # Compute MSE per glass ID
    mse_per_glass = merged.groupby('ID')['sq_error'].mean()

    # Reindex to match y_df exactly (fill any missing IDs with median MSE as fallback)
    mse_series = mse_per_glass.reindex(y_df.index)
    if mse_series.isna().any():
        mse_series = mse_series.fillna(mse_series.median())

    # Calculate inverse MSE weights: p_i = 1 / (MSE_i + epsilon)
    weights = 1.0 / (mse_series + epsilon)

    if normalize:
        weights = weights / weights.mean()

    weights.name = "sample_weight"
    return weights