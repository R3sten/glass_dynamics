"""
Residuals Analysis Module.

Analyzes global viscosity prediction residuals by combining experimental data, 
model predictions (MYEGA parameters), and glass composition. Groups the errors 
by the presence of specific chemical compounds (oxides).
"""

import numpy as np
import pandas as pd
from typing import List
from glass_dynamics.core.logger import logger

def compute_myega_viscosity(T: np.ndarray, log_inf: np.ndarray, Tg: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of the MYEGA viscosity curve.
    """
    T = np.maximum(T, 1e-9)
    term1 = (12.0 - log_inf) * (Tg / T)
    exponent = (m / (12.0 - log_inf) - 1.0) * (Tg / T - 1.0)
    return log_inf + term1 * np.exp(exponent)

def compute_global_viscosity_residuals(
    viscosity_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    composition_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculates the mean and standard deviation of global viscosity prediction 
    residuals (True - Predicted) for glasses containing specific chemical compounds.
    """
    logger.info("Computing global viscosity residuals across chemical compounds.")
    
    tg_col = next(c for c in predictions_df.columns if 'Tg' in c)
    m_col = next(c for c in predictions_df.columns if 'm' in c)
    inf_col = next(c for c in predictions_df.columns if 'inf' in c.lower())

    # 1. Merge experimental viscosity with model predictions on Glass ID
    merged_df = viscosity_df.join(predictions_df, on='ID', how='inner')
    
    # 2. Compute predicted viscosity
    merged_df['log_visc_pred'] = compute_myega_viscosity(
        T=merged_df['T'].values,
        log_inf=merged_df[inf_col].values,
        Tg=merged_df[tg_col].values,
        m=merged_df[m_col].values
    )
    
    # 3. Calculate residual: True - Predicted
    merged_df['residual'] = merged_df['log_visc'] - merged_df['log_visc_pred']
    
    # 4. SAFETY CHECK: Drop overlapping composition columns from merged_df
    # This prevents the ValueError if viscosity_df already contains the oxides.
    overlap_cols = [col for col in composition_df.columns if col in merged_df.columns]
    if overlap_cols:
        merged_df = merged_df.drop(columns=overlap_cols)
    
    # 5. Merge with the clean composition data
    final_df = merged_df.join(composition_df, on='ID', how='inner')
    
    compounds = composition_df.columns.tolist()
    results = []
    
    for comp in compounds:
        mask = final_df[comp] > 0
        count = mask.sum()
        
        if count > 0:
            comp_residuals = final_df.loc[mask, 'residual']
            results.append({
                'Compound': comp,
                'Count': int(count),
                'Mean_Residual': comp_residuals.mean(),
                'Std_Residual': comp_residuals.std()
            })
            
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='Count', ascending=True).reset_index(drop=True)
    
    return df_results