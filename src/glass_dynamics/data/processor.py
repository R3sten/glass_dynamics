import pandas as pd
import numpy as np
from typing import Tuple
from glass_dynamics.core.logger import logger
from .myega import fit_myega

#TODO: add a requirements for viscosity in the mid range
def filter_and_fit_dataset(df_raw: pd.DataFrame, min_r2_threshold: float = 0.95, min_data_points: int = 3, max_rmse_threshold: float = 10.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes the raw viscosity dataset. Groups data by glass ID, fits the MYEGA equation,
    and filters out glasses that don't fit the model well or have insufficient data points.

    Args:
        df_raw (pd.DataFrame): Raw dataframe containing at least ['ID', 'T', 'log_visc', ... features ...].
        min_r2_threshold (float): Minimum R^2 score required to accept the MYEGA fit.
        min_data_points (int): Minimum number of T-eta pairs required to attempt a fit.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - Cleaned DataFrame with fitted Tg, m, and chemical features.
            - DataFrame of rejected glasses (for later analysis on WHY they failed).
    """
    logger.info(f"Starting data processing and MYEGA fitting on {df_raw['ID'].nunique()} unique glasses...")

    accepted_records = []
    rejected_records = []

    # Group by ID (assuming your raw data has multiple T-eta points per glass)
    grouped = df_raw.groupby('ID')

    for id, group in grouped:
        # Extract features (we assume the composition features are constant for the same glass)
        # E.g., taking the first row for composition/features
        glass_features = group.drop(columns=['T', 'log_visc']).iloc[[0]].to_dict('records')[0]
        T_data = group['T'].values
        log_eta_data = group['log_visc'].values

        # 1. Check minimum data points
        if len(T_data) < min_data_points:
            glass_features['Rejection_Reason'] = 'Too few data points'
            rejected_records.append(glass_features)
            continue

        # 2. Fit MYEGA
        fit_results = fit_myega(T_data, log_eta_data)

        if fit_results is None:
            glass_features['Rejection_Reason'] = 'Fit failed to converge'
            rejected_records.append(glass_features)
            continue

        # 3. Apply Quality Filters
        if fit_results['r2'] < min_r2_threshold:
            glass_features['Rejection_Reason'] = f"Low R2 ({fit_results['r2']:.3f})"
            rejected_records.append(glass_features)
            continue

        if fit_results['rmse'] > max_rmse_threshold:
            glass_features['Rejection_Reason'] = f"High RMSE ({fit_results['rmse']:.3f})"
            rejected_records.append(glass_features)
            continue

        # 4. If passed all checks, save the fitted target variables and R2
        glass_features['Tg_fitted'] = fit_results['tg']
        glass_features['m_fitted'] = fit_results['m']
        glass_features['log_eta_inf_fitted'] = fit_results['log_eta_inf']
        glass_features['myega_r2'] = fit_results['r2']
        glass_features['myega_rmse'] = fit_results['rmse']

        accepted_records.append(glass_features)

    df_accepted = pd.DataFrame(accepted_records)
    df_rejected = pd.DataFrame(rejected_records)

    logger.info(f"Processing complete. Accepted glasses: {len(df_accepted)}. Rejected glasses: {len(df_rejected)}.")

    return df_accepted, df_rejected