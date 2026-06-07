"""
Extrapolation Metrics Extraction Module.

Calculates the prediction error (RMSE) and chemical distance (Canberra) 
for each test glass compared to the training manifold. Outputs a structured 
Pandas DataFrame for independent analysis and downstream plotting.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error
from typing import List, Dict
from glass_dynamics.core.logger import logger

def myega_equation_np(T: np.ndarray, log_inf: float, Tg: float, m: float) -> np.ndarray:
    """Numpy implementation of the empirical MYEGA equation."""
    T = np.maximum(T, 1e-9)
    term1 = (12.0 - log_inf) * (Tg / T)
    term2 = np.exp((m / (12.0 - log_inf) - 1.0) * (Tg / T - 1.0))
    return log_inf + term1 * term2

def generate_extrapolation_metrics(
    compounds: List[str],
    viscosity_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    names_dict: Dict[int, str]
) -> pd.DataFrame:
    """
    Computes RMSE and chemical distance for test set glasses, returning 
    the results as a sorted DataFrame.

    Parameters
    ----------
    compounds : List[str]
        List of chemical components used for distance calculation.
    viscosity_df : pd.DataFrame
        Complete viscosity dataset containing compositions and experimental data.
    predictions_df : pd.DataFrame
        DataFrame containing predicted parameters. Its index MUST contain Test IDs.
    names_dict : Dict[int, str]
        Dictionary mapping glass IDs to their formatted string names.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing ['ID', 'Name', 'Distance', 'RMSE'], 
        sorted by RMSE in descending order (highest error first).
    """
    logger.info("Extracting distance and RMSE metrics into a DataFrame.")
    
    # Automatically separate Test and Train IDs
    test_ids = predictions_df.index.unique().to_numpy()
    all_ids = viscosity_df['ID'].unique()
    train_ids = np.setdiff1d(all_ids, test_ids)
    
    # Extract unique glass compositions
    unique_glasses = viscosity_df.drop_duplicates(subset=['ID']).set_index('ID')
    test_features = unique_glasses.loc[test_ids, compounds].values
    train_features = unique_glasses.loc[train_ids, compounds].values

    # Compute minimum Canberra distance from test glasses to train glasses
    min_distances = cdist(test_features, train_features, metric='canberra').min(axis=1)

    # Compute RMSE for each test glass
    rmse_list = []
    
    for glass_id in test_ids:
        mask = (viscosity_df['ID'] == glass_id)
        T_true = viscosity_df.loc[mask, 'T'].values
        y_true = viscosity_df.loc[mask, 'log_visc'].values

        params = predictions_df.loc[glass_id]
        if isinstance(params, pd.DataFrame):
            params = params.iloc[0]

        y_pred_curve = myega_equation_np(
            T=T_true, 
            log_inf=params['log_inf'], 
            Tg=params['Tg'], 
            m=params['m']
        )
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_curve))
        rmse_list.append(rmse)

    # Compile the results into a DataFrame
    names_list = [names_dict.get(gid, str(gid)) for gid in test_ids]
    
    results_df = pd.DataFrame({
        'ID': test_ids,
        'Name': names_list,
        'Distance': min_distances,
        'RMSE': rmse_list
    })
    
    # Sort by RMSE descending so the worst predictions are at the top
    results_df = results_df.sort_values(by='RMSE', ascending=False).reset_index(drop=True)
    
    return results_df