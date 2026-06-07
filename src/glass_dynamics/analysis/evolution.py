"""
Top Features Evolution Analysis Module.

This module tracks the behavior of the most influential features across different 
regularization strengths (alphas). It first identifies the top N features at a 
reference alpha (e.g., the final model's alpha), and then computes their exact 
signed weights across the entire alpha spectrum to assess physical robustness.
"""

import numpy as np
import pandas as pd
from typing import Dict
from glass_dynamics.models.ridge import CustomRidgeRegression
from glass_dynamics.core.logger import logger

def extract_top_features_evolution(
    X: pd.DataFrame,
    y: pd.DataFrame,
    alphas: np.ndarray,
    ref_alpha: float = 0.1,
    top_n: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Extract the weight trajectories for the top N most important features.

    Parameters
    ----------
    X : pd.DataFrame
        Complete feature matrix.
    y : pd.DataFrame
        Complete target matrix containing MYEGA parameters.
    alphas : np.ndarray
        Array of regularization strengths (alpha) to sweep over.
    ref_alpha : float, default=0.1
        The alpha value used to determine which features are the "Top N".
    top_n : int, default=5
        Number of most important features to track.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary mapping each target to a DataFrame. The DataFrame has alphas 
        as the index and the names of the top N features as columns.
    """
    feature_names = X.columns.tolist()
    target_names = y.columns.tolist()
    
    logger.info(f"Tracking evolution of top {top_n} features (identified at alpha={ref_alpha}).")
    
    # 1. Identify the Top N features at the reference alpha (e.g., 0.1)
    ref_model = CustomRidgeRegression(alpha=ref_alpha, fit_intercept=True)
    ref_model.fit(X, y)
    
    # Dictionary to store the names of the top features for each target
    top_feature_names = {}
    
    for j, target in enumerate(target_names):
        # Extract weights for the specific target
        target_weights = ref_model.coef_[:, j] if ref_model.coef_.ndim > 1 else ref_model.coef_
        
        # Get indices of the highest absolute weights
        top_indices = np.argsort(np.abs(target_weights))[-top_n:][::-1]
        top_feature_names[target] = [feature_names[i] for i in top_indices]

    # 2. Compute the exact weights across the entire alpha spectrum
    n_alphas = len(alphas)
    evolution_data = {t: np.zeros((n_alphas, top_n)) for t in target_names}
    
    for i, alpha in enumerate(alphas):
        model = CustomRidgeRegression(alpha=alpha, fit_intercept=True)
        model.fit(X, y)
        
        for j, target in enumerate(target_names):
            target_weights = model.coef_[:, j] if model.coef_.ndim > 1 else model.coef_
            
            # Extract only the weights of the identified top features
            for k, feat_name in enumerate(top_feature_names[target]):
                feat_idx = feature_names.index(feat_name)
                evolution_data[target][i, k] = target_weights[feat_idx]
                
    # 3. Format outputs into clean Pandas DataFrames
    results = {}
    for target in target_names:
        df = pd.DataFrame(
            evolution_data[target], 
            index=alphas, 
            columns=top_feature_names[target]
        )
        df.index.name = 'alpha'
        results[target] = df
        
    return results