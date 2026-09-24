"""
Feature Importance Analysis Module.

Extracts robust feature weights for a specific Ridge regression model.
The module averages weights over multiple randomized train/test splits (seeds) 
to provide statistically reliable importance metrics. As per Sharma et al., 
these weights are kept in their raw (unnormalized) form to reflect their 
true magnitude and algebraic impact on the target variable.
"""

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split
from glass_dynamics.models.ridge import CustomRidgeRegression
from glass_dynamics.core.logger import logger

def extract_importance(
    X: pd.DataFrame,
    y: pd.DataFrame,
    alpha: float = 0.1,
    n_splits: int = 100,
    test_size: float = 0.5,
    sort_by: str = 'magnitude'
) -> Dict[str, pd.DataFrame]:
    """
    Extract robust, unnormalized feature weights for a specific alpha value.

    Parameters
    ----------
    X : pd.DataFrame
        Complete feature matrix.
    y : pd.DataFrame
        Complete target matrix containing MYEGA parameters.
    alpha : float, default=0.1
        The specific regularization parameter to evaluate.
    n_splits : int, default=100
        Number of random train/test splits to average over (robust estimation).
    test_size : float, default=0.5
        Proportion of the dataset to include in the test split.
    sort_by : str, default='magnitude'
        How to sort the returned DataFrames. Options:
        - 'magnitude': Descending by absolute weight (most important first).
        - 'algebraic': Descending by signed weight (from negative to positive).
        - 'index': By the original feature index (0 to N-1).

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary mapping each target to a DataFrame containing:
        ['feature', 'original_index', 'weight', 'std', 'abs_weight'].
    """
    feature_names = X.columns.tolist()
    target_names = y.columns.tolist()
    n_features = len(feature_names)
    n_targets = len(target_names)
    
    logger.info(f"Extracting raw feature importance at alpha={alpha} over {n_splits} splits.")
    
    X_mat = X.to_numpy()
    y_mat = y.to_numpy()
    
    all_weights = np.zeros((n_splits, n_features, n_targets))
    model = CustomRidgeRegression(alpha=alpha, fit_intercept=True)
    
    for seed in range(n_splits):
        X_train, _, y_train, _ = train_test_split(
            X_mat, y_mat, test_size=test_size, random_state=seed
        )
        model.fit(X_train, y_train)
        
        weights = model.coef_
        if weights.ndim == 1:
            weights = weights.reshape(-1, 1)
        all_weights[seed, :, :] = weights
        
    mean_weights = np.mean(all_weights, axis=0)
    std_weights = np.std(all_weights, axis=0)
    
    results = {}
    for j, target in enumerate(target_names):
        t_mean = mean_weights[:, j]
        t_std = std_weights[:, j]
        abs_mean = np.abs(t_mean)
        
        # Build structured DataFrame using RAW weights
        df = pd.DataFrame({
            'feature': feature_names,
            'original_index': np.arange(n_features),
            'weight': t_mean,
            'std': t_std,
            'abs_weight': abs_mean
        })
        
        if sort_by == 'magnitude':
            df = df.sort_values(by='abs_weight', ascending=False)
        elif sort_by == 'algebraic':
            df = df.sort_values(by='weight', ascending=False)
        elif sort_by == 'index':
            df = df.sort_values(by='original_index', ascending=True)
            
        results[target] = df.reset_index(drop=True)
        
    return results