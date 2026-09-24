"""
Feature Importance Analysis Module.

Extracts robust feature weights for ANY linear regression model (Ridge, ElasticNet, PCR).
The module averages weights over multiple randomized train/test splits (seeds) 
to provide statistically reliable importance metrics. Weights are kept in their 
raw (unnormalized) form to reflect their true algebraic impact on the target variable.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from glass_dynamics.core.logger import logger

def extract_importance(
    X: pd.DataFrame,
    y: pd.DataFrame,
    base_model: Any,
    n_splits: int = 100,
    test_size: float = 0.5,
    sort_by: str = 'magnitude'
) -> Dict[str, pd.DataFrame]:
    """
    Extract robust, unnormalized feature weights using any regression model.

    Parameters
    ----------
    X : pd.DataFrame
        Complete feature matrix.
    y : pd.DataFrame
        Complete target matrix containing MYEGA parameters.
    base_model : BaseEstimator
        An un-fitted regression model (e.g., CustomRidge, CustomElasticNet, CustomPCR)
        that exposes a '.coef_' attribute after fitting.
    n_splits : int, default=100
        Number of random train/test splits to average over (robust estimation).
    test_size : float, default=0.5
        Proportion of the dataset to include in the test split.
    sort_by : str, default='magnitude'
        How to sort the returned DataFrames. Options:
        - 'magnitude': Descending by absolute weight (most important first).
        - 'algebraic': Descending by signed weight (from positive to negative).
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
    
    model_name = base_model.__class__.__name__
    logger.info(f"Extracting robust features using {model_name} over {n_splits} splits.")
    
    X_mat = X.to_numpy()
    y_mat = y.to_numpy()
    
    # Shape: (splits, features, targets)
    all_weights = np.zeros((n_splits, n_features, n_targets))
    
    for seed in range(n_splits):
        # 1. Random split
        X_train, _, y_train, _ = train_test_split(
            X_mat, y_mat, test_size=test_size, random_state=seed
        )
        
        # 2. Clone the model to ensure a fresh, unfitted estimator
        model = clone(base_model)
        model.fit(X_train, y_train)
        
        # 3. Safely extract coefficients
        if not hasattr(model, 'coef_'):
            raise AttributeError(f"Model {model_name} does not expose 'coef_'.")
            
        weights = model.coef_
        
        # Standardize scikit-learn coef_ shape to match (n_features, n_targets)
        if weights.ndim == 1:
            weights = weights.reshape(-1, 1)
        elif weights.shape == (n_targets, n_features):
            weights = weights.T  # Transpose if shape is (targets, features)
            
        all_weights[seed, :, :] = weights
        
    # Calculate statistics across all splits
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
        
        # Apply sorting logic
        if sort_by == 'magnitude':
            df = df.sort_values(by='abs_weight', ascending=False)
        elif sort_by == 'algebraic':
            df = df.sort_values(by='weight', ascending=False)
        elif sort_by == 'index':
            df = df.sort_values(by='original_index', ascending=True)
            
        results[target] = df.reset_index(drop=True)
        
    return results