"""
Robust Ridge Traces Analysis Module.

This module replicates the methodology from "Ridge Regression in Practice" 
(Marquardt & Snee), as implemented by Sharma et al. 
For each regularization parameter (alpha), the model is evaluated across multiple 
randomized train-test splits (Monte Carlo cross-validation). 
The final regression weights are averaged to provide robust estimates, 
and their standard deviations are extracted to quantify the uncertainty 
arising from the data sampling.
"""

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split
from glass_dynamics.models.ridge import CustomRidgeRegression
from glass_dynamics.core.logger import logger

def compute_weight_traces(
    X: pd.DataFrame,
    y: pd.DataFrame,
    alphas: np.ndarray,
    n_splits: int = 20,
    test_size: float = 0.5
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Compute robust Ridge regression weights by averaging over multiple random 
    train/test splits for each alpha value.

    Parameters
    ----------
    X : pd.DataFrame
        Complete feature matrix (samples x features).
    y : pd.DataFrame
        Complete target matrix containing MYEGA parameters (samples x targets).
    alphas : np.ndarray
        Array of regularization strengths (alpha) to sweep over.
    n_splits : int, default=20
        Number of random train/test splits to perform for each alpha.
    test_size : float, default=0.5
        Proportion of the dataset to include in the test split (default 50%).

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        A dictionary containing the robust traces for each target.
        Structure:
        {
            'target_name': {
                'mean_weights': pd.DataFrame (alphas as index, features as columns),
                'std_weights': pd.DataFrame (alphas as index, features as columns)
            }
        }
    """
    feature_names = X.columns.tolist()
    target_names = y.columns.tolist()
    n_features = len(feature_names)
    n_targets = len(target_names)
    n_alphas = len(alphas)
    
    logger.info(f"Computing robust weight traces across {n_alphas} alphas.")
    logger.info(f"Averaging over {n_splits} random splits (test_size={test_size}).")
    
    # Pre-allocate dictionaries containing numpy arrays for fast writing
    mean_data = {t: np.zeros((n_alphas, n_features)) for t in target_names}
    std_data = {t: np.zeros((n_alphas, n_features)) for t in target_names}
    
    # Extract underlying numpy arrays for high-speed indexing during splits
    X_mat = X.to_numpy()
    y_mat = y.to_numpy()
    
    for i, alpha in enumerate(alphas):
        # Temporary array to hold weights from all random seeds for the current alpha
        # Shape: (n_splits, n_features, n_targets)
        current_alpha_weights = np.zeros((n_splits, n_features, n_targets))
        
        # Instantiate the custom model once per alpha
        model = CustomRidgeRegression(alpha=alpha, fit_intercept=True)
        
        for seed in range(n_splits):
            # 1. Random split using the current seed
            X_train, _, y_train, _ = train_test_split(
                X_mat, y_mat, test_size=test_size, random_state=seed
            )
            
            # 2. Fit model on the randomized training subset
            model.fit(X_train, y_train)
            
            # 3. Store the weights. (Handle single vs multi-target shapes implicitly)
            weights = model.coef_
            if weights.ndim == 1:
                weights = weights.reshape(-1, 1)
                
            current_alpha_weights[seed, :, :] = weights
            
        # 4. Compute the mean and standard deviation across the seed axis (axis 0)
        alpha_mean_weights = np.mean(current_alpha_weights, axis=0) # Shape: (P, T)
        alpha_std_weights = np.std(current_alpha_weights, axis=0)   # Shape: (P, T)
        
        # 5. Distribute the computed statistics to the target-specific dictionaries
        for j, target in enumerate(target_names):
            mean_data[target][i, :] = alpha_mean_weights[:, j]
            std_data[target][i, :] = alpha_std_weights[:, j]

    logger.info("Robust traces computation completed. Packaging results.")
    
    # Convert results into labeled pandas DataFrames for intuitive plotting
    results = {}
    for target in target_names:
        mean_df = pd.DataFrame(mean_data[target], index=alphas, columns=feature_names)
        mean_df.index.name = "alpha"
        
        std_df = pd.DataFrame(std_data[target], index=alphas, columns=feature_names)
        std_df.index.name = "alpha"
        
        results[target] = {
            "mean_weights": mean_df,
            "std_weights": std_df
        }
        
    return results