"""
Principal Component Regression (PCR) Analysis Module.

Provides functions to extract representative features for each Principal Component (PC)
and to evaluate model performance across varying numbers of components.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from typing import Dict, List
from glass_dynamics.models.pcr import CustomPCR
from glass_dynamics.core.logger import logger

def extract_pc_representatives(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    n_components: int
) -> Dict[str, pd.DataFrame]:
    """
    Extracts the regression weight, explained variance, and top representative 
    feature for each Principal Component.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.DataFrame
        Training target matrix.
    n_components : int
        Number of principal components to fit.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary mapping each target to a DataFrame describing the PCs.
    """
    target_names = y_train.columns.tolist()
    feature_names = X_train.columns.tolist()
    
    # Fit the PCR model
    model = CustomPCR(n_components=n_components, fit_intercept=True)
    model.fit(X_train, y_train)
    
    pca = model.pca_
    regressor = model.regressor_
    
    # pca.components_ shape: (n_components, n_features)
    # regressor.coef_ shape: (n_targets, n_components)
    
    results = {}
    for i, target in enumerate(target_names):
        pc_indices = []
        target_weights = []
        explained_variances = []
        top_features = []
        top_loadings = []
        
        target_coefs = regressor.coef_[i, :]
        
        for pc_idx in range(n_components):
            pc_indices.append(f"PC{pc_idx + 1}")
            target_weights.append(target_coefs[pc_idx])
            explained_variances.append(pca.explained_variance_ratio_[pc_idx] * 100)
            
            # Find the feature with the highest absolute loading for this PC
            loadings = pca.components_[pc_idx, :]
            top_feature_idx = np.argmax(np.abs(loadings))
            
            top_features.append(feature_names[top_feature_idx])
            top_loadings.append(loadings[top_feature_idx])
            
        df = pd.DataFrame({
            'Component': pc_indices,
            'Regression_Weight': target_weights,
            'Abs_Weight': np.abs(target_weights),
            'Explained_Variance_%': explained_variances,
            'Top_Representative_Feature': top_features,
            'Feature_Loading': top_loadings
        })
        
        # Sort by the absolute importance of the PC in predicting the target
        df = df.sort_values(by='Abs_Weight', ascending=False).reset_index(drop=True)
        results[target] = df
        
    return results

def compute_pcr_components_metrics(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    p_array: np.ndarray
) -> Dict[str, pd.DataFrame]:
    """
    Compute R and R^2 metrics across different numbers of Principal Components.
    """
    target_names = y_train.columns.tolist()
    n_p = len(p_array)
    
    logger.info(f"Computing PCR metrics across {n_p} component configurations.")
    
    results_data = {
        target: {
            "train_R": np.zeros(n_p),
            "test_R": np.zeros(n_p),
            "train_R2": np.zeros(n_p),
            "test_R2": np.zeros(n_p)
        } for target in target_names
    }
    
    for i, p in enumerate(p_array):
        # Convert P to int (required by scikit-learn PCA)
        model = CustomPCR(n_components=int(p), fit_intercept=True)
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        for j, target in enumerate(target_names):
            y_t_true = y_train.iloc[:, j].to_numpy()
            y_t_pred = y_train_pred[:, j]
            y_v_true = y_test.iloc[:, j].to_numpy()
            y_v_pred = y_test_pred[:, j]
            
            results_data[target]["train_R2"][i] = r2_score(y_t_true, y_t_pred)
            results_data[target]["test_R2"][i] = r2_score(y_v_true, y_v_pred)
            
            # Safe Pearson calculation
            if np.std(y_t_pred) < 1e-12:
                results_data[target]["train_R"][i] = 0.0
            else:
                results_data[target]["train_R"][i] = pearsonr(y_t_true, y_t_pred)[0]
                
            if np.std(y_v_pred) < 1e-12:
                results_data[target]["test_R"][i] = 0.0
            else:
                results_data[target]["test_R"][i] = pearsonr(y_v_true, y_v_pred)[0]

    metrics_dfs = {}
    for target in target_names:
        df = pd.DataFrame(results_data[target], index=p_array)
        df.index.name = "n_components"
        metrics_dfs[target] = df
        
    return metrics_dfs