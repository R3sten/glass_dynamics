"""
Supervised Principal Component Regression (SPCR) Analysis Module.

Evaluates model performance across varying numbers of components (P).
Since SPCR selects components based on correlation with the target, 
the evaluation is independently optimized for each target variable.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from typing import Dict
from glass_dynamics.models.supervised_pcr import CustomSupervisedPCR
from glass_dynamics.core.logger import logger

def compute_spcr_components_metrics_target(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    p_array: np.ndarray
) -> Dict[str, pd.DataFrame]:
    """
    Compute R and R^2 metrics across different numbers of Principal Components
    using Target-Specific Supervised PCR (SPCR).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.DataFrame
        Training target matrix.
    X_test : pd.DataFrame
        Testing feature matrix.
    y_test : pd.DataFrame
        Testing target matrix.
    p_array : np.ndarray
        Array containing the number of supervised components to evaluate.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary mapping each target to a DataFrame containing
        the computed metrics ('train_R', 'test_R', 'train_R2', 'test_R2') 
        indexed by n_components.
    """
    target_names = y_train.columns.tolist()
    n_p = len(p_array)
    
    logger.info(f"Computing SPCR metrics across {n_p} configurations for {len(target_names)} targets.")
    
    metrics_dfs = {}
    
    # We iterate target-by-target because the optimal SPCR subspace 
    # changes depending on the variable we are trying to predict
    for target in target_names:
        logger.info(f"Optimizing Supervised Subspace for target: {target}")
        
        y_t_true = y_train[target].to_numpy()
        y_v_true = y_test[target].to_numpy()
        
        target_results = {
            "train_R": np.zeros(n_p),
            "test_R": np.zeros(n_p),
            "train_R2": np.zeros(n_p),
            "test_R2": np.zeros(n_p)
        }
        
        for i, p in enumerate(p_array):
            # Instantiate SPCR for the specific number of components
            model = CustomSupervisedPCR(n_components=int(p), fit_intercept=True)
            
            # Fit only on the 1D target
            model.fit(X_train, y_t_true)
            
            y_t_pred = model.predict(X_train)
            y_v_pred = model.predict(X_test)
            
            # Calculate R^2
            target_results["train_R2"][i] = r2_score(y_t_true, y_t_pred)
            target_results["test_R2"][i] = r2_score(y_v_true, y_v_pred)
            
            # Safe Pearson Calculation
            if np.std(y_t_pred) < 1e-12:
                target_results["train_R"][i] = 0.0
            else:
                target_results["train_R"][i] = pearsonr(y_t_true, y_t_pred)[0]
                
            if np.std(y_v_pred) < 1e-12:
                target_results["test_R"][i] = 0.0
            else:
                target_results["test_R"][i] = pearsonr(y_v_true, y_v_pred)[0]

        # Convert dictionary to DataFrame
        df = pd.DataFrame(target_results, index=p_array)
        df.index.name = "n_components"
        metrics_dfs[target] = df
        
    return metrics_dfs


def extract_spcr_representatives(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    n_components: int,
    variance_threshold: float = 1e-4
) -> Dict[str, pd.DataFrame]:
    """
    Extracts the regression weight, explained variance, correlation, and top 
    representative feature for each Principal Component selected by SPCR.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.DataFrame
        Training target matrix.
    n_components : int
        Number of supervised principal components to select.
    variance_threshold : float
        Minimum variance threshold to guard against noise.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary mapping each target to a DataFrame describing its optimal PCs.
    """
    target_names = y_train.columns.tolist()
    feature_names = X_train.columns.tolist()
    results = {}

    for target in target_names:
        # SPCR is target-specific, so we initialize and fit a fresh model per target
        model = CustomSupervisedPCR(
            n_components=n_components, 
            variance_threshold=variance_threshold,
            fit_intercept=True
        )
        model.fit(X_train, y_train[target])
        
        pca = model.pca_
        selected_indices = model.selected_pc_indices_
        coefs = model.regressor_.coef_
        
        pc_names = []
        weights = []
        variances = []
        correlations = []
        top_features = []
        top_loadings = []
        
        for i, pc_idx in enumerate(selected_indices):
            pc_names.append(f"PC{pc_idx + 1}")
            weights.append(coefs[i])
            variances.append(pca.explained_variance_ratio_[pc_idx] * 100)
            correlations.append(model.correlations_[pc_idx])
            
            # Find the feature with the highest absolute loading for this PC
            loadings = pca.components_[pc_idx, :]
            top_feature_idx = np.argmax(np.abs(loadings))
            
            top_features.append(feature_names[top_feature_idx])
            top_loadings.append(loadings[top_feature_idx])
            
        df = pd.DataFrame({
            'Component': pc_names,
            'Regression_Weight': weights,
            'Abs_Weight': np.abs(weights),
            'Target_Correlation': correlations,
            'Explained_Variance_%': variances,
            'Top_Representative_Feature': top_features,
            'Feature_Loading': top_loadings
        })
        
        # Sort by the absolute importance of the PC in predicting the target
        df = df.sort_values(by='Abs_Weight', ascending=False).reset_index(drop=True)
        results[target] = df
        
    return results