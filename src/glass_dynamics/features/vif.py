import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from glass_dynamics.core.logger import logger

def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Variance Inflation Factor (VIF) for each feature using standard OLS.
    VIF > 10 usually indicates high multicollinearity.
    
    Args:
        X (pd.DataFrame): DataFrame of features (should be scaled/standardized first).
        
    Returns:
        pd.DataFrame: A dataframe with 'Feature' and 'VIF' columns, sorted by VIF descending.
    """
    logger.info("Calculating standard OLS VIF for features...")
    vif_data = []
    
    for feature in X.columns:
        y_target = X[feature]
        X_others = X.drop(columns=[feature])
        
        # Fit OLS
        model = LinearRegression()
        model.fit(X_others, y_target)
        r2 = model.score(X_others, y_target)
        
        # Protect against division by zero if R2 is exactly 1
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        vif_data.append({"Feature": feature, "VIF": vif})
        
    df_vif = pd.DataFrame(vif_data).sort_values(by="VIF", ascending=False).reset_index(drop=True)
    return df_vif

def calculate_ridge_vif(X: pd.DataFrame, alpha: float = 1.0) -> pd.DataFrame:
    """
    Calculates a regularized Variance Inflation Factor (VIF) using Ridge Regression.
    Useful for studying how regularization stabilizes feature variance in highly collinear datasets.
    
    Args:
        X (pd.DataFrame): DataFrame of features (MUST be scaled/standardized).
        alpha (float): Regularization strength for Ridge.
        
    Returns:
        pd.DataFrame: A dataframe with 'Feature', 'Ridge_VIF', and the specific alpha used.
    """
    logger.info(f"Calculating Ridge VIF with alpha={alpha}...")
    vif_data = []
    
    for feature in X.columns:
        y_target = X[feature]
        X_others = X.drop(columns=[feature])
        
        # Fit Ridge instead of OLS
        model = Ridge(alpha=alpha)
        model.fit(X_others, y_target)
        
        # Note: In Ridge, R^2 can sometimes be slightly negative if the alpha is huge,
        # but for VIF interpretation we clip it to 0 minimum.
        r2 = model.score(X_others, y_target)
        r2 = max(0.0, r2) 
        
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        vif_data.append({"Feature": feature, "Ridge_VIF": vif, "Alpha": alpha})
        
    df_vif = pd.DataFrame(vif_data).sort_values(by="Ridge_VIF", ascending=False).reset_index(drop=True)
    return df_vif


def iterative_vif_selection(X: pd.DataFrame, threshold: float = 5.0) -> list:
    """
    Iteratively removes features with the highest standard Variance Inflation Factor (VIF)
    until the maximum VIF in the dataset falls below the specified threshold.
    
    Args:
        X (pd.DataFrame): DataFrame of scaled/standardized features.
        threshold (float): The VIF threshold (typically 5.0 or 10.0).
        
    Returns:
        list: A list of feature names that survived the VIF selection.
    """
    features_to_keep = list(X.columns)
    logger.info(f"Starting iterative Standard VIF selection (threshold = {threshold}). This may take a while...")
    
    while len(features_to_keep) > 1:
        X_current = X[features_to_keep]
        vifs = []
        
        for feature in features_to_keep:
            y_target = X_current[feature]
            X_others = X_current.drop(columns=[feature])
            
            # Use sklearn's LinearRegression (fast and stable)
            model = LinearRegression()
            model.fit(X_others, y_target)
            r2 = model.score(X_others, y_target)
            
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
            vifs.append(vif)
            
        max_vif = max(vifs)
        max_idx = vifs.index(max_vif)
        worst_feat = features_to_keep[max_idx]
        
        if max_vif > threshold:
            logger.debug(f"Dropping '{worst_feat}' (VIF: {max_vif:.2f})")
            features_to_keep.remove(worst_feat)
        else:
            logger.info(f"Stability found! Maximum final VIF is: {max_vif:.2f}")
            break
            
    logger.info(f"Number of selected features after Standard VIF: {len(features_to_keep)}")
    return features_to_keep


def fast_iterative_vif_selection(X: pd.DataFrame, threshold: float = 5.0) -> list:
    """
    Iteratively removes features with a Variance Inflation Factor (VIF) above a specified threshold.

    This implementation leverages the Moore-Penrose pseudo-inverse of the correlation matrix
    to compute all VIFs simultaneously, offering significant performance improvements over
    traditional iterative Ordinary Least Squares (OLS) fitting. It includes robust handling
    for zero-variance features and collinearity-induced numerical instabilities.

    Args:
        X (pd.DataFrame): The input dataset containing the features to evaluate.
        threshold (float, optional): The maximum acceptable VIF. Features with a VIF strictly
            greater than this value will be iteratively removed. Defaults to 5.0.

    Returns:
        List[str]: A list of feature names that satisfy the VIF threshold condition.

    Raises:
        ValueError: If the input DataFrame is empty.
    """
    if X.empty:
        raise ValueError("Input DataFrame X cannot be empty.")

    # Identify and drop constant features (variance == 0) prior to correlation matrix computation.
    # Constant features induce division by zero during Pearson correlation calculation, yielding NaNs.
    variances = X.var()
    constant_features = variances[variances == 0].index.tolist()
    
    if constant_features:
        logger.warning(f"Dropping {len(constant_features)} zero-variance features prior to VIF selection.")
        X = X.drop(columns=constant_features)

    features_to_keep = list(X.columns)
    logger.info(f"Initiating fast VIF selection (threshold={threshold}) on {len(features_to_keep)} features.")

    while len(features_to_keep) > 1:
        # Extract raw numpy array for optimized matrix operations
        X_current = X[features_to_keep].values

        # Compute the Pearson correlation matrix.
        # rowvar=False denotes that columns represent variables and rows represent observations.
        corr_matrix = np.corrcoef(X_current, rowvar=False)

        # Sanitize the correlation matrix to prevent SVD non-convergence.
        # Near-zero divisions in highly collinear datasets can introduce NaNs.
        if np.isnan(corr_matrix).any():
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        try:
            # Calculate the pseudo-inverse to handle perfectly collinear feature pairs gracefully.
            inv_corr_matrix = np.linalg.pinv(corr_matrix)
        except np.linalg.LinAlgError:
            # Fallback for persistent SVD non-convergence
            logger.error("SVD failed to converge. Ejecting the last evaluated feature to break deadlock.")
            features_to_keep.pop()
            continue

        # The diagonal of the inverted correlation matrix contains the VIF for each feature.
        vifs = np.diag(inv_corr_matrix)

        max_vif = np.max(vifs)
        max_idx = np.argmax(vifs)
        worst_feat = features_to_keep[max_idx]

        if max_vif > threshold:
            features_to_keep.remove(worst_feat)
        else:
            logger.info(f"VIF selection converged. Maximum final VIF: {max_vif:.2f}")
            break

    logger.info(f"VIF selection complete. Retained {len(features_to_keep)} features.")
    
    return features_to_keep


def iterative_ridge_vif_selection(X: pd.DataFrame, alpha: float = 1.0, threshold: float = 5.0) -> list:
    """
    Iteratively removes features using a Ridge-regularized VIF calculation.
    This stabilizes the selection process when extreme multicollinearity is present.
    
    Args:
        X (pd.DataFrame): DataFrame of scaled/standardized features.
        alpha (float): Regularization strength for the Ridge model.
        threshold (float): The VIF threshold.
        
    Returns:
        list: A list of feature names that survived the Ridge VIF selection.
    """
    features_to_keep = list(X.columns)
    logger.info(f"Starting iterative Ridge VIF selection (alpha={alpha}, threshold={threshold})...")
    
    while len(features_to_keep) > 1:
        X_current = X[features_to_keep]
        vifs = []
        
        for feature in features_to_keep:
            y_target = X_current[feature]
            X_others = X_current.drop(columns=[feature])
            
            model = Ridge(alpha=alpha)
            model.fit(X_others, y_target)
            
            # Ridge R2 can occasionally dip slightly below 0 for bad fits; clip it to 0
            r2 = max(0.0, model.score(X_others, y_target))
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
            vifs.append(vif)
            
        max_vif = max(vifs)
        max_idx = vifs.index(max_vif)
        worst_feat = features_to_keep[max_idx]
        
        if max_vif > threshold:
            logger.debug(f"Dropping '{worst_feat}' (Ridge VIF: {max_vif:.2f})")
            features_to_keep.remove(worst_feat)
        else:
            logger.info(f"Stability found! Maximum final Ridge VIF is: {max_vif:.2f}")
            break
            
    logger.info(f"Number of selected features after Ridge VIF: {len(features_to_keep)}")
    return features_to_keep