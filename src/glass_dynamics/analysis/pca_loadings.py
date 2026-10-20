"""
PCA Loadings Analysis Module.

Extracts the top contributing features for a given Principal Component.
"""

import numpy as np
import pandas as pd
from typing import Any, List
from glass_dynamics.core.logger import logger

def get_top_pc_features(
    model: Any, 
    feature_names: List[str], 
    pc_index: int = 1, 
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extracts the top N features with the highest absolute loadings for a specific PC.
    
    Parameters
    ----------
    model : Any
        Fitted PCA model or a custom pipeline containing a fitted PCA.
    feature_names : List[str]
        List of the original feature names (must match the training columns).
    pc_index : int, default=1
        The Principal Component to analyze (1-based index: 1 = PC1).
    top_n : int, default=20
        The number of top features to return.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing 'Feature', 'Loading', and 'Abs_Loading'.
    """
    logger.info(f"Extracting top {top_n} features for PC{pc_index}.")
    
    # Riconosce automaticamente dove si trova la matrice delle componenti
    if hasattr(model, 'components_'):
        components = model.components_
    elif hasattr(model, 'pca_') and hasattr(model.pca_, 'components_'):
        components = model.pca_.components_
    else:
        raise ValueError("Il modello fornito non possiede l'attributo 'components_'. "
                         "Assicurati che sia stato addestrato e che esponga la PCA.")
        
    # pc_index è 1-based (l'utente chiede PC1, in Python è l'indice 0)
    actual_idx = pc_index - 1
    
    if actual_idx >= len(components):
        raise ValueError(f"PC{pc_index} non esiste. Il modello ha solo {len(components)} componenti.")
        
    loadings = components[actual_idx]
    
    df = pd.DataFrame({
        'Feature': feature_names,
        'Loading': loadings,
        'Abs_Loading': np.abs(loadings)
    })
    
    # Ordina per magnitudo assoluta e prendi le prime N
    df_top = df.sort_values(by='Abs_Loading', ascending=False).head(top_n)
    
    return df_top.reset_index(drop=True)