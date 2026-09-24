"""
Feature Importance Visualization Module.

Generates the signature 'S-curve' sorted weights plot. Includes built-in 
filtering for sparse models (like Elastic Net or Lasso) to remove features 
with zero weight, dynamically adjusting the S-curve to active features only.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from glass_dynamics.core.logger import logger

def plot_sorted_feature_importance(
    importance_dict: Dict[str, pd.DataFrame],
    top_n: int = 4,
    filter_zeros: bool = False,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Plot the algebraically sorted raw weights (S-curve) for each target
    side-by-side, annotating the most important features safely.

    Parameters
    ----------
    importance_dict : Dict[str, pd.DataFrame]
        Dictionary of importance DataFrames containing raw weights.
    top_n : int, default=4
        Number of most important features (by absolute magnitude) to annotate.
    filter_zeros : bool, default=False
        If True, removes features with weights effectively equal to zero 
        (e.g., eliminated by ElasticNet/Lasso sparsity) before plotting.
    figsize : Tuple[int, int], default=(15, 5)
        Overall dimensions of the generated figure.

    Returns
    -------
    plt.Figure
        The rendered matplotlib figure.
    """
    target_names = list(importance_dict.keys())
    n_targets = len(target_names)
    
    logger.info(f"Rendering sorted raw feature importance S-curves for {n_targets} targets.")
    
    fig, axes = plt.subplots(1, n_targets, figsize=figsize, layout='constrained')
    if n_targets == 1:
        axes = [axes]
        
    for ax, target, panel_label in zip(axes, target_names, ['(a)', '(b)', '(c)']):
        df = importance_dict[target]
        
        # --- NEW LOGIC: Filter zero weights for sparse models ---
        if filter_zeros:
            # We use a 1e-6 tolerance to account for floating-point precision 
            # noise in coefficients that mathematically should be zero.
            df = df[df['abs_weight'] > 1e-6].copy()
            
        n_features = len(df)
        
        # Sort weights algebraically to create the continuous S-curve
        df_algebraic = df.sort_values(by='weight', ascending=True).reset_index(drop=True)
        
        ax.plot(
            df_algebraic.index, 
            df_algebraic['weight'], 
            marker='o', 
            ms=5, 
            color='black', 
            linestyle='-', 
            linewidth=1.2
        )
        
        # Apply dynamic margins to preserve dynamic y-axis range
        ax.margins(x=0.15, y=0.25)
        
        # Extract the absolute most important features for annotation
        top_features = df.sort_values(by='abs_weight', ascending=False).head(top_n)
        
        for i, (_, row) in enumerate(top_features.iterrows()):
            feature_name = row['feature']
            weight = row['weight']
            orig_idx = int(row['original_index'])
            
            display_label = f"{feature_name} ({orig_idx})"
            plot_idx = df_algebraic[df_algebraic['feature'] == feature_name].index[0]
            
            # --- Robust Annotation Logic via Screen Points ---
            x_offset_pts = 35 if plot_idx < (n_features / 2) else -35
            y_sign = 1 if weight > 0 else -1
            y_offset_pts = y_sign * (15 + (i % 2) * 20)
            
            ax.annotate(
                display_label,
                xy=(plot_idx, weight),
                xytext=(x_offset_pts, y_offset_pts),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', lw=1, color='black'),
                fontsize='small',
                ha='left' if x_offset_pts > 0 else 'right',
                va='center'
            )
            
        # Axes formatting
        ax.set_ylabel(r'$w$', fontsize=12) if ax == axes[0] else ax.set_ylabel('')
        ax.set_xlabel(r'$f$ (sorted index)', fontsize=12)
        ax.set_title(f"{target}", fontsize=12, fontweight='bold')
        ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
        ax.text(0.05, 0.92, panel_label, transform=ax.transAxes, fontsize=12)

    return fig