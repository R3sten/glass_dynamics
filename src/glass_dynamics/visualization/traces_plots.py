"""
Robust Ridge Traces Visualization Module.

This module replicates the specific visualization style from Sharma et al.
for plotting Ridge regression weight trajectories. It includes dynamic 
normalization (where weights and errors are scaled by the maximum absolute 
weight at each alpha step) and renders standard deviation bands using 
colormap-mapped feature indices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.ticker import LogLocator
from typing import Tuple
from glass_dynamics.core.logger import logger

def plot_normalized_traces(
    mean_weights: pd.DataFrame,
    std_weights: pd.DataFrame,
    target_name: str,
    cmap_name: str = 'CMRmap',
    figsize: Tuple[int, int] = (9, 6)
) -> plt.Figure:
    """
    Plot normalized Ridge regression traces with error bands.
    
    At each alpha value, all weights and standard deviations are normalized 
    by the maximum absolute weight present at that specific alpha. This highlights 
    the relative importance of features as regularization increases.

    Parameters
    ----------
    mean_weights : pd.DataFrame
        DataFrame of mean coefficients, indexed by alpha.
    std_weights : pd.DataFrame
        DataFrame of standard deviations, indexed by alpha.
    target_name : str
        The name of the target variable (e.g., 'Tg_fitted') for titling.
    cmap_name : str, default='CMRmap'
        The name of the matplotlib colormap to use, matching the paper's style.
    figsize : Tuple[int, int], default=(9, 6)
        Dimensions of the generated figure.

    Returns
    -------
    plt.Figure
        The rendered matplotlib figure object.
    """
    logger.info(f"Rendering robust normalized traces for target: {target_name}")
    
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    
    alphas = mean_weights.index.to_numpy()
    features = mean_weights.columns.tolist()
    n_features = len(features)
    
    # 1. Normalization Step (Matching the snippet's logic but using vectorization)
    # Extract the maximum absolute weight at each alpha (row-wise max)
    max_abs_weights = mean_weights.abs().max(axis=1)
    
    # Normalize means and errors by dividing row elements by the row's max absolute value
    norm_mean = mean_weights.div(max_abs_weights, axis=0)
    norm_std = std_weights.div(max_abs_weights, axis=0)
    
    # 2. Colormap Configuration
    # We sample the colormap to get distinct colors for each feature.
    # We add +5 to the resample size (as done in the snippet) to avoid the 
    # absolute extremes of the colormap (which might be pure white/black).
    base_cmap = plt.colormaps.get_cmap(cmap_name).resampled(n_features + 5)
    colors = base_cmap(np.arange(n_features))
    
    # 3. Plotting the lines and error bands
    for i, feature in enumerate(features):
        mean_curve = norm_mean[feature].to_numpy()
        std_curve = norm_std[feature].to_numpy()
        color = colors[i]
        
        # Plot the main trajectory
        ax.plot(alphas, mean_curve, '-', color=color, linewidth=1.5)
        
        # Plot the shaded error area (mean + std, mean - std)
        ax.fill_between(
            alphas, 
            mean_curve + std_curve, 
            mean_curve - std_curve, 
            alpha=0.2, 
            color=color,
            linewidth=0
        )

    # 4. Axes Formatting (Matching the paper aesthetics)
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.set_xlabel(r'$\alpha$', fontsize=14)
    ax.set_ylabel(r'Normalized weights $w / \max(|w|)$', fontsize=14)
    ax.set_xlim([alphas.min(), alphas.max()])
    
    # Set major and minor log locators
    major_locator = LogLocator(numticks=9)
    ax.xaxis.set_major_locator(major_locator)
    ax.set_xticks([1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2], minor=True)
    ax.set_xticklabels(['']*6, minor=True) # Hide minor labels
    
    # Draw a horizontal line at 0 for visual reference
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0.1, color="grey", linestyle=":", linewidth=1.5, alpha=0.8)
    
    # 5. Colorbar generation
    # Maps the feature index to the colormap purely for visual reference
    norm = mcolors.Normalize(vmin=1, vmax=n_features)
    sm = cm.ScalarMappable(norm=norm, cmap=base_cmap)
    cbar = fig.colorbar(sm, ax=ax, aspect=30)
    cbar.set_label('Feature Index', fontsize=12)
    
    # Title
    ax.set_title(f"Ridge Traces: {target_name}", fontsize=14, fontweight='bold')
    
    return fig