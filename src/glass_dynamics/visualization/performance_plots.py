"""
Performance Visualization Module.

Generates minimalist, greyscale/black-and-white plots for model performance 
metrics, formatted as square panels side-by-side.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
from glass_dynamics.core.logger import logger

def plot_target_performance(
    metrics_df: pd.DataFrame, 
    target_name: str, 
    ref_alpha: float = 0.1,
    figsize: Tuple[int, int] = (10, 5)
) -> plt.Figure:
    """
    Plot R and R^2 metrics in two square subplots side-by-side using a strict 
    black and white aesthetic.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame containing performance metrics, indexed by alpha.
    target_name : str
        Name of the target variable for the plot title.
    ref_alpha : float, default=0.1
        Alpha value to mark with a vertical dashed reference line.
    figsize : Tuple[int, int], default=(10, 5)
        Overall dimensions of the figure.

    Returns
    -------
    plt.Figure
        The rendered matplotlib figure.
    """
    logger.info(f"Rendering performance plots (R, R^2) for {target_name}.")
    
    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    alphas = metrics_df.index.to_numpy()
    
    # --- Styling dictionaries for clean B&W aesthetic ---
    train_style = {
        "color": "black", 
        # "marker": "o", 
        "linestyle": "-", 
        "linewidth": 2, 
        # "markersize": 6,
        # "markerfacecolor": "black",
        "label": r'training'
    }
    
    test_style = {
        "color": "black", 
        # "marker": "o", 
        "linestyle": "--", 
        "linewidth": 2, 
        # "markersize": 6,
        # "markerfacecolor": "white", # Open circle
        "label": r'test'
    }
    
    # --- Left Plot: Pearson R ---
    ax1.plot(alphas, metrics_df["train_R"], **train_style)
    ax1.plot(alphas, metrics_df["test_R"], **test_style)
    
    ax1.axvline(x=ref_alpha, color="grey", linestyle=":", linewidth=1.5, alpha=0.8)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\alpha$", fontsize=12)
    ax1.set_ylabel(r"$R$", fontsize=12)
    ax1.set_box_aspect(1) # Forces the plot area to be a perfect square
    ax1.legend(loc="best", frameon=False)
    
    # --- Right Plot: R^2 ---
    ax2.plot(alphas, metrics_df["train_R2"], **train_style)
    ax2.plot(alphas, metrics_df["test_R2"], **test_style)
    
    ax2.axvline(x=ref_alpha, color="grey", linestyle=":", linewidth=1.5, alpha=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\alpha$", fontsize=12)
    ax2.set_ylabel(r"$R^2$", fontsize=12)
    ax2.set_box_aspect(1) # Forces the plot area to be a perfect square
    ax2.legend(loc="best", frameon=False)
    
    # Main title for the figure
    fig.suptitle(f"Performance Metrics: {target_name}", fontsize=14, fontweight="bold")
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    return fig