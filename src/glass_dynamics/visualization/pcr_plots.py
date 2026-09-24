"""
PCR Performance Visualization Module.

Generates minimalist plots tracking model performance metrics (R and R^2) 
as a function of the number of Principal Components (P).
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
from glass_dynamics.core.logger import logger

def plot_pcr_performance(
    metrics_df: pd.DataFrame, 
    target_name: str, 
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot R and R^2 metrics in two square subplots side-by-side as a function 
    of the number of principal components.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame containing performance metrics, indexed by n_components.
    target_name : str
        Name of the target variable for the plot title.
    figsize : Tuple[int, int], default=(12, 5)
        Overall dimensions of the figure.

    Returns
    -------
    plt.Figure
        The rendered matplotlib figure.
    """
    logger.info(f"Rendering PCR performance plots for {target_name}.")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    p_values = metrics_df.index.to_numpy()
    
    train_style = {"color": "black", "linestyle": "-", "linewidth": 2, "label": r'training'}
    test_style = {"color": "black", "linestyle": "--", "linewidth": 2, "label": r'test'}
    
    # --- Left Plot: Pearson R ---
    ax1.plot(p_values, metrics_df["train_R"], **train_style)
    ax1.plot(p_values, metrics_df["test_R"], **test_style)
    
    ax1.set_xlabel(r"Number of Principal Components ($P$)", fontsize=12)
    ax1.set_ylabel(r"Pearson Correlation ($R$)", fontsize=12)
    ax1.set_box_aspect(1)
    ax1.legend(loc="lower right", frameon=False)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- Right Plot: R^2 ---
    ax2.plot(p_values, metrics_df["train_R2"], **train_style)
    ax2.plot(p_values, metrics_df["test_R2"], **test_style)
    
    ax2.set_xlabel(r"Number of Principal Components ($P$)", fontsize=12)
    ax2.set_ylabel(r"Coefficient of Determination ($R^2$)", fontsize=12)
    ax2.set_box_aspect(1)
    ax2.legend(loc="lower right", frameon=False)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    fig.suptitle(f"PCR Performance vs Components: {target_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    return fig