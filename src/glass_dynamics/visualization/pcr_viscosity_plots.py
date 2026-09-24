"""
Global Viscosity Dimensionality Visualization Module.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
from glass_dynamics.core.logger import logger

def plot_global_viscosity_p_grid(
    metrics_df: pd.DataFrame,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plots the global viscosity R^2 and RMSE vs the number of Principal Components
    for both training and test sets in square subplots.
    """
    logger.info(f"Rendering global viscosity dimensionality plots for {model_name}.")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, layout='constrained')
    p_values = metrics_df.index.to_numpy()
    
    train_style = {"color": "black", "linestyle": "-", "linewidth": 2, "label": r'training'}
    test_style = {"color": "black", "linestyle": "--", "linewidth": 2, "label": r'test'}
    
    # --- Left Plot: R^2 ---
    ax1.plot(p_values, metrics_df["Train_R2"], **train_style)
    ax1.plot(p_values, metrics_df["Test_R2"], **test_style)
    
    ax1.set_xlabel(r"Number of Principal Components ($P$)", fontsize=12)
    ax1.set_ylabel(r"Viscosity $R^2$", fontsize=12)
    ax1.set_box_aspect(1) # Forces square aspect ratio
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Cap y-axis to prevent extreme negative outliers from breaking the visual scale
    y_min = max(-1.0, metrics_df["Test_R2"].min() * 1.1)
    ax1.set_ylim(bottom=y_min, top=1.05)
    ax1.legend(loc="lower right", frameon=False)
    
    # --- Right Plot: RMSE ---
    ax2.plot(p_values, metrics_df["Train_RMSE"], **train_style)
    ax2.plot(p_values, metrics_df["Test_RMSE"], **test_style)
    
    ax2.set_xlabel(r"Number of Principal Components ($P$)", fontsize=12)
    ax2.set_ylabel(r"Viscosity RMSE ($\log_{10} \eta$)", fontsize=12)
    ax2.set_box_aspect(1) # Forces square aspect ratio
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc="upper right", frameon=False)
    
    fig.suptitle(f"Global Viscosity Predictive Performance vs $P$ ({model_name})", fontsize=14, fontweight="bold")
    
    return fig