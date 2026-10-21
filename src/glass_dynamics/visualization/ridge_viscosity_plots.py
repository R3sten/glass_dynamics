"""
Ridge Global Viscosity Alpha Grid Visualization Module.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
from glass_dynamics.core.logger import logger

def plot_ridge_viscosity_alpha_grid(
    metrics_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plots the global viscosity R^2 and RMSE vs the regularization penalty alpha
    for both training and test sets in square subplots with a logarithmic x-axis.
    """
    logger.info("Rendering Ridge global viscosity alpha grid plots.")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, layout='constrained')
    alpha_values = metrics_df.index.to_numpy()
    
    train_style = {"color": "black", "linestyle": "-", "linewidth": 2, "label": r'training'}
    test_style = {"color": "black", "linestyle": "--", "linewidth": 2, "label": r'test'}
    
    # --- Left Plot: R^2 ---
    ax1.plot(alpha_values, metrics_df["Train_R2"], **train_style)
    ax1.plot(alpha_values, metrics_df["Test_R2"], **test_style)
    
    ax1.set_xscale('log')
    ax1.set_xlabel(r"Regularization Penalty ($\alpha$)", fontsize=12)
    ax1.set_ylabel(r"Viscosity $R^2$", fontsize=12)
    ax1.set_box_aspect(1)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    y_min = max(-1.0, metrics_df["Test_R2"].min() * 1.1)
    ax1.set_ylim(bottom=y_min, top=1.05)
    ax1.legend(loc="lower left", frameon=False)
    
    # --- Right Plot: RMSE ---
    ax2.plot(alpha_values, metrics_df["Train_RMSE"], **train_style)
    ax2.plot(alpha_values, metrics_df["Test_RMSE"], **test_style)
    
    ax2.set_xscale('log')
    ax2.set_xlabel(r"Regularization Penalty ($\alpha$)", fontsize=12)
    ax2.set_ylabel(r"Viscosity RMSE ($\log_{10} \eta$)", fontsize=12)
    ax2.set_box_aspect(1)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc="upper left", frameon=False)
    
    fig.suptitle(r"Global Viscosity Predictive Performance vs Ridge $\alpha$", fontsize=14, fontweight="bold")
    
    return fig


def plot_ridge_viscosity_alpha_grid_cross_validation(
    metrics_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plots the global viscosity R^2 and RMSE vs the regularization penalty alpha
    for both training and cross-validation sets in square subplots with a logarithmic x-axis.
    """
    logger.info("Rendering Ridge global viscosity CV alpha grid plots.")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, layout='constrained')
    alpha_values = metrics_df.index.to_numpy()
    
    train_style = {"color": "black", "linestyle": "-", "linewidth": 2, "label": r"training"}
    cv_style = {"color": "black", "linestyle": "--", "linewidth": 2, "label": r"validation (5-fold CV)"}
    
    # Backward compatibility: support both 'CV_R2' and legacy 'Test_R2' column names
    val_r2_col = "CV_R2" if "CV_R2" in metrics_df.columns else "Test_R2"
    val_rmse_col = "CV_RMSE" if "CV_RMSE" in metrics_df.columns else "Test_RMSE"

    # --- Left Plot: R^2 ---
    ax1.plot(alpha_values, metrics_df["Train_R2"], **train_style)
    ax1.plot(alpha_values, metrics_df[val_r2_col], **cv_style)
    
    ax1.set_xscale('log')
    ax1.set_xlabel(r"Regularization Penalty ($\alpha$)", fontsize=12)
    ax1.set_ylabel(r"Viscosity $R^2$", fontsize=12)
    ax1.set_box_aspect(1)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    y_min = max(-1.0, metrics_df[val_r2_col].min() * 1.1)
    ax1.set_ylim(bottom=y_min, top=1.05)
    ax1.legend(loc="lower left", frameon=False)
    
    # --- Right Plot: RMSE ---
    ax2.plot(alpha_values, metrics_df["Train_RMSE"], **train_style)
    ax2.plot(alpha_values, metrics_df[val_rmse_col], **cv_style)
    
    ax2.set_xscale('log')
    ax2.set_xlabel(r"Regularization Penalty ($\alpha$)", fontsize=12)
    ax2.set_ylabel(r"Viscosity RMSE ($\log_{10} \eta$)", fontsize=12)
    ax2.set_box_aspect(1)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc="upper left", frameon=False)
    
    fig.suptitle(r"Global Viscosity Predictive Performance vs Ridge $\alpha$ (5-Fold CV)", fontsize=14, fontweight="bold")
    
    return fig