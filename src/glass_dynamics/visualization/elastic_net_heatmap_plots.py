"""
Elastic Net CV Heatmap Visualization Module (High Contrast).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple
from glass_dynamics.core.logger import logger

def plot_cv_dual_heatmap(
    r2_df: pd.DataFrame,
    rmse_df: pd.DataFrame,
    title_prefix: str,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plots a dual 2D heatmap (CV R^2 and CV RMSE) for Elastic Net hyperparameters.
    Implements dynamic contrast clipping and 2-significant-digit formatting.
    """
    logger.info(f"Rendering high-contrast CV dual heatmap for {title_prefix}.")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, layout='constrained')
    
    alphas = r2_df.index.to_numpy()
    l1_ratios = r2_df.columns.to_numpy()
    
    tick_step_x = max(1, len(l1_ratios) // 6)
    tick_step_y = max(1, len(alphas) // 8)
    
    # Asse X: L1 Ratio (2 decimali fissi)
    x_ticks = np.arange(0, len(l1_ratios), tick_step_x)
    x_labels = [f"{x:.2f}" for x in l1_ratios[::tick_step_x]]
    
    # Asse Y: Alpha (2 cifre significative usando il format '.2g')
    y_ticks = np.arange(0, len(alphas), tick_step_y)
    y_labels = [f"{y:.2g}" for y in alphas[::tick_step_y]]
    
    def setup_ax(ax):
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel(r"$L_1$ Ratio (Ridge $\rightarrow$ Lasso)", fontsize=11)
        ax.set_ylabel(r"Regularization Penalty $\alpha$", fontsize=11)
    
    # --- Plot 1: R^2 Heatmap (Dynamic Contrast) ---
    r2_vals = r2_df.to_numpy()
    # Taglia il peggior 25% dei modelli (li satura al blu scuro) per espandere il contrasto sui modelli buoni
    r2_vmin = max(0.0, np.nanpercentile(r2_vals, 25))
    r2_vmax = np.nanmax(r2_vals)
    
    cax1 = ax1.imshow(r2_vals, cmap='viridis', aspect='auto', origin='lower', vmin=r2_vmin, vmax=r2_vmax)
    ax1.set_title(rf"CV $R^2$ (Higher is Better)", fontsize=12)
    setup_ax(ax1)
    
    cbar1 = fig.colorbar(cax1, ax=ax1, pad=0.02)
    cbar1.set_label('Cross-Validation R²', rotation=270, labelpad=15, fontsize=11)
    
    # --- Plot 2: RMSE Heatmap (Dynamic Contrast) ---
    rmse_vals = rmse_df.to_numpy()
    rmse_vmin = np.nanmin(rmse_vals)
    # Taglia il peggior 25% dei modelli (quelli con errore alto satureranno in giallo)
    rmse_vmax = np.nanpercentile(rmse_vals, 75)
    
    cax2 = ax2.imshow(rmse_vals, cmap='viridis', aspect='auto', origin='lower', vmin=rmse_vmin, vmax=rmse_vmax)
    ax2.set_title(rf"CV RMSE (Lower is Better)", fontsize=12)
    setup_ax(ax2)
    
    cbar2 = fig.colorbar(cax2, ax=ax2, pad=0.02)
    cbar2.set_label('Cross-Validation RMSE', rotation=270, labelpad=15, fontsize=11)
    
    fig.suptitle(f"{title_prefix}: 5-Fold Cross Validation", fontsize=14, fontweight="bold")
    
    return fig