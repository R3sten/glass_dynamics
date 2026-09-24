"""
Elastic Net 3D Visualization Module.

Generates 3D surface plots to visualize the model performance (R and R^2) 
across the 2D parameter space (alpha and l1_ratio) of the Elastic Net model.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from glass_dynamics.core.logger import logger

def plot_elastic_net_3d_performance(
    target_results: Dict[str, np.ndarray],
    target_name: str,
    alphas: np.ndarray,
    l1_ratios: np.ndarray,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot 3D surfaces for Test R and Test R^2 as a function of alpha and l1_ratio.

    Parameters
    ----------
    target_results : Dict[str, np.ndarray]
        Dictionary containing the 2D metric arrays for a specific target.
    target_name : str
        Name of the target variable for the plot title.
    alphas : np.ndarray
        Array of regularization strengths (alpha) used in the grid.
    l1_ratios : np.ndarray
        Array of l1_ratios used in the grid.
    figsize : Tuple[int, int], default=(14, 6)
        Overall dimensions of the figure.

    Returns
    -------
    plt.Figure
        The rendered matplotlib figure.
    """
    logger.info(f"Rendering 3D performance surfaces for {target_name}.")
    
    fig = plt.figure(figsize=figsize)
    
    # We use log10 for alphas to ensure the 3D surface is evenly distributed
    # (otherwise, the plot gets squished heavily towards 0 due to log spacing)
    log_alphas = np.log10(alphas)
    
    # Create the 2D meshgrid for plotting
    A_mesh, L_mesh = np.meshgrid(log_alphas, l1_ratios, indexing='ij')

    # --- Left Plot: Test R ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(
        A_mesh, L_mesh, target_results["test_R"], 
        cmap='viridis', edgecolor='none', alpha=0.9
    )
    ax1.set_title(r"Test Pearson Correlation ($R$)", fontsize=12, pad=10)
    ax1.set_xlabel(r"$\log_{10}(\alpha)$", fontsize=10, labelpad=10)
    ax1.set_ylabel(r"$L_1$ Ratio", fontsize=10, labelpad=10)
    ax1.set_zlabel(r"$R$", fontsize=10, labelpad=10)
    
    # Add a colorbar for better value readability
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)

    # --- Right Plot: Test R^2 ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(
        A_mesh, L_mesh, target_results["test_R2"], 
        cmap='plasma', edgecolor='none', alpha=0.9
    )
    ax2.set_title(r"Test Coefficient of Determination ($R^2$)", fontsize=12, pad=10)
    ax2.set_xlabel(r"$\log_{10}(\alpha)$", fontsize=10, labelpad=10)
    ax2.set_ylabel(r"$L_1$ Ratio", fontsize=10, labelpad=10)
    ax2.set_zlabel(r"$R^2$", fontsize=10, labelpad=10)
    
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)

    # Main title
    fig.suptitle(f"Elastic Net 2D Grid Search: {target_name}", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    return fig