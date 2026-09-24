"""
Residuals Visualization Module.

Generates plots showing the mean and standard deviation of global viscosity 
prediction residuals versus chemical compounds.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
from glass_dynamics.core.logger import logger

def plot_viscosity_compound_residuals(
    residuals_df: pd.DataFrame,
    figsize: Tuple[float, float] = (10/1.2, 5/1.2)
) -> plt.Figure:
    """
    Plots the global viscosity residual statistics per chemical compound.

    Parameters
    ----------
    residuals_df : pd.DataFrame
        DataFrame output from `compute_global_viscosity_residuals`.
    figsize : Tuple[float, float], default=(8.33, 4.16)
        Overall dimensions of the figure.

    Returns
    -------
    plt.Figure
        The rendered matplotlib figure.
    """
    logger.info("Rendering global viscosity compound residual plot.")
    
    compounds = residuals_df['Compound'].tolist()
    counts = residuals_df['Count'].tolist()
    means = residuals_df['Mean_Residual'].tolist()
    stds = residuals_df['Std_Residual'].tolist()
    
    # Translation table to convert digits to subscript for chemical formulas (e.g. Na2O -> Na₂O)
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    
    xticks_labels = []
    for i, comp in enumerate(compounds):
        formatted_comp = str(comp).translate(subscript_map)
        # Stagger labels to prevent overlapping
        if i % 2 != 0:
            xticks_labels.append(f'\n{formatted_comp}')
        else:
            xticks_labels.append(formatted_comp)
            
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    
    # Horizontal line at 0 residual (perfect prediction)
    ax.axhline(0, c='black', ls='--', alpha=0.5)
    
    # Plot Error bars
    ax.errorbar(
        range(len(compounds)), 
        means, 
        yerr=stds, 
        marker='o', 
        ls='none',
        capsize=5,
        color='tab:blue'
    )
    
    # Bottom X-Axis (Compounds)
    ax.set_xticks(range(len(compounds)))
    ax.set_xticklabels(xticks_labels, rotation=0, ha='center', fontsize=10)
    ax.set_xlim(-1, len(compounds))
    
    # Top X-Axis (Sample Counts - representing total viscosity points)
    axt = ax.secondary_xaxis('top')
    axt.set_xticks(range(len(compounds)))
    axt.set_xticklabels(counts, rotation=90, ha='center', fontsize=10)
    
    # Y-Axis and Grid
    ax.set_ylabel(r'Viscosity Prediction residual ($\log_{10} \eta$)', fontsize=12)
    ax.grid(color='black', ls=':', alpha=0.2)
    
    return fig