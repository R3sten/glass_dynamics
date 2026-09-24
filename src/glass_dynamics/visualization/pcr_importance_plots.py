"""
PCR Importance Visualization Module.

Generates the signature 'S-curve' sorted weights plot specifically adapted 
for Principal Components. Allows annotation of the most important components 
along with their top representative physical features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from glass_dynamics.core.logger import logger

def plot_sorted_pc_importance(
    pc_dict: Dict[str, pd.DataFrame],
    top_n: int = 0,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Plot the algebraically sorted raw weights (S-curve) of Principal Components 
    for each target side-by-side.

    Parameters
    ----------
    pc_dict : Dict[str, pd.DataFrame]
        Dictionary of DataFrames containing PC representatives and weights, 
        as output by `extract_pc_representatives`.
    top_n : int, default=0
        Number of most important PCs (by absolute magnitude) to annotate.
        Set to 0 to only plot the raw S-curve without labels.
    figsize : Tuple[int, int], default=(15, 5)
        Overall dimensions of the generated figure.

    Returns
    -------
    plt.Figure
        The rendered matplotlib figure.
    """
    target_names = list(pc_dict.keys())
    n_targets = len(target_names)
    
    logger.info(f"Rendering sorted PC importance S-curves for {n_targets} targets.")
    
    fig, axes = plt.subplots(1, n_targets, figsize=figsize, layout='constrained')
    if n_targets == 1:
        axes = [axes]
        
    for ax, target, panel_label in zip(axes, target_names, ['(a)', '(b)', '(c)']):
        df = pc_dict[target]
        n_pcs = len(df)
        
        # Sort PC weights algebraically to create the S-curve
        df_algebraic = df.sort_values(by='Regression_Weight', ascending=True).reset_index(drop=True)
        
        ax.plot(
            df_algebraic.index, 
            df_algebraic['Regression_Weight'], 
            marker='o', 
            ms=5, 
            color='black', 
            linestyle='-', 
            linewidth=1.2
        )
        
        # Apply dynamic margins to preserve dynamic y-axis range
        ax.margins(x=0.15, y=0.25)
        
        if top_n > 0:
            top_pcs = df.sort_values(by='Abs_Weight', ascending=False).head(top_n)
            
            for i, (_, row) in enumerate(top_pcs.iterrows()):
                pc_name = row['Component']
                rep_feature = row['Top_Representative_Feature']
                weight = row['Regression_Weight']
                
                # Multi-line label: PC name and its physical representative
                display_label = f"{pc_name}\n({rep_feature})"
                plot_idx = df_algebraic[df_algebraic['Component'] == pc_name].index[0]
                
                # Robust Annotation Logic via Screen Points
                x_offset_pts = 35 if plot_idx < (n_pcs / 2) else -35
                y_sign = 1 if weight > 0 else -1
                y_offset_pts = y_sign * (15 + (i % 2) * 25)
                
                ax.annotate(
                    display_label,
                    xy=(plot_idx, weight),
                    xytext=(x_offset_pts, y_offset_pts),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', lw=1, color='black'),
                    fontsize=8,
                    ha='left' if x_offset_pts > 0 else 'right',
                    va='center'
                )
                
        # Axes formatting
        ax.set_ylabel(r'$w_{PC}$', fontsize=12) if ax == axes[0] else ax.set_ylabel('')
        ax.set_xlabel(r'PC (sorted index)', fontsize=12)
        ax.set_title(f"{target}", fontsize=12, fontweight='bold')
        ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
        ax.text(0.05, 0.92, panel_label, transform=ax.transAxes, fontsize=12)

    return fig