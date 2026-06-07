"""
Viscosity Grid Visualization Module.

Handles the generation of Matplotlib grid plots comparing experimental data 
against MYEGA curves. Relies on external modules to supply the pre-computed 
bootstrap confidence bands.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Union
from glass_dynamics.core.logger import logger
from glass_dynamics.analysis.myega_bootstrap import extract_myega_confidence_bands

def plot_viscosity_grids_bootstrap(
    viscosity_df: pd.DataFrame,
    distributions_dict: Dict[int, pd.DataFrame],
    names_dict: Dict[int, str],
    save_dir: Union[str, Path],
    confidence: float = 95.0,
    num_rows: int = 5,
    num_cols: int = 4
):
    """
    Plots grids of experimental vs predicted viscosity curves with bootstrap uncertainty bands.
    """
    test_ids = list(distributions_dict.keys())
    logger.info(f"Rendering bootstrap prediction grids for {len(test_ids)} glasses.")

    custom_style = {
        'font.family': 'serif',
        'font.serif': 'DejaVu Serif',
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'figure.dpi': 150,
        'mathtext.fontset': 'dejavuserif',
    }
    
    chunk_size = num_rows * num_cols
    base_size = 3
    
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(custom_style):
        for i in range(0, len(test_ids), chunk_size):
            chunk_ids = test_ids[i:i + chunk_size]
            fig_idx = (i // chunk_size) + 1
            current_rows = int(np.ceil(len(chunk_ids) / num_cols))
            
            fig, axes = plt.subplots(
                current_rows, num_cols, 
                figsize=(base_size * num_cols, base_size * current_rows),
                squeeze=False
            )
            
            for ax_idx, ax in enumerate(axes.flatten()):
                if ax_idx >= len(chunk_ids):
                    ax.set_visible(False)
                    continue
                    
                glass_id = chunk_ids[ax_idx]
                
                # Experimental Data
                df_exp = viscosity_df[viscosity_df['ID'] == glass_id]
                T_exp = df_exp['T'].to_numpy()
                y_exp = df_exp['log_visc'].to_numpy()
                
                # Smooth temperature array spanning the experimental range
                T_smooth = np.linspace(min(T_exp), max(T_exp), 200)
                
                # Fetch the N bootstrap predictions for this specific glass
                glass_dists = distributions_dict[glass_id]

                # Compute the final curves using the logical module
                mean_curve, lower_band, upper_band = extract_myega_confidence_bands(
                    T_smooth, glass_dists, confidence
                )

                # --- DRAWING ---
                # Error Bands
                ax.plot(T_smooth, lower_band, ls='--', c='tab:red', alpha=0.8, lw=1.5)
                ax.plot(T_smooth, upper_band, ls='--', c='tab:red', alpha=0.8, lw=1.5)
                
                # Experimental points
                ax.plot(T_exp, y_exp, marker='o', ls='none', 
                        markeredgecolor='black', markerfacecolor='steelblue', alpha=0.9)

                # Annotations
                glass_name = names_dict.get(glass_id, str(glass_id))
                ax.set_title(glass_name, fontsize=8, pad=5)
                ax.text(0.95, 0.95, str(glass_id), transform=ax.transAxes, 
                        fontsize=10, fontweight='bold', va='top', ha='right')
                
                row_idx, col_idx = divmod(ax_idx, num_cols)
                if row_idx == current_rows - 1:
                    ax.set_xlabel('$T$ [K]', fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(r'$\log_{10} (\eta)$', fontsize=10)
            
            plt.tight_layout()
            
            file_path = output_dir / f'all_liquids_{fig_idx}.pdf'
            fig.savefig(file_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
            
            if fig_idx <= 1:
                plt.show(fig)
            else:
                plt.close(fig)