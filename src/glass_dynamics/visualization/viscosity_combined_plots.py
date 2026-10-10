"""
Combined Viscosity Grid Visualization Module.

Generates Matplotlib grid plots comparing experimental data against the 
deterministic MYEGA curve, overlaid with dashed bootstrap confidence bands.
Subplots are perfectly square.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Union, Any
from glass_dynamics.core.logger import logger
from glass_dynamics.analysis.myega_bootstrap import extract_myega_confidence_bands

def compute_myega_curve(T: np.ndarray, log_inf: float, Tg: float, m: float) -> np.ndarray:
    """Computes a single MYEGA viscosity curve."""
    T = np.maximum(T, 1e-9)
    term1 = (12.0 - log_inf) * (Tg / T)
    term2 = np.exp((m / (12.0 - log_inf) - 1.0) * (Tg / T - 1.0))
    return log_inf + term1 * term2

def plot_viscosity_grids_combined(
    viscosity_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    distributions_dict: Dict[Any, pd.DataFrame],
    names_dict: Dict[int, str],
    save_dir: Union[str, Path],
    confidence: float = 95.0,
    num_rows: int = 5,
    num_cols: int = 4
):
    """
    Plots square grids of experimental vs predicted deterministic viscosity curves,
    overlaid with bootstrap uncertainty bounds (dashed lines).
    """
    test_ids = predictions_df.index.tolist()
    logger.info(f"Rendering combined deterministic+bootstrap grids for {len(test_ids)} glasses.")

    tg_col = next(c for c in predictions_df.columns if 'Tg' in c)
    m_col = next(c for c in predictions_df.columns if 'm' in c)
    inf_col = next(c for c in predictions_df.columns if 'inf' in c.lower())

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
    base_size = 3.5 
    
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
                ax.set_box_aspect(1) # Forza riquadro quadrato
                
                # 1. Experimental Data
                df_exp = viscosity_df[viscosity_df['ID'] == glass_id]
                T_exp = df_exp['T'].to_numpy()
                y_exp = df_exp['log_visc'].to_numpy()
                
                if len(T_exp) > 0:
                    T_smooth = np.linspace(min(T_exp) * 0.95, max(T_exp) * 1.05, 100)
                else:
                    T_smooth = np.linspace(500, 1500, 100)
                
                # 2. Deterministic Curve (Solid Red Line)
                pred_row = predictions_df.loc[glass_id]
                y_det = compute_myega_curve(
                    T_smooth, pred_row[inf_col], pred_row[tg_col], pred_row[m_col]
                )
                
                # 3. Bootstrap Bounds (Dashed Lines)
                glass_dists = distributions_dict[glass_id]
                _, lower_band, upper_band = extract_myega_confidence_bands(
                    T_smooth, glass_dists, confidence
                )

                # --- DRAWING ---
                # A. Bootstrap Bands (Dashed)
                ax.plot(T_smooth, lower_band, ls='--', c='tab:red', alpha=0.7, lw=1.2)
                ax.plot(T_smooth, upper_band, ls='--', c='tab:red', alpha=0.7, lw=1.2)
                
                # B. Deterministic Curve (Solid)
                ax.plot(T_smooth, y_det, ls='-', c='tab:red', lw=2.0, label='Predicted (Det.)')
                
                # C. Experimental Points (Blue Scatter)
                ax.plot(T_exp, y_exp, marker='o', ls='none', 
                        markeredgecolor='black', markerfacecolor='steelblue', alpha=0.9, label='Exp.')

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
            
            file_path = output_dir / f'test_liquids_combined_{fig_idx}.pdf'
            fig.savefig(file_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
            
            if fig_idx <= 1:
                plt.show(fig)
            else:
                plt.close(fig)