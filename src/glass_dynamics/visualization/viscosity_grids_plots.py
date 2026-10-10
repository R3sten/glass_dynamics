"""
Deterministic Viscosity Grid Visualization Module.

Handles the generation of Matplotlib grid plots comparing experimental data 
against the deterministic MYEGA curves predicted by the model, without uncertainty bands.
Features square subplots for consistent aesthetics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Union
from glass_dynamics.core.logger import logger

def compute_myega_curve(T: np.ndarray, log_inf: float, Tg: float, m: float) -> np.ndarray:
    """Computes a single MYEGA viscosity curve."""
    T = np.maximum(T, 1e-9)
    term1 = (12.0 - log_inf) * (Tg / T)
    term2 = np.exp((m / (12.0 - log_inf) - 1.0) * (Tg / T - 1.0))
    return log_inf + term1 * term2

def plot_viscosity_grids(
    viscosity_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    names_dict: Dict[int, str],
    save_dir: Union[str, Path],
    num_rows: int = 5,
    num_cols: int = 4
):
    """
    Plots grids of experimental vs predicted viscosity curves (deterministic).
    Each subplot is forced to be perfectly square.
    
    Parameters
    ----------
    viscosity_df : pd.DataFrame
        Raw dataset containing 'ID', 'T', and 'log_visc'.
    predictions_df : pd.DataFrame
        DataFrame indexed by Glass ID, containing the predicted physical 
        parameters (e.g., 'Tg', 'm', 'log_eta_inf').
    names_dict : Dict[int, str]
        Mapping from Glass ID to chemical composition string.
    save_dir : str or Path
        Directory to save the PDF grids.
    num_rows : int, default=5
    num_cols : int, default=4
    """
    test_ids = predictions_df.index.tolist()
    logger.info(f"Rendering deterministic prediction grids for {len(test_ids)} glasses.")

    # Find the column names dynamically just in case they vary slightly
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
    base_size = 3  # Leggermente più grande per accomodare il riquadro quadrato
    
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
                
                # Forzare il riquadro ad essere quadrato
                ax.set_box_aspect(1)
                
                # Experimental Data
                df_exp = viscosity_df[viscosity_df['ID'] == glass_id]
                T_exp = df_exp['T'].to_numpy()
                y_exp = df_exp['log_visc'].to_numpy()
                
                # Estrai i parametri predetti per questo vetro
                pred_row = predictions_df.loc[glass_id]
                tg_pred = pred_row[tg_col]
                m_pred = pred_row[m_col]
                inf_pred = pred_row[inf_col]
                
                # Smooth temperature array spanning the experimental range
                if len(T_exp) > 0:
                    T_smooth = np.linspace(min(T_exp) * 0.95, max(T_exp) * 1.05, 100)
                else:
                    T_smooth = np.linspace(500, 1500, 100) # Fallback if missing
                
                # Calcola la curva predetta dal modello
                y_pred = compute_myega_curve(T_smooth, inf_pred, tg_pred, m_pred)

                # --- DRAWING ---
                # Predicted Curve
                ax.plot(T_smooth, y_pred, ls='-', c='tab:red', lw=2.0, label='Predicted')
                
                # Experimental points
                ax.plot(T_exp, y_exp, marker='o', ls='none', 
                        markeredgecolor='black', markerfacecolor='steelblue', alpha=0.9, label='Exp')

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
            
            file_path = output_dir / f'test_liquids_deterministic_{fig_idx}.pdf'
            fig.savefig(file_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
            
            if fig_idx <= 1:
                plt.show(fig)
            else:
                plt.close(fig)