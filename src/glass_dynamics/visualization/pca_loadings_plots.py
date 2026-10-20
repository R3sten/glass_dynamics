"""
PCA Loadings Visualization Module.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
from glass_dynamics.core.logger import logger

def plot_top_pc_loadings(
    loadings_df: pd.DataFrame, 
    pc_index: int, 
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plots a horizontal bar chart of the top PCA loadings.
    """
    logger.info(f"Rendering loadings plot for PC{pc_index}.")
    
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    
    # Invertiamo l'ordine per il plot orizzontale (vogliamo la feature più forte in alto)
    plot_df = loadings_df.sort_values(by='Abs_Loading', ascending=True)
    
    # Colori: Rosso per pesi positivi, Acciaio per pesi negativi
    colors = ['tab:red' if val > 0 else 'steelblue' for val in plot_df['Loading']]
    
    ax.barh(plot_df['Feature'], plot_df['Loading'], color=colors, edgecolor='black', alpha=0.85)
    
    ax.set_xlabel('PC Loading (Weight)', fontsize=11)
    ax.set_title(f'Top {len(loadings_df)} Features for PC{pc_index}', fontsize=12, fontweight='bold')
    ax.grid(axis='x', linestyle=':', alpha=0.7)
    
    # Linea marcata sullo zero per separare visivamente i contributi
    ax.axvline(0, color='black', linewidth=1.2)
    
    return fig