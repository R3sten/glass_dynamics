"""
Density and Correlation Plotting Module.

Generates a 2D histogram (density plot) comparing true vs. predicted values,
complete with an inset 1D histogram of the prediction residuals, matching
the aesthetic standards of the Cassar methodology.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import Tuple, Optional
from glass_dynamics.core.logger import logger

def plot_viscosity_density_correlation(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    bin_size: float = 0.3,
    axis_limits: Tuple[float, float] = (-3, 16),
    residual_limits: Tuple[float, float] = (-2.5, 2.5),
    label_name: str = r'$\log_{10} (\eta)$'
) -> plt.Figure:
    """
    Plots a 2D density histogram of reported vs. predicted values, including 
    an inset plot showing the distribution of prediction residuals.

    Parameters
    ----------
    y_true : np.ndarray
        Array of ground truth values (e.g., experimental viscosities).
    y_pred : np.ndarray
        Array of predicted values.
    bin_size : float, default=0.3
        The size of the square bins for the 2D histogram.
    axis_limits : Tuple[float, float], default=(-3, 16)
        The minimum and maximum limits for both the X and Y main axes.
    residual_limits : Tuple[float, float], default=(-2.5, 2.5)
        The minimum and maximum limits for the X-axis of the inset residual plot.
    label_name : str, default=r'$\log_{10} (\eta)$'
        The mathematical label to display on the main axes.

    Returns
    -------
    plt.Figure
        The rendered matplotlib figure.
    """
    logger.info("Rendering 2D density correlation plot with residual inset.")

    # Define the custom aesthetic parameters localized only to this plot
    # This prevents overriding the global matplotlib state in the user's notebook
    custom_style = {
        'font.family': 'serif',
        'font.serif': 'DejaVu Serif',
        'axes.formatter.limits': [-2, 5],
        'axes.formatter.useoffset': False,
        'axes.formatter.use_mathtext': True,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'legend.framealpha': 1,
        'legend.edgecolor': 'k',
        'figure.figsize': [4, 4],
        'figure.dpi': 150,
        'errorbar.capsize': 5,
        'mathtext.fontset': 'dejavuserif',
    }

    with plt.rc_context(custom_style):
        fig, ax = plt.subplots(ncols=1, nrows=1)

        # 1. Compute 2D Histogram boundaries based on actual data
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        
        # Create bins ensuring they cover the entire data range
        edges = np.arange(min_val, max_val + bin_size, bin_size)

        # 2. Compute 2D Histogram density
        H, xedges, yedges = np.histogram2d(y_true, y_pred, bins=(edges, edges))
        H = H.T  # Transpose needed because histogram2d returns (x_bins, y_bins)

        # 3. Plot the 2D density map using logarithmic normalization
        X, Y = np.meshgrid(xedges, yedges)
        cm = ax.pcolormesh(X, Y, H, cmap='viridis_r', norm=LogNorm())
        
        # Add a colorbar for the density
        cb = fig.colorbar(cm, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('Density')

        # 4. Plot the ideal perfect-prediction baseline (y = x)
        ideal_min = axis_limits[0] - 5
        ideal_max = axis_limits[1] + 5
        ax.plot(
            [ideal_min, ideal_max], 
            [ideal_min, ideal_max],
            ls='-', c='k', lw=0.7, alpha=0.7
        )

        # Main axes formatting
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)
        ax.set_xlabel(rf'Reported {label_name}')
        ax.set_ylabel(rf'Predicted {label_name}')

        # 5. Create Inset Plot for Residuals
        ax_inset = inset_axes(
            ax,
            width='33%',
            height='33%',
            loc=2,  # Location 2 is 'upper left'
        )

        # Inset axes formatting
        ax_inset.set_xlabel('Pred. residual')
        ax_inset.set_ylabel('Frequency', fontsize=10)
        ax_inset.tick_params(axis='both', which='major', labelsize=8)
        ax_inset.yaxis.tick_right()
        ax_inset.yaxis.set_ticks_position('both')
        ax_inset.yaxis.set_label_position('right')

        # Calculate prediction residuals
        residuals = y_true - y_pred
        residual_bins = np.linspace(-5, 5, 100)
        
        # Plot 1D histogram of residuals
        ax_inset.hist(
            residuals,
            bins=residual_bins,
            fc='gray',
            align='mid',
            log=False,
            ec='none',
            histtype='stepfilled', # Smoother filled appearance
            alpha=0.7,
        )

        # Configure limits and specific ticks for the inset
        ax_inset.set_xlim(residual_limits)
        ax_inset.xaxis.set_ticks([-2, -1, 0, 1, 2])

    return fig