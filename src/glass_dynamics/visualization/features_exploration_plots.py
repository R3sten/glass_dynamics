"""
Feature Exploration and Visualization Module.

This module provides tools to explore the feature space of the dataset, 
including correlation matrix visualization and feature metadata extraction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

def plot_pearson_correlation(
    X_train: pd.DataFrame, 
    show_numbers: bool = True, 
    figsize: Tuple[int, int] = (20, 18)
) -> plt.Figure:
    """
    Computes and plots the lower triangle of the Pearson correlation matrix.
    The visual style explicitly matches the project's baseline aesthetics.

    Parameters
    ----------
    X_train : pd.DataFrame
        The training feature matrix.
    show_numbers : bool, default=True
        If True, displays the correlation coefficients inside the heatmap cells.
    figsize : Tuple[int, int], default=(20, 18)
        Dimensions of the output figure. Defaults to a large canvas to 
        accommodate many features comfortably.

    Returns
    -------
    plt.Figure
        The rendered matplotlib figure.
    """
    # Pearson correlation matrix on Train Set
    corr_matrix = X_train.corr()

    # Mask to show just half of the matrix (hide the upper triangle)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Initialize the matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)

    # Drawing the heatmap
    # cmap='coolwarm': Dark Blue = -1 (inverse correlation), 
    # Dark Red = +1 (direct correlation), White = 0 (independent)
    sns.heatmap(
        corr_matrix, 
        mask=mask, 
        cmap='coolwarm', 
        vmax=1.0, 
        vmin=-1.0, 
        center=0,
        square=True, 
        linewidths=.2, 
        cbar_kws={"shrink": .5, "label": "Pearson Correlation Coefficient"},
        annot=show_numbers
    ) 

    # Apply specific formatting to title and ticks
    ax.set_title('Features correlation Heatmap', fontsize=20, pad=20)
    
    # Format tick labels to match the requested rotation and font size
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()

    return fig


def generate_feature_metadata_table(X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the column names of the dataset to extract feature metadata 
    (Type, Aggregator, Chemical Property) and returns a formatted and sorted DataFrame.

    Assumes column names follow the pattern: 'Type_Aggregator_Property_Name'
    (e.g., 'W_Mean_Atomic_Radius').

    Parameters
    ----------
    X_train : pd.DataFrame
        The training feature matrix containing the features as columns.

    Returns
    -------
    pd.DataFrame
        A structured DataFrame containing the parsed feature information, 
        sorted by Type (descending), Aggregator, and Chemical Property.
    """
    selected_features = X_train.columns.tolist()
    table_data = []

    for feat in selected_features:
        # Split the feature name by underscores
        parts = feat.split('_')
        
        # Ensure the feature name has at least Type, Aggregator, and Property
        if len(parts) >= 3:
            feat_type = parts[0]
            aggregator = parts[1]
            # Join the remaining parts to form the full property name
            property_name = " ".join(parts[2:])
        else:
            # Fallback for features that don't match the expected naming convention
            feat_type = parts[0] if len(parts) > 0 else "Unknown"
            aggregator = parts[1] if len(parts) > 1 else "Unknown"
            property_name = "Unknown"

        # Append parsed data to our list
        table_data.append({
            'Type': 'W' if feat_type == 'W' else 'A',
            'Aggregator': aggregator,
            'Chemical Property': property_name.replace('_', ' '),
            'Full Feature Name': feat
        })

    # Convert the list of dictionaries to a pandas DataFrame
    df_table = pd.DataFrame(table_data)

    # Sort the values logically: 
    # Type (W then A -> descending=False/True), Aggregator (ascending), Property (ascending)
    df_table = df_table.sort_values(
        by=['Type', 'Aggregator', 'Chemical Property'], 
        ascending=[False, True, True]
    ).reset_index(drop=True)

    return df_table