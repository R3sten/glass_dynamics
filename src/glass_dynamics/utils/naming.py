"""
Chemical Naming Utility Module.

Provides functions to translate raw chemical composition fractions into 
human-readable chemical formulas using proper subscript typography.
"""

import pandas as pd
from typing import Dict, List

def generate_glass_names(viscosity_df: pd.DataFrame, compounds: List[str]) -> Dict[int, str]:
    """
    Converts molar fractions of chemical compounds into readable string names 
    with proper subscripts (e.g., '50SiO₂.50Na₂O').

    Parameters
    ----------
    viscosity_df : pd.DataFrame
        The dataset containing the compositions and 'ID' column.
    compounds : List[str]
        List of column names representing the chemical compounds.

    Returns
    -------
    Dict[int, str]
        A dictionary mapping each glass 'ID' to its formatted chemical name.
    """
    # Translation table for standard numbers to Unicode subscripts
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    names_dict = {}
    
    # Iterate over unique glass IDs
    for glass_id in viscosity_df['ID'].unique():
        # Get the first row corresponding to this ID (composition is constant per ID)
        glass_data = viscosity_df[viscosity_df['ID'] == glass_id].iloc[0]
        
        # Extract compounds with a molar fraction greater than 0
        active_compounds = {c: glass_data[c] for c in compounds if glass_data[c] > 0}
        
        # Build the name: Multiply by 100, format to 0 decimals, and translate compound names
        name_parts = [
            f"{fraction * 100:.0f}{comp.translate(subscript_map)}" 
            for comp, fraction in active_compounds.items()
        ]
        
        names_dict[glass_id] = '.'.join(name_parts)
        
    return names_dict